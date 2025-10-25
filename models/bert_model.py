"""
BERT-based Abuse Detection Model

This module implements a BERT-based model for detecting harassment and abuse
in social media text using transformer architecture.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertModel, 
    BertConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTAbuseDataset(Dataset):
    """Dataset class for BERT-based abuse detection."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels (0: normal, 1: harassment)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTAbuseDetector(nn.Module):
    """BERT-based model for abuse detection."""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 freeze_bert: bool = False):
        """
        Initialize BERT abuse detector.
        
        Args:
            model_name: Name of the BERT model to use
            num_classes: Number of output classes
            dropout_rate: Dropout rate for the classifier
            freeze_bert: Whether to freeze BERT parameters
        """
        super(BERTAbuseDetector, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.config = BertConfig.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for classification
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class BERTAbuseDetectorTrainer:
    """Trainer class for BERT abuse detection model."""
    
    def __init__(self, 
                 model: BERTAbuseDetector,
                 tokenizer: BertTokenizer,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize trainer.
        
        Args:
            model: BERT abuse detection model
            tokenizer: BERT tokenizer
            device: Device to run training on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader: DataLoader, optimizer, scheduler, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader: DataLoader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              num_epochs: int = 3,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500,
              save_path: Optional[str] = None):
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_path: Path to save the trained model
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_dataloader, optimizer, scheduler, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_dataloader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
    
    def predict(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to predict on
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        predictions = []
        
        # Create dataset and dataloader
        dataset = BERTAbuseDataset(texts, [0] * len(texts), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Store predictions
                for i in range(len(preds)):
                    predictions.append({
                        'prediction': preds[i].item(),
                        'probability': probs[i].cpu().numpy().tolist(),
                        'confidence': torch.max(probs[i]).item()
                    })
        
        return predictions
    
    def save_model(self, path: str):
        """Save the trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model and return comprehensive metrics.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }


def create_bert_model(config: Dict[str, Any]) -> Tuple[BERTAbuseDetector, BertTokenizer]:
    """
    Create BERT model and tokenizer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = config.get('model_name', 'bert-base-uncased')
    num_classes = config.get('num_classes', 2)
    dropout_rate = config.get('dropout_rate', 0.3)
    freeze_bert = config.get('freeze_bert', False)
    
    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Create model
    model = BERTAbuseDetector(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_bert=freeze_bert
    )
    
    return model, tokenizer
