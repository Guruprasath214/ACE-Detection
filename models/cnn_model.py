"""
CNN-based Text Classification Model

This module implements a CNN-based model for detecting harassment and abuse
in social media text using convolutional neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Tuple, Optional, Any
import json
from pathlib import Path
import logging
from collections import Counter
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNTextDataset(Dataset):
    """Dataset class for CNN-based text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int = 100):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels (0: normal, 1: harassment)
            vocab: Vocabulary dictionary mapping words to indices
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.unk_token = vocab.get('<UNK>', 0)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, self.unk_token) for token in tokens]
        
        # Pad or truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([0] * (self.max_length - len(token_ids)))
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CNNTextClassifier(nn.Module):
    """CNN-based model for text classification."""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 300,
                 num_filters: int = 100,
                 filter_sizes: List[int] = [3, 4, 5],
                 num_classes: int = 2,
                 dropout_rate: float = 0.5,
                 pretrained_embeddings: Optional[np.ndarray] = None):
        """
        Initialize CNN text classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters for each filter size
            filter_sizes: List of filter sizes for convolution
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            pretrained_embeddings: Pretrained word embeddings (optional)
        """
        super(CNNTextClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (filter_size, embedding_dim))
            for filter_size in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, input_ids):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            
        Returns:
            Logits for classification
        """
        # Embedding layer
        embedded = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]
        embedded = embedded.unsqueeze(1)      # [batch_size, 1, seq_length, embedding_dim]
        
        # Convolutional layers
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, new_seq_length, 1]
            pooled = F.max_pool2d(conv_out, (conv_out.size(2), 1))  # [batch_size, num_filters, 1, 1]
            conv_outputs.append(pooled.squeeze(3).squeeze(2))  # [batch_size, num_filters]

        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        
        # Dropout and classification
        concatenated = self.dropout(concatenated)
        logits = self.fc(concatenated)
        
        return logits


class CNNTextClassifierTrainer:
    """Trainer class for CNN text classification model."""
    
    def __init__(self, 
                 model: CNNTextClassifier,
                 vocab: Dict[str, int],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize trainer.
        
        Args:
            model: CNN text classification model
            vocab: Vocabulary dictionary
            device: Device to run training on
        """
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader: DataLoader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
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
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
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
              num_epochs: int = 10,
              learning_rate: float = 0.001,
              save_path: Optional[str] = None):
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save the trained model
        """
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_dataloader, optimizer, criterion)
            
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
        dataset = CNNTextDataset(texts, [0] * len(texts), self.vocab)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
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
            'vocab': self.vocab,
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
        self.vocab = checkpoint['vocab']
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
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
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


def build_vocab(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """
    Build vocabulary from texts.
    
    Args:
        texts: List of texts
        min_freq: Minimum frequency for a word to be included in vocabulary
        
    Returns:
        Vocabulary dictionary mapping words to indices
    """
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Create vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


def create_cnn_model(config: Dict[str, Any], vocab: Dict[str, int]) -> CNNTextClassifier:
    """
    Create CNN model from configuration.
    
    Args:
        config: Configuration dictionary
        vocab: Vocabulary dictionary
        
    Returns:
        CNN text classification model
    """
    vocab_size = len(vocab)
    embedding_dim = config.get('embedding_dim', 300)
    num_filters = config.get('num_filters', 100)
    filter_sizes = config.get('filter_sizes', [3, 4, 5])
    num_classes = config.get('num_classes', 2)
    dropout_rate = config.get('dropout_rate', 0.5)
    
    model = CNNTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    return model


def load_pretrained_embeddings(embedding_path: str, vocab: Dict[str, int], embedding_dim: int = 300) -> np.ndarray:
    """
    Load pretrained word embeddings.
    
    Args:
        embedding_path: Path to pretrained embeddings file
        vocab: Vocabulary dictionary
        embedding_dim: Dimension of embeddings
        
    Returns:
        Embedding matrix
    """
    embeddings = np.random.normal(0, 0.1, (len(vocab), embedding_dim))
    
    try:
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                    embedding = np.array([float(x) for x in parts[1:]])
                    embeddings[vocab[word]] = embedding
    except FileNotFoundError:
        logger.warning(f"Pretrained embeddings file not found: {embedding_path}")
    
    return embeddings
