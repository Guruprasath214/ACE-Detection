"""

Deep Emoji Model for Text Classification

This module implements a deep learning model specialized for emoji sentiment
analysis and emoji-enhanced text classification for harassment detection.

"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import re
import emoji
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from pathlib import Path
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmojiFeatureExtractor:
    """Extract emoji-related features from text."""

    def __init__(self):
        """Initialize emoji feature extractor."""
        # Emoji sentiment mapping (simplified version)
        self.emoji_sentiment = {
            # Positive emojis
            '😀': 0.8, '😃': 0.8, '😄': 0.8, '😁': 0.8, '😆': 0.8, '😅': 0.7, '😂': 0.6, '🤣': 0.7,
            '😊': 0.7, '😇': 0.6, '🙂': 0.5, '🙃': 0.4, '😉': 0.5, '😌': 0.5, '😍': 0.9, '🥰': 0.9,
            '😘': 0.8, '😗': 0.7, '😙': 0.7, '😚': 0.7, '😋': 0.6, '😛': 0.5, '😝': 0.6, '😜': 0.6,
            '🤪': 0.5, '🤨': 0.2, '🧐': 0.3, '🤓': 0.4, '😎': 0.6, '🤩': 0.8, '🥳': 0.8, '😏': 0.4,
            '😒': -0.3, '😞': -0.5, '😔': -0.5, '😟': -0.4, '😕': -0.3, '🙁': -0.5, '☹️': -0.6, '😣': -0.5,
            '😖': -0.6, '😫': -0.6, '😩': -0.7, '🥺': -0.4, '😢': -0.7, '😭': -0.8, '😤': -0.6, '😠': -0.7,
            '😡': -0.8, '🤬': -0.9, '🤯': -0.5, '😳': -0.3, '🥵': -0.4, '🥶': -0.3, '😱': -0.6, '😨': -0.6,
            '😰': -0.6, '😥': -0.5, '😓': -0.4, '🤗': 0.7, '🤔': 0.2, '🤭': 0.4, '🤫': 0.3, '🤥': -0.4,
            '😶': 0.0, '😐': 0.0, '😑': -0.1, '😬': -0.3, '🙄': -0.4, '😯': -0.2, '😦': -0.4, '😧': -0.5,
            '😮': -0.2, '😲': -0.3, '🥱': -0.2, '😴': -0.1, '🤤': 0.3, '😪': -0.2, '😵': -0.5, '🤐': 0.1,
            '🥴': -0.3, '🤢': -0.7, '🤮': -0.8, '🤧': -0.4, '😷': -0.2, '🤒': -0.5, '🤕': -0.6, '🤑': 0.5,
            '🤠': 0.4, '😈': -0.3, '👿': -0.8, '👹': -0.7, '👺': -0.7, '🤡': -0.2, '💩': -0.6, '👻': -0.3,
            '💀': -0.5, '☠️': -0.7, '👽': 0.1, '👾': 0.2, '🤖': 0.3, '🎃': -0.2, '😺': 0.6, '😸': 0.7,
            '😹': 0.6, '😻': 0.8, '😼': 0.5, '😽': 0.7, '🙀': -0.5, '😿': -0.6, '😾': -0.7,

            # Hearts
            '❤️': 0.9, '🧡': 0.8, '💛': 0.8, '💚': 0.8, '💙': 0.8, '💜': 0.8, '🖤': -0.3, '🤍': 0.7, '🤎': 0.7,

            # Gestures
            '👍': 0.6, '👎': -0.6, '👌': 0.5, '✌️': 0.7, '🤞': 0.6, '🤟': 0.7, '🤘': 0.6, '🤙': 0.5, '👈': 0.1,
            '👉': 0.1, '👆': 0.2, '🖕': -0.9, '👇': 0.1, '☝️': 0.3, '👋': 0.5, '🤚': 0.2, '🖐️': 0.3, '✋': 0.1,
            '🖖': 0.4, '👏': 0.7, '🙌': 0.8, '🤲': 0.4, '🤝': 0.6, '🙏': 0.3, '✍️': 0.2, '💅': 0.4, '🤳': 0.3,
            '💪': 0.6, '🦾': 0.5, '🦿': 0.2, '🦵': 0.1, '🦶': 0.1,

            # Symbols
            '💯': 0.8, '🔥': 0.7, '⭐': 0.6, '✨': 0.7, '💫': 0.6, '💥': 0.5, '💢': -0.7, '💦': -0.2, '💨': 0.3,
            '🕳️': -0.3, '💣': -0.8, '💬': 0.2, '👁️‍🗨️': 0.1, '🗨️': 0.2, '🗯️': -0.4, '💭': 0.3, '💤': -0.1
        }

        # Harassment-related emoji patterns
        self.harassment_emojis = {
            'threatening': ['🤬', '😡', '💢', '💣', '🔪', '🗡️', '🔫', '💥', '☠️', '👿', '👹', '👺'],
            'mocking': ['🤡', '😏', '🙄', '😒', '😤', '😠', '🖕', '🤥'],
            'negative': ['😞', '😔', '😟', '😕', '🙁', '☹️', '😣', '😖', '😫', '😩', '😢', '😭', '😤', '😠', '😡', '🤬']
        }

    def extract_emojis(self, text: str) -> List[str]:
        """
        Extract all emojis from text.

        Args:
            text: Input text

        Returns:
            List of emojis found in text
        """
        return [char for char in text if char in emoji.EMOJI_DATA]

    def get_emoji_sentiment(self, emoji_char: str) -> float:
        """
        Get sentiment score for an emoji.

        Args:
            emoji_char: Emoji character

        Returns:
            Sentiment score (-1 to 1)
        """
        return self.emoji_sentiment.get(emoji_char, 0.0)

    def analyze_emoji_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze emoji sentiment in text.

        Args:
            text: Input text

        Returns:
            Emoji sentiment analysis
        """
        emojis = self.extract_emojis(text)
        if not emojis:
            return {
                'emoji_count': 0,
                'sentiment_score': 0.0,
                'harassment_score': 0.0,
                'dominant_sentiment': 'neutral',
                'emoji_list': []
            }

        # Calculate average sentiment
        sentiments = [self.get_emoji_sentiment(e) for e in emojis]
        avg_sentiment = np.mean(sentiments)

        # Calculate harassment score
        harassment_score = 0.0
        for emoji_char in emojis:
            if emoji_char in self.harassment_emojis['threatening']:
                harassment_score += 0.8
            elif emoji_char in self.harassment_emojis['mocking']:
                harassment_score += 0.6
            elif emoji_char in self.harassment_emojis['negative']:
                harassment_score += 0.4

        harassment_score = min(harassment_score / len(emojis), 1.0) if emojis else 0.0

        # Determine dominant sentiment
        if avg_sentiment > 0.3:
            dominant = 'positive'
        elif avg_sentiment < -0.3:
            dominant = 'negative'
        else:
            dominant = 'neutral'

        return {
            'emoji_count': len(emojis),
            'sentiment_score': avg_sentiment,
            'harassment_score': harassment_score,
            'dominant_sentiment': dominant,
            'emoji_list': emojis
        }


class DeepEmojiDataset(Dataset):
    """Dataset class for deep emoji model."""

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 tokenizer,
                 emoji_extractor: EmojiFeatureExtractor,
                 max_length: int = 128):
        """
        Initialize dataset.

        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer: BERT tokenizer
            emoji_extractor: Emoji feature extractor
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.emoji_extractor = emoji_extractor
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

        # Extract emoji features
        emoji_features = self.emoji_extractor.analyze_emoji_sentiment(text)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emoji_features': torch.tensor([
                emoji_features['emoji_count'] / 10.0,  # Normalize
                emoji_features['sentiment_score'],
                emoji_features['harassment_score']
            ], dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DeepEmojiModel(nn.Module):
    """Deep learning model for emoji-enhanced text classification."""

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 emoji_feature_dim: int = 3,
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize deep emoji model.

        Args:
            model_name: Name of the base transformer model
            emoji_feature_dim: Dimension of emoji features
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(DeepEmojiModel, self).__init__()

        # Base transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config

        # Emoji feature processing
        self.emoji_processor = nn.Sequential(
            nn.Linear(emoji_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, emoji_features):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            emoji_features: Emoji feature vector

        Returns:
            Logits for classification
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Use [CLS] token representation
        text_features = transformer_outputs.pooler_output

        # Process emoji features
        emoji_processed = self.emoji_processor(emoji_features)

        # Concatenate features
        combined_features = torch.cat([text_features, emoji_processed], dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits


class DeepEmojiModelTrainer:
    """Trainer class for deep emoji model."""

    def __init__(self,
                 model: DeepEmojiModel,
                 tokenizer,
                 emoji_extractor: EmojiFeatureExtractor,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize trainer.

        Args:
            model: Deep emoji model
            tokenizer: Tokenizer
            emoji_extractor: Emoji feature extractor
            device: Device to run training on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.emoji_extractor = emoji_extractor
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
            emoji_features = batch['emoji_features'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = self.model(input_ids, attention_mask, emoji_features)
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
                emoji_features = batch['emoji_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask, emoji_features)
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
              val_dataloader: DataLoader = None,
              num_epochs: int = 3,
              learning_rate: float = 2e-5,
              warmup_steps: int = 100,
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
            val_loss, val_acc = 0.0, 0.0
            if val_dataloader:
                val_loss, val_acc = self.validate_epoch(val_dataloader, criterion)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if val_dataloader:
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
        dataset = DeepEmojiDataset(
            texts, [0] * len(texts), self.tokenizer, self.emoji_extractor
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emoji_features = batch['emoji_features'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask, emoji_features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                # Store predictions
                for i in range(len(preds)):
                    emoji_analysis = self.emoji_extractor.analyze_emoji_sentiment(texts[len(predictions) + i])
                    predictions.append({
                        'text': texts[len(predictions) + i],
                        'prediction': preds[i].item(),
                        'probability': probs[i].cpu().numpy().tolist(),
                        'confidence': torch.max(probs[i]).item(),
                        'emoji_analysis': emoji_analysis
                    })

        return predictions

    def save_model(self, path: str):
        """Save the trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'emoji_extractor': self.emoji_extractor,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)

        logger.info(f"Deep emoji model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        self.emoji_extractor = checkpoint['emoji_extractor']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])

        logger.info(f"Deep emoji model loaded from {path}")

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emoji_features = batch['emoji_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask, emoji_features)
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


def create_deep_emoji_model(config: Dict[str, Any]) -> Tuple[DeepEmojiModel, Any, EmojiFeatureExtractor]:
    """
    Create deep emoji model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model, tokenizer, emoji_extractor)
    """
    model_name = config.get('model_name', 'bert-base-uncased')
    emoji_feature_dim = config.get('emoji_feature_dim', 3)
    hidden_dim = config.get('hidden_dim', 256)
    num_classes = config.get('num_classes', 2)
    dropout_rate = config.get('dropout_rate', 0.3)

    # Create components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    emoji_extractor = EmojiFeatureExtractor()

    model = DeepEmojiModel(
        model_name=model_name,
        emoji_feature_dim=emoji_feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    return model, tokenizer, emoji_extractor
