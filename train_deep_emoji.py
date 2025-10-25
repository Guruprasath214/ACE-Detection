"""
Training script for Deep Emoji Model

This script provides a standalone training pipeline for the Deep Emoji model
that can be trained on CSV datasets with text and label columns.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
import yaml

# Import Deep Emoji components
from models.deep_emoji_model import (
    DeepEmojiModel,
    DeepEmojiModelTrainer,
    DeepEmojiDataset,
    create_deep_emoji_model,
    EmojiFeatureExtractor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "deployment/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using default config.")
        return get_default_config()


def get_default_config() -> dict:
    """Get default configuration for Deep Emoji model."""
    return {
        'models': {
            'deep_emoji': {
                'model_name': 'bert-base-uncased',
                'emoji_feature_dim': 3,
                'hidden_dim': 256,
                'num_classes': 2,
                'dropout_rate': 0.3,
                'max_length': 128,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 3
            }
        },
        'data': {
            'max_sequence_length': 128,
            'min_text_length': 3,
            'max_text_length': 1000,
            'test_size': 0.2,
            'val_size': 0.1
        }
    }


def load_and_split_data(data_path: str, config: dict):
    """
    Load and split CSV data for training.

    Args:
        data_path: Path to CSV file with 'text' and 'label' columns
        config: Configuration dictionary

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    logger.info(f"Loading data from {data_path}")

    # Load data
    df = pd.read_csv(data_path)

    # Validate data format
    required_columns = ['text', 'label']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Data must contain columns: {required_columns}")

    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.len() >= config['data']['min_text_length']]
    df = df[df['text'].str.len() <= config['data']['max_text_length']]

    # Convert labels to int if needed
    df['label'] = df['label'].astype(int)

    # Split data
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels,
        test_size=config['data']['test_size'],
        random_state=42,
        stratify=labels
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config['data']['val_size'] / (1 - config['data']['test_size']),
        random_state=42,
        stratify=y_temp
    )

    train_data = {'texts': X_train, 'labels': y_train}
    val_data = {'texts': X_val, 'labels': y_val}
    test_data = {'texts': X_test, 'labels': y_test}

    logger.info(f"Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

    return train_data, val_data, test_data


def train_deep_emoji_model(data_path: str,
                          config_path: str = "deployment/config.yaml",
                          save_path: str = "models/saved/deep_emoji_model.pth"):
    """
    Train the Deep Emoji model on CSV data.

    Args:
        data_path: Path to CSV training data
        config_path: Path to configuration file
        save_path: Path to save trained model
    """
    # Load configuration
    config = load_config(config_path)

    # Load and split data
    train_data, val_data, test_data = load_and_split_data(data_path, config)

    # Create model and components
    model_config = config['models']['deep_emoji']
    model, tokenizer, emoji_extractor = create_deep_emoji_model(model_config)

    # Create trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = DeepEmojiModelTrainer(model, tokenizer, emoji_extractor, device)

    # Create datasets
    train_dataset = DeepEmojiDataset(
        train_data['texts'],
        train_data['labels'],
        tokenizer,
        emoji_extractor,
        max_length=model_config['max_length']
    )

    val_dataset = DeepEmojiDataset(
        val_data['texts'],
        val_data['labels'],
        tokenizer,
        emoji_extractor,
        max_length=model_config['max_length']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config['batch_size'],
        shuffle=False
    )

    # Train the model
    logger.info("Starting Deep Emoji model training...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=model_config['num_epochs'],
        learning_rate=model_config['learning_rate'],
        save_path=save_path
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_dataset = DeepEmojiDataset(
        test_data['texts'],
        test_data['labels'],
        tokenizer,
        emoji_extractor,
        max_length=model_config['max_length']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=model_config['batch_size'],
        shuffle=False
    )

    test_metrics = trainer.evaluate(test_loader)

    # Print results
    logger.info("Training completed!")
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_path)
    logger.info(f"Model saved to {save_path}")

    return trainer, test_metrics


def predict_with_deep_emoji(texts: List[str],
                           model_path: str = "models/saved/deep_emoji_model.pth",
                           config_path: str = "deployment/config.yaml"):
    """
    Make predictions using a trained Deep Emoji model.

    Args:
        texts: List of texts to predict
        model_path: Path to trained model
        config_path: Path to configuration file

    Returns:
        List of prediction dictionaries
    """
    # Load configuration
    config = load_config(config_path)
    model_config = config['models']['deep_emoji']

    # Create model and components
    model, tokenizer, emoji_extractor = create_deep_emoji_model(model_config)

    # Create trainer and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = DeepEmojiModelTrainer(model, tokenizer, emoji_extractor, device)
    trainer.load_model(model_path)

    # Make predictions
    predictions = trainer.predict(texts)

    return predictions


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description="Train Deep Emoji Model")
    parser.add_argument("--data", required=True, help="Path to CSV training data")
    parser.add_argument("--config", default="deployment/config.yaml", help="Path to config file")
    parser.add_argument("--save_path", default="models/saved/deep_emoji_model.pth", help="Path to save trained model")
    parser.add_argument("--predict", nargs='*', help="Make predictions on provided texts")

    args = parser.parse_args()

    if args.predict:
        # Prediction mode
        predictions = predict_with_deep_emoji(args.predict, args.save_path, args.config)
        for i, pred in enumerate(predictions):
            print(f"Text {i+1}: {pred['text']}")
            print(f"  Prediction: {pred['prediction']} (Confidence: {pred['confidence']:.4f})")
            print(f"  Emoji Analysis: {pred['emoji_analysis']}")
            print()
    else:
        # Training mode
        trainer, metrics = train_deep_emoji_model(args.data, args.config, args.save_path)
        print("\nTraining Summary:")
        print(f"Model saved to: {args.save_path}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1-Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
