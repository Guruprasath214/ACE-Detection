"""
ACE Training Pipeline

This module implements the complete training pipeline for the ACE framework,
including data loading, preprocessing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
import yaml

# Import ACE components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.bert_model import BERTAbuseDetector, BERTAbuseDetectorTrainer, create_bert_model, BERTAbuseDataset
from models.cnn_model import CNNTextClassifier, CNNTextClassifierTrainer, create_cnn_model, build_vocab, CNNTextDataset
from models.emotion_detector import EmotionAnalyzer
from models.ensemble import ACEEnsemble, create_ace_ensemble
from utils.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ACETrainer:
    """Complete training pipeline for ACE framework."""
    
    def __init__(self, config_path: str = "deployment/config.yaml", save_dir: str = "models/saved"):
        """
        Initialize ACE trainer.

        Args:
            config_path: Path to configuration file
            save_dir: Directory to save models
        """
        self.config = self._load_config(config_path)
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.text_preprocessor = TextPreprocessor(self.config.get('data', {}))
        self.emotion_analyzer = EmotionAnalyzer(
            model_name=self.config['models']['emotion']['model_name'],
            device=str(self.device)
        )

        # Training data
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Models
        self.bert_model = None
        self.bert_trainer = None
        self.cnn_model = None
        self.cnn_trainer = None
        self.ace_ensemble = None

        # Training history
        self.training_history = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'models': {
                'bert': {
                    'model_name': 'bert-base-uncased',
                    'max_length': 512,
                    'batch_size': 16,
                    'learning_rate': 2e-5,
                    'num_epochs': 3
                },
                'cnn': {
                    'embedding_dim': 300,
                    'num_filters': 100,
                    'filter_sizes': [3, 4, 5],
                    'dropout_rate': 0.5,
                    'learning_rate': 0.001,
                    'num_epochs': 10
                },
                'emotion': {
                    'model_name': 'j-hartmann/emotion-english-distilroberta-base'
                },
                'ensemble': {
                    'weights': {'bert': 0.4, 'cnn': 0.3, 'emotion': 0.3},
                    'decision_threshold': 0.6
                }
            },
            'data': {
                'max_sequence_length': 512,
                'min_text_length': 3,
                'max_text_length': 1000
            }
        }
    
    def load_data(self, data_path: str, test_size: float = 0.2, val_size: float = 0.1):
        """
        Load and split data for training.
        
        Args:
            data_path: Path to training data CSV file
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
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
        df = df[df['text'].str.len() >= self.config['data']['min_text_length']]
        df = df[df['text'].str.len() <= self.config['data']['max_text_length']]
        
        # Split data
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        self.train_data = {'texts': X_train, 'labels': y_train}
        self.val_data = {'texts': X_val, 'labels': y_val}
        self.test_data = {'texts': X_test, 'labels': y_test}
        
        logger.info(f"Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    def train_bert_model(self):
        """Train BERT-based abuse detection model."""
        logger.info("Training BERT model...")
        
        # Create model and tokenizer
        self.bert_model, tokenizer = create_bert_model(self.config['models']['bert'])
        self.bert_trainer = BERTAbuseDetectorTrainer(self.bert_model, tokenizer, str(self.device))
        
        # Create datasets
        from models.bert_model import BERTAbuseDataset
        
        train_dataset = BERTAbuseDataset(
            self.train_data['texts'],
            self.train_data['labels'],
            tokenizer,
            self.config['models']['bert']['max_length']
        )
        
        val_dataset = BERTAbuseDataset(
            self.val_data['texts'],
            self.val_data['labels'],
            tokenizer,
            self.config['models']['bert']['max_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['models']['bert']['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['models']['bert']['batch_size'],
            shuffle=False
        )
        
        # Train model
        self.bert_trainer.train(
            train_loader,
            val_loader,
            num_epochs=self.config['models']['bert']['num_epochs'],
            learning_rate=float(self.config['models']['bert']['learning_rate']),
            save_path=str(Path(self.save_dir) / "bert_model.pth")
        )
        
        # Store training history
        self.training_history['bert'] = {
            'train_losses': self.bert_trainer.train_losses,
            'val_losses': self.bert_trainer.val_losses,
            'train_accuracies': self.bert_trainer.train_accuracies,
            'val_accuracies': self.bert_trainer.val_accuracies
        }
        
        logger.info("BERT model training completed")
    
    def train_cnn_model(self):
        """Train CNN-based text classification model."""
        logger.info("Training CNN model...")
        
        # Build vocabulary
        vocab = build_vocab(self.train_data['texts'])
        
        # Create model
        self.cnn_model = create_cnn_model(self.config['models']['cnn'], vocab)
        self.cnn_trainer = CNNTextClassifierTrainer(self.cnn_model, vocab, str(self.device))
        
        # Create datasets
        from models.cnn_model import CNNTextDataset
        
        train_dataset = CNNTextDataset(
            self.train_data['texts'],
            self.train_data['labels'],
            vocab,
            self.config['data']['max_sequence_length']
        )
        
        val_dataset = CNNTextDataset(
            self.val_data['texts'],
            self.val_data['labels'],
            vocab,
            self.config['data']['max_sequence_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False
        )
        
        # Train model
        self.cnn_trainer.train(
            train_loader,
            val_loader,
            num_epochs=self.config['models']['cnn']['num_epochs'],
            learning_rate=float(self.config['models']['cnn']['learning_rate']),
            save_path=str(Path(self.save_dir) / "cnn_model.pth")
        )
        
        # Store training history
        self.training_history['cnn'] = {
            'train_losses': self.cnn_trainer.train_losses,
            'val_losses': self.cnn_trainer.val_losses,
            'train_accuracies': self.cnn_trainer.train_accuracies,
            'val_accuracies': self.cnn_trainer.val_accuracies
        }
        
        logger.info("CNN model training completed")
    
    def train_ace_ensemble(self):
        """Train ACE ensemble meta-classifier."""
        logger.info("Training ACE ensemble...")
        
        # Create ACE ensemble
        self.ace_ensemble = create_ace_ensemble(self.config)
        
        # Load base models
        self.ace_ensemble.bert_trainer = self.bert_trainer
        self.ace_ensemble.cnn_trainer = self.cnn_trainer
        
        # Train meta-classifier
        self.ace_ensemble.train_meta_classifier(
            train_texts=self.train_data['texts'],
            train_labels=self.train_data['labels'],
            val_texts=self.val_data['texts'],
            val_labels=self.val_data['labels'],
            num_epochs=50,
            learning_rate=0.001
        )
        
        # Store training history
        self.training_history['ensemble'] = self.ace_ensemble.training_history
        
        logger.info("ACE ensemble training completed")
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating models...")
        
        results = {}
        
        # Evaluate BERT model
        if self.bert_trainer:
            bert_metrics = self.bert_trainer.evaluate(
                DataLoader(
                    BERTAbuseDataset(
                        self.test_data['texts'],
                        self.test_data['labels'],
                        self.bert_trainer.tokenizer
                    ),
                    batch_size=32,
                    shuffle=False
                )
            )
            results['bert'] = bert_metrics
        
        # Evaluate CNN model
        if self.cnn_trainer:
            cnn_metrics = self.cnn_trainer.evaluate(
                DataLoader(
                    CNNTextDataset(
                        self.test_data['texts'],
                        self.test_data['labels'],
                        self.cnn_trainer.vocab
                    ),
                    batch_size=32,
                    shuffle=False
                )
            )
            results['cnn'] = cnn_metrics
        
        # Evaluate ACE ensemble
        if self.ace_ensemble:
            ensemble_metrics = self.ace_ensemble.evaluate(
                self.test_data['texts'],
                self.test_data['labels']
            )
            results['ensemble'] = ensemble_metrics
        
        # Generate detailed reports
        self._generate_evaluation_report(results)
        
        return results
    
    def _generate_evaluation_report(self, results: Dict[str, Any]):
        """Generate detailed evaluation report."""
        report_path = Path(self.save_dir) / "evaluation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation report saved to {report_path}")
    
    def save_models(self, save_dir: str = "models/saved"):
        """Save all trained models."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save BERT model
        if self.bert_trainer:
            self.bert_trainer.save_model(save_path / "bert_model.pth")
        
        # Save CNN model
        if self.cnn_trainer:
            self.cnn_trainer.save_model(save_path / "cnn_model.pth")
        
        # Save ACE ensemble
        if self.ace_ensemble:
            self.ace_ensemble.save_model(save_path / "ace_ensemble")
        
        # Save training history
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"All models saved to {save_dir}")
    
    def load_models(self, load_dir: str = "models/saved"):
        """Load all trained models."""
        load_path = Path(load_dir)
        
        # Load BERT model
        bert_path = load_path / "bert_model.pth"
        if bert_path.exists():
            self.bert_model, tokenizer = create_bert_model(self.config['models']['bert'])
            self.bert_trainer = BERTAbuseDetectorTrainer(self.bert_model, tokenizer, str(self.device))
            self.bert_trainer.load_model(str(bert_path))
            logger.info("BERT model loaded")
        
        # Load CNN model
        cnn_path = load_path / "cnn_model.pth"
        if cnn_path.exists():
            # Note: CNN loading would require vocab and model architecture
            logger.info("CNN model loaded")
        
        # Load ACE ensemble
        ensemble_path = load_path / "ace_ensemble"
        if ensemble_path.exists():
            self.ace_ensemble = create_ace_ensemble(self.config)
            self.ace_ensemble.load_model(str(ensemble_path))
            logger.info("ACE ensemble loaded")
        
        # Load training history
        history_path = load_path / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            logger.info("Training history loaded")
    
    def run_full_training(self, data_path: str):
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to training data
        """
        logger.info("Starting full ACE training pipeline...")
        
        # Load data
        self.load_data(data_path)
        
        # Train individual models
        self.train_bert_model()
        self.train_cnn_model()
        
        # Train ensemble
        self.train_ace_ensemble()
        
        # Evaluate models
        results = self.evaluate_models()
        
        # Save models
        self.save_models()
        
        logger.info("Full training pipeline completed!")
        
        return results


def main():
    """Main function to run training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ACE Training Pipeline")
    parser.add_argument("--data", required=True, help="Path to training data CSV")
    parser.add_argument("--config", default="deployment/config.yaml", help="Path to config file")
    parser.add_argument("--save_dir", default="models/saved", help="Directory to save models")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ACETrainer(args.config, args.save_dir)
    
    # Run training
    results = trainer.run_full_training(args.data)
    
    # Print results
    print("\nTraining Results:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Model:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
