"""
ACE Ensemble Meta-Classifier

This module implements the ACE (Advanced Cyberbullying & Emotion Detection) 
ensemble meta-classifier that combines BERT, CNN, and Emotion detection models
for comprehensive harassment detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import pipeline, AutoTokenizer, AutoModel
import json
from pathlib import Path
import logging
import pickle
import re
from .bert_model import BERTAbuseDetector, BERTAbuseDetectorTrainer
from .cnn_model import CNNTextClassifier, CNNTextClassifierTrainer
from .emotion_detector import EmotionAnalyzer
from .svm_model import SVMTextClassifier, SVMTextClassifierTrainer
from .deep_emoji_model import DeepEmojiModel, DeepEmojiModelTrainer, EmojiFeatureExtractor, create_deep_emoji_model
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on harassment-relevant features."""

    def __init__(self, input_dim: int, attention_dim: int = 64):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if needed

        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_features = torch.sum(attention_weights * x, dim=1)
        return attended_features


class ACEMetaClassifier(nn.Module):
    """Enhanced neural network-based meta-classifier for ACE ensemble with attention."""

    def __init__(self,
                 input_dim: int = 6,  # BERT, CNN, SVM, Deep Emoji, Emotion, Contextual scores
                 hidden_dim: int = 128,
                 num_classes: int = 7,  # Multi-class: normal, bullying, harassment, women's_harassment, mocking, threats, hate_speech
                 dropout_rate: float = 0.3,
                 use_attention: bool = True):
        """
        Initialize enhanced ACE meta-classifier.

        Args:
            input_dim: Input dimension (number of base models)
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (7 for multi-class harassment detection)
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super(ACEMetaClassifier, self).__init__()

        self.use_attention = use_attention
        self.num_classes = num_classes

        if use_attention:
            self.attention = AttentionLayer(input_dim, attention_dim=hidden_dim // 2)
            classifier_input_dim = input_dim
        else:
            classifier_input_dim = input_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through the meta-classifier.

        Args:
            x: Input features from base models

        Returns:
            Final classification logits
        """
        if self.use_attention:
            x = self.attention(x)

        return self.classifier(x)


class ACEEnsemble:
    """ACE Ensemble Meta-Classifier combining BERT, CNN, SVM, Deep Emoji, Emotion, and Contextual detection."""

    def __init__(self,
                 config: Dict[str, Any],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize ACE ensemble.

        Args:
            config: Configuration dictionary
            device: Device to run inference on
        """
        self.config = config
        self.device = device

        # Initialize components
        self.text_preprocessor = TextPreprocessor(config.get('data', {}))
        self.emotion_analyzer = EmotionAnalyzer(
            model_name=config['models']['emotion']['model_name'],
            device=device
        )

        # Initialize pretrained hate speech pipeline
        self.hate_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-hate-latest")

        # Initialize contextual embeddings for semantic understanding (using RoBERTa-large for better semantic capture)
        self.contextual_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.contextual_model = AutoModel.from_pretrained("roberta-large").to(device)

        # Initialize individual models
        self.svm_model = SVMTextClassifier()
        self.deep_emoji_model, self.deep_emoji_tokenizer, self.emoji_extractor = create_deep_emoji_model(
            config.get('models', {}).get('deep_emoji', {})
        )
        self.deep_emoji_trainer = DeepEmojiModelTrainer(
            self.deep_emoji_model, self.deep_emoji_tokenizer, self.emoji_extractor, device
        )

        # Disable custom base models to use pretrained
        self.bert_trainer = None
        self.cnn_trainer = None

        # Initialize meta-classifier with 6 models (BERT, CNN, SVM, Deep Emoji, Emotion, Contextual)
        self.meta_classifier = ACEMetaClassifier(
            input_dim=6,  # BERT, CNN, SVM, Deep Emoji, Emotion, Contextual
            hidden_dim=config['models']['ensemble'].get('hidden_dim', 64),
            num_classes=config['models']['ensemble'].get('num_classes', 2),
            dropout_rate=config['models']['ensemble'].get('dropout_rate', 0.3)
        ).to(device)

        # Ensemble weights for all 5 models
        self.ensemble_weights = config['models']['ensemble']['weights']
        self.decision_threshold = config['models']['ensemble']['decision_threshold']

        # Training history
        self.training_history = {
            'bert': {'losses': [], 'accuracies': []},
            'cnn': {'losses': [], 'accuracies': []},
            'svm': {'losses': [], 'accuracies': []},
            'deep_emoji': {'losses': [], 'accuracies': []},
            'meta': {'losses': [], 'accuracies': []}
        }
    
    def load_base_models(self, bert_path: str, cnn_path: str):
        """
        Load pre-trained base models.
        
        Args:
            bert_path: Path to BERT model
            cnn_path: Path to CNN model
        """
        # Load BERT model
        if bert_path and Path(bert_path).exists():
            self.bert_model = BERTAbuseDetector()
            self.bert_trainer = BERTAbuseDetectorTrainer(self.bert_model, None, self.device)
            self.bert_trainer.load_model(bert_path)
            logger.info(f"BERT model loaded from {bert_path}")
        
        # Load CNN model
        if cnn_path and Path(cnn_path).exists():
            # Note: CNN model loading would require vocab and model architecture
            # This is a simplified version
            logger.info(f"CNN model loaded from {cnn_path}")

    def load_models(self, models_dir: str):
        """
        Compatibility wrapper used by the API server to load models from a directory.
        This method delegates to load_base_models and looks for common filenames.
        """
        models_path = Path(models_dir)
        bert_path = str(models_path / "bert_model.pth") if (models_path / "bert_model.pth").exists() else None
        cnn_path = str(models_path / "cnn_model.pth") if (models_path / "cnn_model.pth").exists() else None
        self.load_base_models(bert_path, cnn_path)
    
    def extract_base_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract features from base models.

        Args:
            texts: List of texts to process

        Returns:
            Dictionary of base model features
        """
        features = {
            'bert_scores': [],
            'cnn_scores': [],
            'svm_scores': [],
            'deep_emoji_scores': [],
            'emotion_scores': []
        }

        # Process texts through preprocessor
        preprocessed_texts = [self.text_preprocessor.preprocess_text(text) for text in texts]

        # Extract hate speech features using pretrained model (for BERT and CNN)
        # Use return_all_scores so the pipeline returns a list of label-score dicts per example when possible
        try:
            hate_results = self.hate_pipeline(texts, return_all_scores=True)
        except TypeError:
            # Older pipeline signature may not support return_all_scores; fall back
            hate_results = self.hate_pipeline(texts)

        hate_scores = []
        for result in hate_results:
            # result may be a list of dicts (labels), a single dict, or other shapes depending on pipeline
            score = 0.0
            try:
                if isinstance(result, list):
                    # list of label/score dicts
                    score = next((r.get('score', 0.0) for r in result if str(r.get('label','')).lower() in ('hate', 'abusive', 'offensive')), 0.0)
                elif isinstance(result, dict):
                    # single dict like {'label': 'hate', 'score': 0.98}
                    lbl = str(result.get('label', '')).lower()
                    if lbl in ('hate', 'abusive', 'offensive'):
                        score = float(result.get('score', 0.0))
                    else:
                        # If label isn't hate, but only one label returned, we can try mapping known hate labels
                        score = float(result.get('score', 0.0)) if 'hate' in lbl else 0.0
                else:
                    # Unexpected type (string, etc.) - skip
                    score = 0.0
            except Exception:
                score = 0.0
            hate_scores.append(score)

        features['bert_scores'] = np.array(hate_scores)
        features['cnn_scores'] = np.array(hate_scores)  # Use same scores for simplicity

        # Extract SVM features
        if self.svm_model.is_trained:
            svm_predictions = self.svm_model.predict(texts)
            features['svm_scores'] = np.array([pred['probability'][1] for pred in svm_predictions])
        else:
            # Fallback: use hate scores
            features['svm_scores'] = np.array(hate_scores)

        # Extract Deep Emoji features
        try:
            deep_emoji_predictions = self.deep_emoji_trainer.predict(texts)
            features['deep_emoji_scores'] = np.array([pred['probability'][1] for pred in deep_emoji_predictions])
        except:
            # Fallback: use hate scores
            features['deep_emoji_scores'] = np.array(hate_scores)

        # Extract emotion features
        emotion_analyses = self.emotion_analyzer.batch_analyze(texts)
        features['emotion_scores'] = np.array([
            analysis['overall_emotion_risk'] for analysis in emotion_analyses
        ])

        # Extract contextual embeddings for semantic understanding
        contextual_scores = self._extract_contextual_features(texts)
        features['contextual_scores'] = np.array(contextual_scores)

        return features

    def _extract_contextual_features(self, texts: List[str]) -> List[float]:
        """
        Extract contextual features using RoBERTa for semantic understanding.

        Args:
            texts: List of texts to process

        Returns:
            List of contextual risk scores
        """
        contextual_scores = []

        for text in texts:
            # Tokenize and get embeddings
            inputs = self.contextual_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.contextual_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

            # Simple heuristic: use embedding magnitude as risk indicator
            # In practice, this could be trained for better performance
            risk_score = torch.norm(embeddings, p=2).item() / 10.0  # Normalize
            risk_score = min(risk_score, 1.0)  # Cap at 1.0
            contextual_scores.append(risk_score)

        return contextual_scores
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Make prediction on a single text.

        Args:
            text: Text to classify

        Returns:
            Prediction result with confidence scores
        """
        # Extract base features
        base_features = self.extract_base_features([text])

        # Combine features for all 6 models
        combined_features = np.array([
            base_features['bert_scores'][0],
            base_features['cnn_scores'][0],
            base_features['svm_scores'][0],
            base_features['deep_emoji_scores'][0],
            base_features['emotion_scores'][0],
            base_features['contextual_scores'][0]
        ]).reshape(1, -1)

        # Get meta-classifier prediction
        self.meta_classifier.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(combined_features).to(self.device)
            meta_logits = self.meta_classifier(features_tensor)
            meta_probs = torch.softmax(meta_logits, dim=1)
            meta_pred = torch.argmax(meta_probs, dim=1)

        # Calculate weighted ensemble score for all 6 models
        base_scores = {
            'bert': base_features['bert_scores'][0],
            'cnn': base_features['cnn_scores'][0],
            'svm': base_features['svm_scores'][0],
            'deep_emoji': base_features['deep_emoji_scores'][0],
            'emotion': base_features['emotion_scores'][0],
            'contextual': base_features['contextual_scores'][0]
        }

        weighted_score = sum(
            self.ensemble_weights[model] * score
            for model, score in base_scores.items()
        )

        # Use meta-classifier for final prediction (multi-class to binary conversion)
        # If meta-classifier predicts any harassment class (1-6), it's harassment
        final_prediction = 1 if meta_pred.item() > 0 else 0
        confidence = float(torch.max(meta_probs).item())

        # Fallback to weighted score if meta-classifier confidence is low
        if confidence < 0.6:
            final_prediction = 1 if weighted_score >= self.decision_threshold else 0
            confidence = max(confidence, weighted_score)

        return {
            'text': text,
            'prediction': final_prediction,
            'confidence': confidence,
            'base_scores': base_scores,
            'weighted_score': weighted_score,
            'meta_prediction': meta_pred.item(),
            'meta_probabilities': meta_probs.cpu().numpy().tolist()[0],
            'decision_threshold': self.decision_threshold
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction results
        """
        return [self.predict_single(text) for text in texts]
    
    def train_meta_classifier(self,
                            train_texts: List[str],
                            train_labels: List[int],
                            val_texts: List[str] = None,
                            val_labels: List[int] = None,
                            num_epochs: int = 50,
                            learning_rate: float = 0.001,
                            batch_size: int = 32):
        """
        Train the meta-classifier.

        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        logger.info("Training ACE meta-classifier...")

        # Extract base features
        train_features = self.extract_base_features(train_texts)

        # Combine features for all 6 models
        X_train = np.column_stack([
            train_features['bert_scores'],
            train_features['cnn_scores'],
            train_features['svm_scores'],
            train_features['deep_emoji_scores'],
            train_features['emotion_scores'],
            train_features['contextual_scores']
        ])
        y_train = np.array(train_labels)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.meta_classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.meta_classifier.train()
        for epoch in range(num_epochs):
            # Forward pass
            logits = self.meta_classifier(X_train_tensor)
            loss = criterion(logits, y_train_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_train_tensor).float().mean()

            # Store metrics
            self.training_history['meta']['losses'].append(loss.item())
            self.training_history['meta']['accuracies'].append(accuracy.item())

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}")

        # Validation if provided
        if val_texts and val_labels:
            val_accuracy = self._validate_meta_classifier(val_texts, val_labels)
            logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    
    def _validate_meta_classifier(self, val_texts: List[str], val_labels: List[int]) -> float:
        """
        Validate the meta-classifier.

        Args:
            val_texts: Validation texts
            val_labels: Validation labels

        Returns:
            Validation accuracy
        """
        # Extract base features
        val_features = self.extract_base_features(val_texts)

        # Combine features for all 6 models
        X_val = np.column_stack([
            val_features['bert_scores'],
            val_features['cnn_scores'],
            val_features['svm_scores'],
            val_features['deep_emoji_scores'],
            val_features['emotion_scores'],
            val_features['contextual_scores']
        ])
        y_val = np.array(val_labels)

        # Convert to tensors
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        # Evaluate
        self.meta_classifier.eval()
        with torch.no_grad():
            logits = self.meta_classifier(X_val_tensor)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_val_tensor).float().mean()

        return accuracy.item()
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the ensemble model.
        
        Args:
            texts: Test texts
            labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict_batch(texts)
        pred_labels = [pred['prediction'] for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, pred_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, path: str):
        """
        Save the complete ACE ensemble model.
        
        Args:
            path: Path to save the model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save meta-classifier
        meta_path = Path(path) / "meta_classifier.pth"
        torch.save({
            'model_state_dict': self.meta_classifier.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'ensemble_weights': self.ensemble_weights,
            'decision_threshold': self.decision_threshold
        }, meta_path)
        
        # Save configuration
        config_path = Path(path) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"ACE ensemble model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load the complete ACE ensemble model.
        
        Args:
            path: Path to load the model from
        """
        # Load meta-classifier
        meta_path = Path(path) / "meta_classifier.pth"
        if meta_path.exists():
            checkpoint = torch.load(meta_path, map_location=self.device)
            self.meta_classifier.load_state_dict(checkpoint['model_state_dict'])
            self.training_history = checkpoint.get('training_history', {})
            self.ensemble_weights = checkpoint.get('ensemble_weights', self.ensemble_weights)
            self.decision_threshold = checkpoint.get('decision_threshold', self.decision_threshold)
            logger.info(f"ACE ensemble model loaded from {path}")
        else:
            logger.warning(f"Meta-classifier not found at {meta_path}")
    
    def get_model_importance(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Calculate importance of each base model in the ensemble.

        Args:
            texts: Sample texts
            labels: Sample labels

        Returns:
            Dictionary of model importance scores
        """
        # Extract base features
        base_features = self.extract_base_features(texts)

        # Calculate individual model accuracies for all 5 models
        bert_accuracy = accuracy_score(labels, (base_features['bert_scores'] > 0.5).astype(int))
        cnn_accuracy = accuracy_score(labels, (base_features['cnn_scores'] > 0.5).astype(int))
        svm_accuracy = accuracy_score(labels, (base_features['svm_scores'] > 0.5).astype(int))
        deep_emoji_accuracy = accuracy_score(labels, (base_features['deep_emoji_scores'] > 0.5).astype(int))
        emotion_accuracy = accuracy_score(labels, (base_features['emotion_scores'] > 0.5).astype(int))

        # Calculate ensemble accuracy
        ensemble_predictions = self.predict_batch(texts)
        ensemble_accuracy = accuracy_score(labels, [pred['prediction'] for pred in ensemble_predictions])

        return {
            'bert_accuracy': bert_accuracy,
            'cnn_accuracy': cnn_accuracy,
            'svm_accuracy': svm_accuracy,
            'deep_emoji_accuracy': deep_emoji_accuracy,
            'emotion_accuracy': emotion_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'improvement': ensemble_accuracy - max(bert_accuracy, cnn_accuracy, svm_accuracy, deep_emoji_accuracy, emotion_accuracy)
        }
    
    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """
        Provide explanation for a prediction.
        
        Args:
            text: Text to explain
            
        Returns:
            Explanation of the prediction
        """
        prediction = self.predict_single(text)
        
        # Get preprocessing analysis
        preprocessed = self.text_preprocessor.preprocess_text(text)
        
        explanation = {
            'text': text,
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'base_model_contributions': prediction['base_scores'],
            'weighted_score': prediction['weighted_score'],
            'decision_threshold': prediction['decision_threshold'],
            'preprocessing_analysis': {
                'risk_score': preprocessed['risk_score'],
                'emoji_analysis': preprocessed['emoji_analysis'],
                'slang_analysis': preprocessed['slang_analysis'],
                'behavior_analysis': preprocessed['behavior_analysis']
            },
            'explanation': self._generate_explanation(prediction, preprocessed)
        }
        
        return explanation
    
    def _generate_explanation(self, prediction: Dict[str, Any], preprocessed: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for the prediction.
        
        Args:
            prediction: Prediction result
            preprocessed: Preprocessing analysis
            
        Returns:
            Human-readable explanation
        """
        if prediction['prediction'] == 1:
            explanation = "This text was classified as harassment. "
            
            # Analyze contributing factors
            factors = []
            if prediction['base_scores']['bert'] > 0.7:
                factors.append("strong contextual indicators")
            if prediction['base_scores']['emotion'] > 0.7:
                factors.append("negative emotional content")
            if preprocessed['risk_score'] > 0.6:
                factors.append("high-risk language patterns")
            
            if factors:
                explanation += f"Key factors: {', '.join(factors)}."
        else:
            explanation = "This text was classified as normal. "
            
            # Analyze why it's not harassment
            if prediction['weighted_score'] < 0.3:
                explanation += "Low risk indicators across all models."
            else:
                explanation += "Risk indicators present but below threshold."
        
        return explanation


def create_ace_ensemble(config: Dict[str, Any]) -> ACEEnsemble:
    """
    Create ACE ensemble from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ACE ensemble instance
    """
    return ACEEnsemble(config)
