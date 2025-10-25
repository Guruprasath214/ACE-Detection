"""
Emotion Detection Component

This module implements emotion detection for harassment analysis using
pre-trained transformer models and custom emotion classification.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetector:
    """Emotion detection using pre-trained transformer models."""
    
    def __init__(self, 
                 model_name: str = "j-hartmann/emotion-english-distilroberta-base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize emotion detector.
        
        Args:
            model_name: Name of the pre-trained emotion model
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Create pipeline for easy inference
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        # Emotion categories and their harassment relevance
        self.emotion_categories = {
            'anger': {'harassment_relevance': 0.8, 'intensity_threshold': 0.7},
            'disgust': {'harassment_relevance': 0.6, 'intensity_threshold': 0.6},
            'fear': {'harassment_relevance': 0.7, 'intensity_threshold': 0.6},
            'joy': {'harassment_relevance': 0.1, 'intensity_threshold': 0.8},
            'neutral': {'harassment_relevance': 0.2, 'intensity_threshold': 0.5},
            'sadness': {'harassment_relevance': 0.5, 'intensity_threshold': 0.6},
            'surprise': {'harassment_relevance': 0.3, 'intensity_threshold': 0.5}
        }
    
    def detect_emotions(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect emotions in a list of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of emotion detection results
        """
        results = []
        
        for text in texts:
            # Get emotion predictions
            emotion_results = self.pipeline(text)
            
            # Process results
            emotion_scores = {}
            for result in emotion_results:
                emotion_scores[result['label']] = result['score']
            
            # Calculate harassment risk based on emotions
            harassment_risk = self._calculate_harassment_risk(emotion_scores)
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            results.append({
                'text': text,
                'emotion_scores': emotion_scores,
                'dominant_emotion': dominant_emotion[0],
                'dominant_confidence': dominant_emotion[1],
                'harassment_risk': harassment_risk,
                'emotion_analysis': self._analyze_emotion_patterns(emotion_scores)
            })
        
        return results
    
    def _calculate_harassment_risk(self, emotion_scores: Dict[str, float]) -> float:
        """
        Calculate harassment risk based on emotion scores.
        
        Args:
            emotion_scores: Dictionary of emotion scores
            
        Returns:
            Harassment risk score (0-1)
        """
        risk_score = 0.0
        
        for emotion, score in emotion_scores.items():
            if emotion in self.emotion_categories:
                category_info = self.emotion_categories[emotion]
                relevance = category_info['harassment_relevance']
                threshold = category_info['intensity_threshold']
                
                # Only consider emotions above threshold
                if score >= threshold:
                    risk_score += score * relevance
        
        # Normalize risk score
        return min(risk_score, 1.0)
    
    def _analyze_emotion_patterns(self, emotion_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze emotion patterns for harassment indicators.
        
        Args:
            emotion_scores: Dictionary of emotion scores
            
        Returns:
            Analysis of emotion patterns
        """
        analysis = {
            'negative_emotions': 0.0,
            'positive_emotions': 0.0,
            'neutral_emotions': 0.0,
            'high_intensity_emotions': [],
            'emotion_volatility': 0.0
        }
        
        # Categorize emotions
        negative_emotions = ['anger', 'disgust', 'fear', 'sadness']
        positive_emotions = ['joy']
        neutral_emotions = ['neutral', 'surprise']
        
        for emotion, score in emotion_scores.items():
            if emotion in negative_emotions:
                analysis['negative_emotions'] += score
            elif emotion in positive_emotions:
                analysis['positive_emotions'] += score
            elif emotion in neutral_emotions:
                analysis['neutral_emotions'] += score
            
            # Track high intensity emotions
            if score >= 0.7:
                analysis['high_intensity_emotions'].append((emotion, score))
        
        # Calculate emotion volatility (standard deviation of scores)
        scores = list(emotion_scores.values())
        if len(scores) > 1:
            analysis['emotion_volatility'] = np.std(scores)
        
        return analysis
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict emotions for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotion prediction result
        """
        results = self.detect_emotions([text])
        return results[0] if results else {}


class CustomEmotionClassifier(nn.Module):
    """Custom emotion classifier for fine-tuned harassment detection."""
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 num_emotions: int = 7,
                 dropout_rate: float = 0.3):
        """
        Initialize custom emotion classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_emotions: Number of emotion classes
            dropout_rate: Dropout rate
        """
        super(CustomEmotionClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_emotions)
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
        Forward pass through the model.
        
        Args:
            x: Input features
            
        Returns:
            Emotion logits
        """
        return self.classifier(x)


class EmotionFeatureExtractor:
    """Extract emotion-related features from text."""
    
    def __init__(self):
        """Initialize emotion feature extractor."""
        self.harassment_emotion_indicators = {
            'anger_indicators': [
                'angry', 'mad', 'furious', 'rage', 'hate', 'disgusted',
                'annoyed', 'irritated', 'frustrated', 'outraged'
            ],
            'fear_indicators': [
                'scared', 'afraid', 'terrified', 'worried', 'anxious',
                'nervous', 'frightened', 'panicked', 'threatened'
            ],
            'sadness_indicators': [
                'sad', 'depressed', 'hurt', 'upset', 'crying', 'tears',
                'broken', 'devastated', 'heartbroken', 'miserable'
            ],
            'disgust_indicators': [
                'disgusted', 'revolted', 'sickened', 'appalled', 'horrified',
                'repulsed', 'nauseated', 'offended'
            ]
        }
    
    def extract_emotion_features(self, text: str) -> Dict[str, float]:
        """
        Extract emotion-related features from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of emotion features
        """
        text_lower = text.lower()
        features = {}
        
        # Count emotion indicators
        for emotion_type, indicators in self.harassment_emotion_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            features[emotion_type] = count / len(indicators)  # Normalize
        
        # Calculate overall negative emotion score
        features['negative_emotion_score'] = sum([
            features.get('anger_indicators', 0),
            features.get('fear_indicators', 0),
            features.get('sadness_indicators', 0),
            features.get('disgust_indicators', 0)
        ]) / 4
        
        # Count exclamation marks (intensity indicator)
        features['exclamation_count'] = text.count('!')
        
        # Count caps words (intensity indicator)
        caps_words = [word for word in text.split() if word.isupper() and len(word) > 1]
        features['caps_word_count'] = len(caps_words)
        
        # Calculate caps ratio
        total_words = len(text.split())
        features['caps_ratio'] = features['caps_word_count'] / total_words if total_words > 0 else 0
        
        return features
    
    def extract_emotion_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extract emotion-related n-grams from text.
        
        Args:
            text: Text to analyze
            n: N-gram size
            
        Returns:
            List of emotion-related n-grams
        """
        words = text.lower().split()
        emotion_ngrams = []
        
        # Get all emotion indicators
        all_indicators = []
        for indicators in self.harassment_emotion_indicators.values():
            all_indicators.extend(indicators)
        
        # Extract n-grams containing emotion indicators
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            if any(indicator in ngram for indicator in all_indicators):
                emotion_ngrams.append(ngram)
        
        return emotion_ngrams


class EmotionAnalyzer:
    """Comprehensive emotion analysis for harassment detection."""
    
    def __init__(self, 
                 model_name: str = "j-hartmann/emotion-english-distilroberta-base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize emotion analyzer.
        
        Args:
            model_name: Name of the pre-trained emotion model
            device: Device to run inference on
        """
        self.emotion_detector = EmotionDetector(model_name, device)
        self.feature_extractor = EmotionFeatureExtractor()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive emotion analysis on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Comprehensive emotion analysis results
        """
        # Get emotion predictions
        emotion_result = self.emotion_detector.predict_single(text)
        
        # Extract emotion features
        emotion_features = self.feature_extractor.extract_emotion_features(text)
        
        # Extract emotion n-grams
        emotion_ngrams = self.feature_extractor.extract_emotion_ngrams(text)
        
        # Combine results
        analysis = {
            'text': text,
            'emotion_prediction': emotion_result,
            'emotion_features': emotion_features,
            'emotion_ngrams': emotion_ngrams,
            'overall_emotion_risk': self._calculate_overall_risk(emotion_result, emotion_features)
        }
        
        return analysis
    
    def _calculate_overall_risk(self, emotion_result: Dict[str, Any], emotion_features: Dict[str, float]) -> float:
        """
        Calculate overall emotion-based harassment risk.
        
        Args:
            emotion_result: Emotion prediction result
            emotion_features: Emotion features
            
        Returns:
            Overall risk score (0-1)
        """
        # Get harassment risk from emotion prediction
        emotion_risk = emotion_result.get('harassment_risk', 0.0)
        
        # Get negative emotion score from features
        negative_emotion_score = emotion_features.get('negative_emotion_score', 0.0)
        
        # Get intensity indicators
        exclamation_intensity = min(emotion_features.get('exclamation_count', 0) / 5, 1.0)
        caps_intensity = min(emotion_features.get('caps_ratio', 0) * 10, 1.0)
        
        # Calculate weighted overall risk
        overall_risk = (
            emotion_risk * 0.4 +
            negative_emotion_score * 0.3 +
            exclamation_intensity * 0.15 +
            caps_intensity * 0.15
        )
        
        return min(overall_risk, 1.0)
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Perform emotion analysis on multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of emotion analysis results
        """
        return [self.analyze_text(text) for text in texts]
    
    def save_model(self, path: str):
        """Save the emotion analyzer model."""
        # This would save the custom emotion classifier if we had one
        # For now, we'll save the configuration
        config = {
            'model_name': self.emotion_detector.model_name,
            'device': self.emotion_detector.device
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Emotion analyzer configuration saved to {path}")
    
    def load_model(self, path: str):
        """Load the emotion analyzer model."""
        with open(path, 'r') as f:
            config = json.load(f)
        
        # Reinitialize with saved configuration
        self.emotion_detector = EmotionDetector(
            model_name=config['model_name'],
            device=config['device']
        )
        
        logger.info(f"Emotion analyzer loaded from {path}")


def create_emotion_analyzer(config: Dict[str, Any]) -> EmotionAnalyzer:
    """
    Create emotion analyzer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Emotion analyzer instance
    """
    model_name = config.get('model_name', 'j-hartmann/emotion-english-distilroberta-base')
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    return EmotionAnalyzer(model_name=model_name, device=device)
