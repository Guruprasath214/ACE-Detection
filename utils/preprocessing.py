"""
Advanced Text Preprocessing for ACE Detection Framework

This module provides comprehensive text preprocessing capabilities including:
- Emoji analysis and normalization
- Slang detection and handling
- Behavioral pattern analysis
- Feature extraction for ML models
"""

import re
import json
import emoji
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
try:
    from nltk.stem import WordNetLemmatizer
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    WordNetLemmatizer = None
from textblob import TextBlob
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except (LookupError, Exception):
    try:
        nltk.download('wordnet')
    except Exception:
        print("Warning: Could not download wordnet. Some features may not work.")


class EmojiProcessor:
    """Process and analyze emojis in text for harassment detection."""
    
    def __init__(self, emoji_map_path: str = "assets/emoji_map.json"):
        """Initialize emoji processor with mapping data."""
        self.emoji_map = self._load_emoji_map(emoji_map_path)
        self.lemmatizer = WordNetLemmatizer() if WORDNET_AVAILABLE else None
        
    def _load_emoji_map(self, path: str) -> Dict:
        """Load emoji mapping from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Emoji map not found at {path}. Using default mapping.")
            return self._get_default_emoji_map()
    
    def _get_default_emoji_map(self) -> Dict:
        """Default emoji mapping if file not found."""
        return {
            "negative_emotions": {
                "anger": ["😠", "😡", "🤬", "💢"],
                "sadness": ["😢", "😭", "💔", "😞"],
                "fear": ["😨", "😰", "😱", "😖"]
            },
            "threatening_emotions": {
                "aggression": ["👊", "💪", "🗡️", "⚔️"],
                "intimidation": ["👁️", "👀", "🔍", "🎯"]
            }
        }
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract all emojis from text."""
        return emoji.emoji_list(text)
    
    def analyze_emoji_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze emoji sentiment and emotional indicators."""
        emojis = self.extract_emojis(text)
        sentiment_scores = {
            'negative': 0.0,
            'threatening': 0.0,
            'manipulative': 0.0,
            'positive': 0.0,
            'neutral': 0.0
        }
        
        for emoji_data in emojis:
            emoji_char = emoji_data['emoji']
            
            # Check each category
            for category, emotions in self.emoji_map.items():
                if category in ['negative_emotions', 'threatening_emotions', 
                               'manipulative_emotions', 'positive_emotions', 'neutral_emotions']:
                    for emotion_type, emoji_list in emotions.items():
                        if emoji_char in emoji_list:
                            if category == 'negative_emotions':
                                sentiment_scores['negative'] += 1.0
                            elif category == 'threatening_emotions':
                                sentiment_scores['threatening'] += 1.0
                            elif category == 'manipulative_emotions':
                                sentiment_scores['manipulative'] += 1.0
                            elif category == 'positive_emotions':
                                sentiment_scores['positive'] += 1.0
                            elif category == 'neutral_emotions':
                                sentiment_scores['neutral'] += 1.0
        
        # Normalize scores
        total_emojis = len(emojis)
        if total_emojis > 0:
            for key in sentiment_scores:
                sentiment_scores[key] /= total_emojis
        
        return sentiment_scores
    
    def normalize_emojis(self, text: str) -> str:
        """Normalize emojis to their text descriptions."""
        return emoji.demojize(text)


class SlangProcessor:
    """Process slang and informal language for harassment detection."""
    
    def __init__(self, slang_path: str = "assets/slang.json"):
        """Initialize slang processor with dictionary."""
        self.slang_dict = self._load_slang_dict(slang_path)
        
    def _load_slang_dict(self, path: str) -> Dict:
        """Load slang dictionary from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Slang dictionary not found at {path}. Using default dictionary.")
            return self._get_default_slang_dict()
    
    def _get_default_slang_dict(self) -> Dict:
        """Default slang dictionary if file not found."""
        return {
            "harassment_indicators": {
                "direct_threats": ["kill yourself", "kys", "die", "suicide"],
                "sexual_harassment": ["slut", "whore", "bitch", "hoe"],
                "body_shaming": ["fat", "ugly", "disgusting", "gross"]
            }
        }
    
    def detect_harassment_patterns(self, text: str) -> Dict[str, float]:
        """Detect harassment patterns in text."""
        text_lower = text.lower()
        pattern_scores = {}
        
        for category, patterns in self.slang_dict.items():
            category_score = 0.0
            for pattern_type, phrases in patterns.items():
                for phrase in phrases:
                    if phrase.lower() in text_lower:
                        category_score += 1.0
            
            pattern_scores[category] = category_score
        
        # Normalize scores by text length
        text_length = len(text.split())
        if text_length > 0:
            for key in pattern_scores:
                pattern_scores[key] /= text_length
        
        return pattern_scores
    
    def replace_slang(self, text: str) -> str:
        """Replace slang with normalized terms."""
        # This is a simplified version - in practice, you'd want more sophisticated replacement
        replacements = {
            'kys': 'kill yourself',
            'thot': 'that hoe over there',
            'hoe': 'prostitute',
            'slut': 'promiscuous person'
        }
        
        text_lower = text.lower()
        for slang, replacement in replacements.items():
            text_lower = text_lower.replace(slang, replacement)
        
        return text_lower


class BehaviorAnalyzer:
    """Analyze behavioral patterns in text for harassment detection."""
    
    def __init__(self):
        """Initialize behavior analyzer."""
        self.nlp = spacy.load("en_core_web_sm")
        
    def analyze_communication_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze communication patterns that might indicate harassment."""
        doc = self.nlp(text)
        
        patterns = {
            'repetition': self._detect_repetition(text),
            'escalation': self._detect_escalation(text),
            'isolation_tactics': self._detect_isolation_tactics(text),
            'power_imbalance': self._detect_power_imbalance(doc),
            'emotional_manipulation': self._detect_emotional_manipulation(text)
        }
        
        return patterns
    
    def _detect_repetition(self, text: str) -> float:
        """Detect repetitive patterns in text."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate repetition score
        max_repetition = max(word_counts.values()) if word_counts else 1
        return min(max_repetition / len(words), 1.0)
    
    def _detect_escalation(self, text: str) -> float:
        """Detect escalation patterns in text."""
        escalation_indicators = [
            'getting worse', 'escalating', 'increasing', 'more intense',
            'stepping up', 'taking it further', 'next level', 'again',
            'still', 'continues', 'persistent', 'ongoing'
        ]
        
        text_lower = text.lower()
        escalation_count = sum(1 for indicator in escalation_indicators 
                             if indicator in text_lower)
        
        return min(escalation_count / len(escalation_indicators), 1.0)
    
    def _detect_isolation_tactics(self, text: str) -> float:
        """Detect isolation tactics in text."""
        isolation_indicators = [
            'nobody likes you', 'everyone hates you', 'you have no friends',
            'you\'re alone', 'no one cares', 'you\'re nothing',
            'not invited', 'left out', 'ignored', 'blocked'
        ]
        
        text_lower = text.lower()
        isolation_count = sum(1 for indicator in isolation_indicators 
                            if indicator in text_lower)
        
        return min(isolation_count / len(isolation_indicators), 1.0)
    
    def _detect_power_imbalance(self, doc) -> float:
        """Detect power imbalance indicators."""
        power_indicators = [
            'boss', 'teacher', 'parent', 'adult', 'authority', 'superior',
            'older', 'experienced', 'in charge', 'your superior'
        ]
        
        power_count = 0
        for token in doc:
            if token.text.lower() in power_indicators:
                power_count += 1
        
        return min(power_count / len(power_indicators), 1.0)
    
    def _detect_emotional_manipulation(self, text: str) -> float:
        """Detect emotional manipulation patterns."""
        manipulation_indicators = [
            'you\'re crazy', 'you\'re imagining', 'that never happened',
            'you\'re overreacting', 'you\'re too sensitive', 'you\'re paranoid',
            'you made me', 'it\'s your fault', 'because of you', 'you caused this'
        ]
        
        text_lower = text.lower()
        manipulation_count = sum(1 for indicator in manipulation_indicators 
                               if indicator in text_lower)
        
        return min(manipulation_count / len(manipulation_indicators), 1.0)


class FeatureExtractor:
    """Extract features from text for machine learning models."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text."""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        }
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features using TextBlob."""
        blob = TextBlob(text)
        
        features = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'positive_words': len([word for word in blob.words if blob.sentiment.polarity > 0]),
            'negative_words': len([word for word in blob.words if blob.sentiment.polarity < 0])
        }
        
        return features
    
    def extract_ngram_features(self, text: str, n: int = 2) -> List[str]:
        """Extract n-gram features from text."""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams


class TextPreprocessor:
    """Main text preprocessing class that combines all preprocessing components."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize text preprocessor with configuration."""
        self.config = config or {}
        self.emoji_processor = EmojiProcessor()
        self.slang_processor = SlangProcessor()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.feature_extractor = FeatureExtractor()
        
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text preprocessing for harassment detection."""
        if not text or not isinstance(text, str):
            return self._get_empty_result()
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract features
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'emoji_analysis': self.emoji_processor.analyze_emoji_sentiment(text),
            'slang_analysis': self.slang_processor.detect_harassment_patterns(text),
            'behavior_analysis': self.behavior_analyzer.analyze_communication_patterns(text),
            'linguistic_features': self.feature_extractor.extract_linguistic_features(text),
            'sentiment_features': self.feature_extractor.extract_sentiment_features(text),
            'ngram_features': self.feature_extractor.extract_ngram_features(text)
        }
        
        # Calculate overall risk score
        result['risk_score'] = self._calculate_risk_score(result)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags if configured
        if self.config.get('remove_mentions', True):
            text = re.sub(r'@\w+', '', text)
        
        if self.config.get('remove_hashtags', False):
            text = re.sub(r'#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize emojis if configured
        if self.config.get('normalize_emojis', True):
            text = self.emoji_processor.normalize_emojis(text)
        
        return text
    
    def _calculate_risk_score(self, result: Dict[str, Any]) -> float:
        """Calculate overall risk score based on all features."""
        risk_factors = []
        
        # Emoji risk
        emoji_risk = (result['emoji_analysis']['negative'] + 
                     result['emoji_analysis']['threatening'] + 
                     result['emoji_analysis']['manipulative'])
        risk_factors.append(emoji_risk)
        
        # Slang risk
        slang_risk = sum(result['slang_analysis'].values())
        risk_factors.append(slang_risk)
        
        # Behavior risk
        behavior_risk = sum(result['behavior_analysis'].values()) / len(result['behavior_analysis'])
        risk_factors.append(behavior_risk)
        
        # Sentiment risk (negative polarity)
        sentiment_risk = max(0, -result['sentiment_features']['polarity'])
        risk_factors.append(sentiment_risk)
        
        # Calculate weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Adjust weights as needed
        risk_score = sum(w * r for w, r in zip(weights, risk_factors))
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'original_text': '',
            'cleaned_text': '',
            'emoji_analysis': {'negative': 0.0, 'threatening': 0.0, 'manipulative': 0.0, 'positive': 0.0, 'neutral': 0.0},
            'slang_analysis': {},
            'behavior_analysis': {},
            'linguistic_features': {},
            'sentiment_features': {'polarity': 0.0, 'subjectivity': 0.0, 'positive_words': 0, 'negative_words': 0},
            'ngram_features': [],
            'risk_score': 0.0
        }
    
    def batch_preprocess(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Preprocess multiple texts in batch."""
        return [self.preprocess_text(text) for text in texts]
