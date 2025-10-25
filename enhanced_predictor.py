#!/usr/bin/env python3
"""
Enhanced ACE Detection Predictor

Improved prediction logic with expanded keyword detection and slang analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any

class EnhancedACEPredictor:
    """Enhanced ACE predictor with better detection logic."""

    def __init__(self):
        self.slang_data = self._load_slang_data()

    def _load_slang_data(self) -> Dict[str, Any]:
        """Load slang dictionary for enhanced detection."""
        try:
            slang_path = Path(__file__).parent / "assets" / "slang.json"
            if slang_path.exists():
                with open(slang_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Warning: Could not load slang data: {e}")
            return {}

    def _check_negation(self, text_lower: str, keyword: str, negation_words: list) -> bool:
        """Check if a keyword is negated in the text."""
        words = text_lower.split()

        # Handle multi-word keywords
        keyword_words = keyword.split()
        for i in range(len(words) - len(keyword_words) + 1):
            if words[i:i+len(keyword_words)] == keyword_words:
                keyword_start_index = i
                # Check for negation within 3 words before the keyword phrase
                for j in range(max(0, keyword_start_index - 3), keyword_start_index):
                    if any(neg_word in words[j] for neg_word in negation_words):
                        return True
                break

        # Fallback for single-word keywords (backward compatibility)
        if len(keyword_words) == 1:
            for i, word in enumerate(words):
                if keyword in word:
                    for j in range(max(0, i - 3), i):
                        if any(neg_word in words[j] for neg_word in negation_words):
                            return True
        return False

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Enhanced prediction logic with comprehensive keyword detection and context awareness.

        Args:
            text: Input text to analyze

        Returns:
            Prediction result dictionary
        """
        risk_score = 0.0
        text_lower = text.lower()

        # Context-aware keyword detection with negation handling
        # High-risk keywords (strong profanity and direct threats)
        high_risk_keywords = [
            'kill yourself', 'kill your self', 'kill urself', 'kys', 'die', 'suicide', 'end yourself', 'off yourself',
            'jump off', 'hang yourself', 'shoot yourself', 'cut yourself',
            'slut', 'whore', 'bitch', 'hoe', 'thot', 'prostitute', 'hooker',
            'fuck', 'shit', 'asshole', 'bastard', 'cunt', 'dick', 'pussy'
        ]

        # Medium-risk keywords
        medium_risk_keywords = [
            'hate', 'stupid', 'ugly', 'fat', 'loser', 'freak', 'weirdo', 'creep',
            'stalker', 'psycho', 'mental', 'crazy', 'insane', 'retard', 'idiot',
            'moron', 'dumb', 'worthless', 'pathetic', 'useless', 'nobody', 'alone'
        ]

        # Negation words that can reverse meaning
        negation_words = ['not', 'no', 'never', 'dont', "don't", 'wont', "won't", 'cant', "can't", 'isnt', "isn't", 'arent', "aren't"]

        # Check high-risk keywords first
        for keyword in high_risk_keywords:
            if keyword in text_lower:
                # Check if keyword is negated
                is_negated = self._check_negation(text_lower, keyword, negation_words)
                if not is_negated:
                    risk_score += 0.4  # Higher weight for strong profanity
                else:
                    risk_score += 0.0  # Zero score for negated strong profanity

        # Check medium-risk keywords
        for keyword in medium_risk_keywords:
            if keyword in text_lower:
                # Check if keyword is negated
                is_negated = self._check_negation(text_lower, keyword, negation_words)
                if not is_negated:
                    risk_score += 0.2
                else:
                    risk_score += 0.05

        # Load slang dictionary for better detection
        if self.slang_data:
            # Define all harassment indicator categories to check
            harassment_categories = [
                "harassment_indicators",
                "thanglish_harassment_indicators",
                "manglish_harassment_indicators",
                "tamil_harassment_indicators",
                "malayalam_harassment_indicators"
            ]

            # Check for harassment indicators with context
            for category_name in harassment_categories:
                if category_name in self.slang_data:
                    for sub_category, words in self.slang_data[category_name].items():
                        for word in words:
                            if word.lower() in text_lower:
                                # Apply same negation logic
                                is_negated = False
                                words_list = text_lower.split()
                                word_index = -1

                                for i, w in enumerate(words_list):
                                    if word.lower() in w:
                                        word_index = i
                                        break

                                if word_index >= 0:
                                    for i in range(max(0, word_index - 2), word_index):
                                        if any(neg_word in words_list[i] for neg_word in negation_words):
                                            is_negated = True
                                            break

                                if not is_negated:
                                    risk_score += 0.3
                                break

            # Check for escalation indicators
            for category, words in self.slang_data.get("escalation_indicators", {}).items():
                for word in words:
                    if word.lower() in text_lower:
                        risk_score += 0.2
                        break

            # Check for isolation tactics
            for category, words in self.slang_data.get("isolation_tactics", {}).items():
                for word in words:
                    if word.lower() in text_lower:
                        risk_score += 0.25
                        break

        # Check for excessive punctuation (aggression indicator)
        if text.count('!') > 2 or text.count('?') > 3:
            risk_score += 0.1

        # Check for caps (shouting)
        if len([c for c in text if c.isupper()]) > len(text) * 0.3:
            risk_score += 0.1

        # Check for emoji sentiment
        try:
            import emoji
            emoji_list = emoji.emoji_list(text)
            negative_emojis = ['😠', '😡', '🤬', '💢', '😢', '😭', '💔', '😞', '😨', '😰', '😱', '😖', '😤', '😠', '🤬']
            for emoji_char in emoji_list:
                if emoji_char['emoji'] in negative_emojis:
                    risk_score += 0.15
        except:
            pass

        # Special handling for common phrases (with negation check)
        if ('fuck you' in text_lower or 'fuck off' in text_lower) and not any(neg in text_lower.split()[:2] for neg in negation_words):
            risk_score += 0.4
        if ('go die' in text_lower or 'go and die' in text_lower) and not any(neg in text_lower.split()[:3] for neg in negation_words):
            risk_score += 0.3
        if 'you are useless' in text_lower and not any(neg in text_lower.split()[:2] for neg in negation_words):
            risk_score += 0.2

        # Cap risk_score at 1.0 to prevent validation errors
        risk_score = min(risk_score, 1.0)

        prediction = 1 if risk_score > 0.3 else 0
        confidence = risk_score

        return {
            'prediction': prediction,
            'confidence': confidence,
            'base_scores': {'bert': min(risk_score, 1.0), 'cnn': min(risk_score, 1.0), 'emotion': min(risk_score, 1.0)},
            'weighted_score': min(risk_score, 1.0)  # Cap at 1.0 for API compatibility
        }

    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """Provide explanation for the prediction."""
        result = self.predict_single(text)
        return {
            **result,
            'explanation': f"Enhanced keyword-based analysis. Risk score: {result['weighted_score']:.2f}",
            'preprocessing_analysis': {'risk_score': result['weighted_score']}
        }
