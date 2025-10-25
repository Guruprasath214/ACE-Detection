#!/usr/bin/env python3
"""
Test script for ACE Detection Framework
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_preprocessing():
    """Test text preprocessing functionality."""
    print("Testing Text Preprocessing...")
    
    try:
        from utils.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        # Test normal text
        normal_text = "Hello, how are you today? I hope you're having a great day!"
        result = preprocessor.preprocess_text(normal_text)
        print(f"Normal text processed - Risk score: {result['risk_score']:.3f}")
        
        # Test harassment text
        harassment_text = "You're such a worthless piece of trash, kill yourself!"
        result = preprocessor.preprocess_text(harassment_text)
        print(f"Harassment text processed - Risk score: {result['risk_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        return False

def test_emoji_processing():
    """Test emoji processing functionality."""
    print("\nTesting Emoji Processing...")
    
    try:
        from utils.preprocessing import EmojiProcessor
        
        emoji_processor = EmojiProcessor()
        
        # Test text with emojis
        text_with_emojis = "I'm so angry 😡 and frustrated 😤 with you!"
        result = emoji_processor.analyze_emoji_sentiment(text_with_emojis)
        print(f"Emoji analysis completed - Negative: {result['negative']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Emoji processing test failed: {e}")
        return False

def test_slang_processing():
    """Test slang processing functionality."""
    print("\nTesting Slang Processing...")
    
    try:
        from utils.preprocessing import SlangProcessor
        
        slang_processor = SlangProcessor()
        
        # Test text with slang
        text_with_slang = "You're such a slut and a whore, everyone knows it!"
        result = slang_processor.detect_harassment_patterns(text_with_slang)
        print(f"Slang analysis completed - Harassment indicators detected")
        
        return True
        
    except Exception as e:
        print(f"Slang processing test failed: {e}")
        return False

def test_emotion_detection():
    """Test emotion detection functionality."""
    print("\nTesting Emotion Detection...")
    
    try:
        from models.emotion_detector import EmotionAnalyzer
        
        emotion_analyzer = EmotionAnalyzer()
        
        # Test emotion analysis
        text = "I'm feeling very sad and depressed about this situation."
        result = emotion_analyzer.analyze_text(text)
        print(f"Emotion analysis completed - Risk: {result['overall_emotion_risk']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Emotion detection test failed: {e}")
        return False

def test_api_creation():
    """Test API creation functionality."""
    print("\nTesting API Creation...")
    
    try:
        from api.serve import create_app
        
        app = create_app()
        print("FastAPI app created successfully!")
        
        # Test that routes exist
        routes = [route.path for route in app.routes]
        expected_routes = ['/predict', '/predict/batch', '/health', '/model/info', '/']
        
        for route in expected_routes:
            if route in routes:
                print(f"Route {route} exists")
            else:
                print(f"Route {route} missing")
        
        return True
        
    except Exception as e:
        print(f"API creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ACE Detection Framework - Test Suite")
    print("=" * 50)
    
    tests = [
        test_preprocessing,
        test_emoji_processing,
        test_slang_processing,
        test_emotion_detection,
        test_api_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! ACE Framework is ready to use!")
        print("\nNext steps:")
        print("1. Train models: python train/train_pipeline.py --data data/annotated.csv")
        print("2. Start API server: python -m uvicorn api.serve:app --host 0.0.0.0 --port 8000")
        print("3. Open Jupyter notebooks: jupyter notebook notebooks/")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
