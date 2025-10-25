#!/usr/bin/env python3
"""
ACE Detection Framework Demo
"""

import sys
from pathlib import Path
import json
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def demo_text_preprocessing():
    """Demonstrate text preprocessing."""
    print("=" * 60)
    print("ACE DETECTION FRAMEWORK - TEXT PREPROCESSING DEMO")
    print("=" * 60)
    
    try:
        from utils.preprocessing import TextPreprocessor
    except Exception as e:
        print(f"Could not import TextPreprocessor: {e}")
        print("Skipping text preprocessing demo...")
        return
    
    preprocessor = TextPreprocessor()
    
    # Test texts
    test_texts = [
        "Hello! How are you today? I hope you're having a wonderful day!",
        "You're such a worthless piece of trash, kill yourself!",
        "Great job on the presentation! Your hard work really paid off.",
        "I'm going to find you and make you pay for what you did.",
        "Thanks for being such a supportive friend. You're amazing!",
        "You're a slut and everyone knows it. You deserve to die."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: '{text}'")
        result = preprocessor.preprocess_text(text)
        
        print(f"   Risk Score: {result['risk_score']:.3f}")
        print(f"   Emoji Analysis: {result['emoji_analysis']}")
        print(f"   Slang Analysis: {result['slang_analysis']}")
        print(f"   Behavior Analysis: {result['behavior_analysis']}")
        
        # Determine if it's harassment
        if result['risk_score'] > 0.1:
            print("   CLASSIFICATION: HARASSMENT")
        else:
            print("   CLASSIFICATION: NORMAL")

def demo_emotion_detection():
    """Demonstrate emotion detection."""
    print("\n" + "=" * 60)
    print("ACE DETECTION FRAMEWORK - EMOTION DETECTION DEMO")
    print("=" * 60)
    
    try:
        from models.emotion_detector import EmotionAnalyzer
    except Exception as e:
        print(f"Could not import EmotionAnalyzer: {e}")
        print("Skipping emotion detection demo...")
        return
    
    emotion_analyzer = EmotionAnalyzer()
    
    # Test texts
    test_texts = [
        "I'm feeling so happy and excited about this opportunity!",
        "I'm really angry and frustrated with this situation.",
        "I'm scared and worried about what might happen next.",
        "I'm sad and disappointed about the outcome."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: '{text}'")
        result = emotion_analyzer.analyze_text(text)
        
        print(f"   Dominant Emotion: {result['emotion_prediction']['dominant_emotion']}")
        print(f"   Confidence: {result['emotion_prediction']['dominant_confidence']:.3f}")
        print(f"   Harassment Risk: {result['emotion_prediction']['harassment_risk']:.3f}")
        print(f"   Overall Risk: {result['overall_emotion_risk']:.3f}")

async def demo_api_prediction():
    """Demonstrate API prediction functionality."""
    print("\n" + "=" * 60)
    print("ACE DETECTION FRAMEWORK - API PREDICTION DEMO")
    print("=" * 60)

    try:
        from api.serve import ACEAPIServer

        # Initialize server
        server = ACEAPIServer()
    except Exception as e:
        print(f"Could not initialize API server: {e}")
        print("Skipping API prediction demo...")
        return

    # Test texts
    test_texts = [
        "You're such a beautiful and kind person!",
        "Kill yourself you worthless piece of garbage!",
        "Congratulations on your achievement!",
        "I'm going to make you suffer for this!"
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: '{text}'")

        try:
            # Create text input
            from api.serve import TextInput
            text_input = TextInput(text=text, include_explanation=True)

            # Get prediction (await the coroutine)
            result = await server.predict_single(text_input)

            print(f"   Prediction: {'HARASSMENT' if result.prediction == 1 else 'NORMAL'}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Risk Score: {result.risk_score:.3f}")
            print(f"   Base Scores: {result.base_scores}")
            if result.explanation:
                print(f"   Explanation: {result.explanation}")

        except Exception as e:
            print(f"   Error: {e}")

async def main():
    """Run the demo."""
    print("ACE DETECTION FRAMEWORK - COMPREHENSIVE DEMO")
    print("Advanced Cyberbullying & Emotion Detection")
    print("=" * 60)

    try:
        # Run demos
        demo_text_preprocessing()
        demo_emotion_detection()
        await demo_api_prediction()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe ACE Detection Framework is working correctly!")
        print("\nNext steps:")
        print("1. Train models with your data")
        print("2. Deploy the API server")
        print("3. Integrate with your application")

    except Exception as e:
        print(f"\nError running demo: {e}")
        return False

    return True

if __name__ == "__main__":
    asyncio.run(main())
