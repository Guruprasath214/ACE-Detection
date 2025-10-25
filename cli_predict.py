#!/usr/bin/env python3
"""
ACE Detection CLI Tool

Command-line interface for real-time harassment detection.
Usage: python cli_predict.py "your text here"
Or run interactively: python cli_predict.py
"""

import sys
import requests
import json
from typing import Optional

class ACECLI:
    """Command-line interface for ACE Detection."""

    def __init__(self, api_url: str = "http://127.0.0.1:8001"):
        """Initialize CLI with API endpoint."""
        self.api_url = api_url
        self.predict_endpoint = f"{api_url}/predict"

    def predict_text(self, text: str, include_explanation: bool = True) -> dict:
        """
        Predict harassment for given text using the API.

        Args:
            text: Input text to analyze
            include_explanation: Whether to include detailed explanation

        Returns:
            Prediction result dictionary
        """
        payload = {
            "text": text,
            "include_explanation": include_explanation
        }

        try:
            response = requests.post(
                self.predict_endpoint,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {str(e)}",
                "prediction": None,
                "confidence": 0.0
            }

    def format_result(self, result: dict) -> str:
        """Format prediction result for terminal display."""
        if "error" in result:
            return f"❌ Error: {result['error']}"

        prediction = result.get('prediction', 0)
        confidence = result.get('confidence', 0.0)
        risk_score = result.get('risk_score', 0.0)

        # Determine prediction label
        label = "🚨 HARASSMENT DETECTED" if prediction == 1 else "✅ SAFE CONTENT"
        emoji = "🚨" if prediction == 1 else "✅"

        # Format output
        output = f"""
{emoji} ACE Detection Result
{'='*50}
Input Text: {result.get('text', 'N/A')}
Prediction: {label}
Confidence: {confidence:.3f}
Risk Score: {risk_score:.3f}
"""

        # Add explanation if available
        if result.get('explanation'):
            output += f"\nExplanation: {result['explanation']}\n"

        # Add base scores if available
        base_scores = result.get('base_scores', {})
        if base_scores:
            output += "\nModel Scores:\n"
            for model, score in base_scores.items():
                output += f"  {model.title()}: {score:.3f}\n"

        return output

    def interactive_mode(self):
        """Run in interactive mode, prompting for text input."""
        print("🤖 ACE Detection CLI - Interactive Mode")
        print("Enter text to analyze (or 'quit' to exit):")
        print("-" * 50)

        while True:
            try:
                text = input("\nEnter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif not text:
                    print("⚠️  Please enter some text to analyze.")
                    continue

                print("\n🔍 Analyzing...")
                result = self.predict_text(text)
                print(self.format_result(result))

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main CLI entry point."""
    cli = ACECLI()

    # Check if text provided as argument
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"🔍 Analyzing: {text}")
        result = cli.predict_text(text)
        print(cli.format_result(result))
    else:
        # Interactive mode
        cli.interactive_mode()

if __name__ == "__main__":
    main()
