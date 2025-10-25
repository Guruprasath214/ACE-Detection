#!/usr/bin/env python3
"""
Predict from CSV file using trained ACE models.
"""

import pandas as pd
import sys
import csv
from pathlib import Path

# Add ACE-Detection to path
sys.path.append(str(Path("ACE-Detection")))
sys.path.append(str(Path(".")))

try:
    from enhanced_predictor import EnhancedACEPredictor  # type: ignore
except ImportError:
    try:
        from ACE_Detection.enhanced_predictor import EnhancedACEPredictor  # type: ignore
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("enhanced_predictor", str(Path("ACE-Detection/enhanced_predictor.py")))
        enhanced_predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_predictor)
        EnhancedACEPredictor = enhanced_predictor.EnhancedACEPredictor

def predict_from_csv(csv_path: str, output_path: str = None):
    """
    Predict harassment from CSV file.

    Args:
        csv_path: Path to CSV file with 'text' column
        output_path: Path to save predictions (optional)
    """
    # Load data with error handling for malformed CSV
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        # Fallback to manual CSV parsing for malformed files
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        df = pd.DataFrame(data)

    if 'text' not in df.columns:
        raise ValueError("CSV must contain 'text' column")

    # Initialize predictor
    predictor = EnhancedACEPredictor()

    # Make predictions
    predictions = []
    for text in df['text']:
        result = predictor.predict_single(text)
        predictions.append({
            'text': text,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'base_scores': result['base_scores'],
            'weighted_score': result['weighted_score']
        })

    # Create results DataFrame
    results_df = pd.DataFrame(predictions)

    # Print summary
    print("Prediction Summary:")
    print(f"Total texts: {len(results_df)}")
    print(f"Harassment detected: {results_df['prediction'].sum()}")
    print(f"Normal texts: {(results_df['prediction'] == 0).sum()}")
    print(".2f")

    # Print sample predictions
    print("\nSample Predictions:")
    for i, row in results_df.head(10).iterrows():
        label = "HARASSMENT" if row['prediction'] == 1 else "NORMAL"
        print(f"{i+1}. [{label}] {row['text'][:50]}... (conf: {row['confidence']:.2f})")

    # Save if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")

    return results_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_from_csv.py <csv_path> [output_path]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    predict_from_csv(csv_path, output_path)
