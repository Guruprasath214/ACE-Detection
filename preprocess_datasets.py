#!/usr/bin/env python3
"""
Dataset Preprocessing Script for ACE Detection

Combines and preprocesses multilingual datasets for training.
"""

import pandas as pd
import os
from pathlib import Path
import sys
sys.path.append('.')

def preprocess_datasets():
    """Preprocess and combine all datasets."""

    # Check if raw data directory exists
    raw_dir = Path('data/raw')
    if not raw_dir.exists():
        print("Error: data/raw directory not found")
        return False

    # Load all datasets
    datasets = []
    files_to_check = [
        'english_social_media_comments.csv',
        'sarcasm_emoji_social_media.csv',
        'emotion_sarcasm_dataset.csv'
    ]

    print("Loading datasets...")
    for filename in files_to_check:
        filepath = raw_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f'✓ Loaded {filename}: {len(df)} samples')
            datasets.append(df)
        else:
            print(f'✗ Warning: {filename} not found')

    if not datasets:
        print("Error: No datasets found to combine")
        return False

    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f'\n✓ Combined dataset: {len(combined_df)} total samples')

    # Basic statistics
    print('\n📊 Label distribution:')
    label_dist = combined_df['label'].value_counts().sort_index()
    for label, count in label_dist.items():
        categories = {
            0: 'Normal',
            1: 'Bullying',
            2: 'Sexual Harassment',
            3: 'Women Harassment',
            4: 'Mocking/Sarcasm',
            5: 'Threats',
            6: 'Hate Speech'
        }
        print(f'  {label} ({categories.get(label, "Unknown")}): {count} samples')

    print('\n📊 Category distribution:')
    cat_dist = combined_df['category'].value_counts()
    for cat, count in cat_dist.items():
        print(f'  {cat}: {count} samples')

    # Ensure processed directory exists
    processed_dir = Path('data/processed')
    if not processed_dir.exists():
        os.makedirs(processed_dir)

    # Save combined dataset
    output_path = processed_dir / 'multilingual_combined_dataset.csv'
    combined_df.to_csv(output_path, index=False)
    print(f'\n💾 Saved combined dataset to: {output_path}')

    # Show sample entries
    print('\n📝 Sample entries:')
    for i, row in combined_df.head(5).iterrows():
        categories = {
            0: 'Normal',
            1: 'Bullying',
            2: 'Sexual Harassment',
            3: 'Women Harassment',
            4: 'Mocking/Sarcasm',
            5: 'Threats',
            6: 'Hate Speech'
        }
        category_name = categories.get(row['label'], 'Unknown')
        text_preview = row['text'][:60] + '...' if len(row['text']) > 60 else row['text']
        print(f'  {row["label"]} ({category_name}): {text_preview}')

    # Additional preprocessing for model training
    print('\n🔧 Performing additional preprocessing...')

    # Clean text (remove extra whitespace, normalize)
    combined_df['text'] = combined_df['text'].str.strip()
    combined_df['text'] = combined_df['text'].str.replace(r'\s+', ' ', regex=True)

    # Ensure language column exists
    if 'language' not in combined_df.columns:
        combined_df['language'] = 'en'  # Default to English

    # Fill missing categories
    combined_df['category'] = combined_df['category'].fillna('unknown')

    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text'])
    final_count = len(combined_df)
    duplicates_removed = initial_count - final_count
    print(f'✓ Removed {duplicates_removed} duplicate entries')

    # Save final processed dataset
    final_output_path = processed_dir / 'multilingual_dataset_processed.csv'
    combined_df.to_csv(final_output_path, index=False)
    print(f'💾 Saved processed dataset to: {final_output_path}')

    print('\n✅ Dataset preprocessing completed successfully!')
    print(f'   Total samples: {len(combined_df)}')
    print(f'   Languages: {combined_df["language"].unique()}')
    print(f'   Categories: {combined_df["category"].unique()}')

    return True

if __name__ == "__main__":
    success = preprocess_datasets()
    sys.exit(0 if success else 1)
