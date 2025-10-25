#!/usr/bin/env python3
"""
Test script for the updated ACE ensemble with 6 models
"""

import sys
import os
sys.path.append('.')

def test_ensemble():
    from models.ensemble import ACEEnsemble
    import yaml
    from pathlib import Path

    # Load config
    config_path = "deployment/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update config to include all 6 models
    config['models']['ensemble']['weights'] = {
        'bert': 0.2,
        'cnn': 0.2,
        'svm': 0.2,
        'deep_emoji': 0.2,
        'emotion': 0.2,
        'contextual': 0.0  # Contextual model weight (can be adjusted)
    }
    config['models']['deep_emoji'] = {
        'model_name': 'cardiffnlp/twitter-roberta-base-emoji',
        'max_length': 128
    }

    print('Testing ACE Ensemble instantiation...')
    try:
        ace = ACEEnsemble(config)
        print('✓ ACE Ensemble instantiated successfully')
    except Exception as e:
        print(f'✗ Failed to instantiate ACE Ensemble: {e}')
        return False

    print('Testing feature extraction...')
    test_texts = ['Hello world', 'You are stupid!']
    try:
        features = ace.extract_base_features(test_texts)
        expected_keys = ['bert_scores', 'cnn_scores', 'svm_scores', 'deep_emoji_scores', 'emotion_scores', 'contextual_scores']
        if all(key in features for key in expected_keys):
            print('✓ Feature extraction successful - all 6 model scores extracted')
            for key, scores in features.items():
                print(f'  {key}: {len(scores)} scores')
        else:
            print(f'✗ Missing feature keys: {[k for k in expected_keys if k not in features]}')
            return False
    except Exception as e:
        print(f'✗ Feature extraction failed: {e}')
        return False

    print('Testing single prediction...')
    try:
        result = ace.predict_single('Hello world')
        if 'prediction' in result and 'base_scores' in result:
            print('✓ Single prediction successful')
            print(f'  Prediction: {result["prediction"]}')
            print(f'  Base scores keys: {list(result["base_scores"].keys())}')
            if len(result["base_scores"]) != 5:
                print('✗ Single prediction should have 5 base scores (contextual not included in weighted score)')
                return False
        else:
            print('✗ Single prediction missing expected keys')
            return False
    except Exception as e:
        print(f'✗ Single prediction failed: {e}')
        return False

    print('Testing batch prediction...')
    try:
        results = ace.predict_batch(test_texts)
        if len(results) == len(test_texts):
            print('✓ Batch prediction successful')
        else:
            print('✗ Batch prediction returned wrong number of results')
            return False
    except Exception as e:
        print(f'✗ Batch prediction failed: {e}')
        return False

    print('Testing model importance calculation...')
    try:
        labels = [0, 1]
        importance = ace.get_model_importance(test_texts, labels)
        expected_keys = ['bert_accuracy', 'cnn_accuracy', 'svm_accuracy', 'deep_emoji_accuracy', 'emotion_accuracy', 'ensemble_accuracy', 'improvement']
        if all(key in importance for key in expected_keys):
            print('✓ Model importance calculation successful - all 5 models included (contextual not in importance calc)')
        else:
            print(f'✗ Missing importance keys: {[k for k in expected_keys if k not in importance]}')
            return False
    except Exception as e:
        print(f'✗ Model importance calculation failed: {e}')
        return False

    print('Testing meta-classifier training...')
    try:
        train_texts = ['Hello', 'You are great', 'Bad text', 'Terrible']
        train_labels = [0, 0, 1, 1]
        ace.train_meta_classifier(train_texts, train_labels, num_epochs=2)
        print('✓ Meta-classifier training successful')
    except Exception as e:
        print(f'✗ Meta-classifier training failed: {e}')
        return False

    print('All critical-path tests passed!')
    return True

if __name__ == "__main__":
    success = test_ensemble()
    sys.exit(0 if success else 1)
