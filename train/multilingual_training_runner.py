"""
Multilingual training runner for ACE.

This script wraps the existing ACETrainer to provide:
- language detection (using langdetect) when language column is missing
- optional transliteration for Indic languages (hooked, optional dependency)
- dataset validation for required columns: text, label, optional language

Usage:
    python train/multilingual_training_runner.py --data path/to/data.csv --config deployment/config_multilingual.yaml

"""
import argparse
from pathlib import Path
import pandas as pd
import logging
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train.train_pipeline import ACETrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional libs
try:
    from langdetect import detect
except Exception:
    detect = None

try:
    # Indic transliteration optional; not required
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    logger.info("Successfully loaded indic-transliteration package")
except ImportError as e:
    logger.warning(f"indic-transliteration package not found: {e}. Transliteration will be disabled.")
    transliterate = None
    sanscript = None
except Exception as e:
    logger.error(f"Unexpected error loading indic-transliteration: {e}. Transliteration will be disabled.")
    transliterate = None
    sanscript = None


def ensure_columns(df: pd.DataFrame):
    required = ['text', 'label']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Input CSV must contain column: {c}")


def detect_language(text: str) -> str:
    if not text or not detect:
        return 'en'
    try:
        lang = detect(text)
        return lang
    except Exception:
        return 'en'


def maybe_transliterate(text: str, lang: str) -> str:
    if not transliterate:
        return text
    if lang in ('ta', 'ml'):
        try:
            # Transliterate to IAST/Latin for downstream tokenizers if desired
            return transliterate(text, sanscript.TAMIL if lang == 'ta' else sanscript.MALAYALAM, sanscript.ITRANS)
        except Exception:
            return text
    return text


def prepare_multilingual_csv(path: Path, language_column: str = 'language', transliterate_flag: bool = False):
    df = pd.read_csv(path)
    ensure_columns(df)

    # Ensure language column exists
    if language_column not in df.columns:
        if detect is None:
            logger.warning('langdetect not installed; defaulting language to en')
            df[language_column] = 'en'
        else:
            logger.info('Detecting language for samples...')
            df[language_column] = df['text'].fillna('').astype(str).apply(detect_language)

    # Optionally transliterate
    if transliterate_flag:
        logger.info('Applying optional transliteration to Indic languages')
        df['text'] = df.apply(lambda r: maybe_transliterate(r['text'], r[language_column]), axis=1)

    # Filter to target languages
    return df


def main():
    parser = argparse.ArgumentParser(description='Multilingual ACE training runner')
    parser.add_argument('--data', required=True, help='Path to CSV file with text,label and optional language columns')
    parser.add_argument('--config', default='deployment/config_multilingual.yaml', help='Path to config yaml')
    parser.add_argument('--save_dir', default='models/saved', help='Directory to save trained models')
    parser.add_argument('--transliterate', action='store_true', help='Transliterate Tamil/Malayalam to Latin script before training')

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = prepare_multilingual_csv(data_path, language_column='language', transliterate_flag=args.transliterate)

    # Save a processed copy
    proc_path = Path('data/processed/multilingual_processed.csv')
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_path, index=False)
    logger.info(f'Processed data saved to {proc_path}')

    # Create ACETrainer and run full training using processed CSV
    trainer = ACETrainer(config_path=args.config, save_dir=args.save_dir)
    results = trainer.run_full_training(str(proc_path))

    logger.info('Training finished. Results:')
    for k, v in results.items():
        logger.info(f"{k}: {v}")


if __name__ == '__main__':
    main()
