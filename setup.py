#!/usr/bin/env python3
"""
ACE Detection Framework Setup Script
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("ACE Detection Framework - Setup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements_minimal.txt").exists():
        print("❌ Please run this script from the ACE-Detection directory")
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements_minimal.txt", "Installing Python packages"):
        return False
    
    # Install spaCy
    if not run_command("pip install spacy", "Installing spaCy"):
        return False
    
    # Download NLTK data
    if not run_command('python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'stopwords\'); nltk.download(\'wordnet\'); nltk.download(\'punkt_tab\')"', "Downloading NLTK data"):
        return False
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model"):
        return False
    
    # Run tests
    if not run_command("python test_framework.py", "Running framework tests"):
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nYour ACE Detection Framework is ready to use!")
    print("\nNext steps:")
    print("1. Train models: python train/train_pipeline.py --data data/annotated.csv")
    print("2. Start API server: python -m uvicorn api.serve:app --host 0.0.0.0 --port 8000")
    print("3. Open Jupyter notebooks: jupyter notebook notebooks/")
    print("4. View API documentation: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
