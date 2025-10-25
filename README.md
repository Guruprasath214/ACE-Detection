# ACE: Advanced Cyberbullying & Emotion Detection Framework

A comprehensive deep learning framework for detecting women harassment and cyber abuse in social media platforms using advanced AI models including CNN, Transformers (BERT), and Emotion detection.

## 🎯 Overview

ACE (Advanced Cyberbullying & Emotion Detection) is a single, adaptive algorithm designed to overcome the limitations of existing models by:

- Using advanced models like CNN, Transformers (BERT), and Emotion detection
- Including emoji, slang, and behavior analysis to catch hidden patterns
- Providing real-time monitoring to reduce harm and create safer online spaces

## 🏗️ Architecture

```
ACE-Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── annotated.csv            # Training data (text, label)
│   ├── test.csv                 # Optional test set
│   ├── raw/                     # Raw data dumps (tweets, comments)
│   └── processed/               # After cleaning/preprocessing
│
├── assets/
│   ├── slang.json               # Slang dictionary
│   └── emoji_map.json           # Optional emoji descriptor map
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py         # Normalization, emoji/slang handling, feature extraction
│
├── models/
│   ├── __init__.py
│   ├── bert_model.py            # BERT-based abuse detector
│   ├── cnn_model.py             # CNN-based text classifier
│   ├── emotion_detector.py      # Emotion model wrapper
│   ├── ensemble.py              # ACE meta-classifier
│   └── saved/                   # Trained model weights/checkpoints
│
├── train/
│   ├── __init__.py
│   ├── train_pipeline.py        # Main training script
│   ├── fine_tune_bert.py        # (Optional) full BERT fine-tuning
│   └── evaluation.py            # Metrics and reports
│
├── api/
│   ├── __init__.py
│   └── serve.py                 # FastAPI server (real-time API)
│
├── notebooks/
│   ├── EDA.ipynb                # Data exploration
│   ├── model_testing.ipynb      # Testing different models
│   └── demo.ipynb               # Quick demo for reports/presentation
│
└── deployment/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── start.sh
    ├── config.yaml
    └── k8s/                     # Kubernetes manifests if needed
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ACE-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python -c "import nltk; nltk.download('all')"
python -m spacy download en_core_web_sm
```

### Training

1. Prepare your data in `data/annotated.csv` with columns: `text`, `label`
2. Run training:
```bash
python train/train_pipeline.py
```

### API Server

Start the real-time detection API:
```bash
python api/serve.py
```

The API will be available at `http://localhost:8000`

## 📊 Features

### Multi-Model Ensemble
- **BERT-based Detection**: Contextual understanding using transformer models
- **CNN Classification**: Pattern recognition in text sequences
- **Emotion Analysis**: Sentiment and emotional state detection
- **ACE Meta-Classifier**: Intelligent combination of all models

### Advanced Text Processing
- Emoji analysis and normalization
- Slang and informal language handling
- Behavioral pattern detection
- Multi-language support

### Real-time Monitoring
- FastAPI-based REST API
- Real-time text analysis
- Batch processing capabilities
- Scalable deployment options

## 🔧 Configuration

Edit `deployment/config.yaml` to customize:
- Model parameters
- API settings
- Data processing options
- Deployment configurations

## 📈 Performance

The ACE framework achieves high accuracy in detecting:
- Direct harassment and abuse
- Subtle cyberbullying patterns
- Emotional manipulation
- Context-dependent threats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is designed to assist in identifying potential harassment and abuse. It should be used as part of a comprehensive moderation strategy and not as the sole decision-making mechanism.

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.
