"""
ACE FastAPI Server

This module implements a FastAPI server for real-time harassment detection
using the ACE framework.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
import asyncio
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime
import time

# Import ACE components
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.ensemble import ACEEnsemble
    from utils.preprocessing import TextPreprocessor
except ImportError as e:
    print(f"Error: Could not import required components: {e}")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class TextInput(BaseModel):
    """Input model for single text prediction."""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to analyze")
    include_explanation: bool = Field(False, description="Include detailed explanation")
    include_preprocessing: bool = Field(False, description="Include preprocessing analysis")


class BatchTextInput(BaseModel):
    """Input model for batch text prediction."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    include_explanation: bool = Field(False, description="Include detailed explanation")
    include_preprocessing: bool = Field(False, description="Include preprocessing analysis")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    text: str
    prediction: int = Field(..., description="0: Normal, 1: Harassment")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    risk_score: float = Field(..., ge=0, description="Overall risk score")
    base_scores: Dict[str, float] = Field(..., description="Scores from base models")
    explanation: Optional[str] = Field(None, description="Explanation of prediction")
    preprocessing_analysis: Optional[Dict[str, Any]] = Field(None, description="Preprocessing analysis")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_count: int
    harassment_count: int
    normal_count: int
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str


class ACEAPIServer:
    """ACE API Server class."""
    
    def __init__(self, config_path: str = str(Path(__file__).parent.parent / "deployment" / "config.yaml")):
        """
        Initialize ACE API server.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.ace_ensemble = None
        self.text_preprocessor = None
        self.model_loaded = False
        self.start_time = datetime.now()
        
        # Initialize components
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'max_request_size': 10485760,
                'timeout': 30
            },
            'models': {
                'bert': {
                    'model_name': 'bert-base-uncased'
                },
                'cnn': {
                    'model_name': 'cnn'
                },
                'emotion': {
                    'model_name': 'j-hartmann/emotion-english-distilroberta-base'
                },
                'deep_emoji': {
                    'model_name': 'cardiffnlp/twitter-roberta-base-emoji',
                    'max_length': 128
                },
                'ensemble': {
                    'weights': {
                        'bert': 0.2,
                        'cnn': 0.2,
                        'svm': 0.2,
                        'deep_emoji': 0.2,
                        'emotion': 0.2,
                        'contextual': 0.0
                    },
                    'decision_threshold': 0.6,
                    'hidden_dim': 64,
                    'dropout_rate': 0.3
                }
            },
            'security': {
                'enable_cors': True,
                'allowed_origins': ['*'],
                'rate_limit': 100
            }
        }
    
    def _initialize_components(self):
        """Initialize ACE components."""
        try:
            # Initialize text preprocessor
            self.text_preprocessor = TextPreprocessor(self.config.get('data', {}))
        except Exception as e:
            logger.warning(f"Could not initialize text preprocessor: {e}. Using default.")
            self.text_preprocessor = None

        try:
            # Initialize ACE Ensemble with 6 ML models
            self.ace_ensemble = ACEEnsemble(self.config)
            self.model_loaded = True
            logger.info("ACE Ensemble (6 ML models) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ACE Ensemble: {e}")
            raise
    
    async def predict_single(self, text_input: TextInput) -> PredictionResponse:
        """
        Make prediction on a single text.
        
        Args:
            text_input: Input text and options
            
        Returns:
            Prediction response
        """
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Make prediction
            if text_input.include_explanation:
                result = self.ace_ensemble.explain_prediction(text_input.text)
                explanation = result.get('explanation', '')
                preprocessing_analysis = result.get('preprocessing_analysis', {})
            else:
                result = self.ace_ensemble.predict_single(text_input.text)
                explanation = None
                preprocessing_analysis = None
            
            # Get preprocessing analysis if requested
            if text_input.include_preprocessing and not preprocessing_analysis:
                preprocessing_analysis = self.text_preprocessor.preprocess_text(text_input.text)
            
            # Create response
            risk_score = result.get('weighted_score', result['confidence'])
            response = PredictionResponse(
                text=text_input.text,
                prediction=result['prediction'],
                confidence=result['confidence'],
                risk_score=min(risk_score, 1.0),  # Ensure risk_score is capped at 1.0
                base_scores=result['base_scores'],
                explanation=explanation,
                preprocessing_analysis=preprocessing_analysis,
                timestamp=datetime.now().isoformat()
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Single prediction completed in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in single prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    async def predict_batch(self, batch_input: BatchTextInput) -> BatchPredictionResponse:
        """
        Make predictions on multiple texts.
        
        Args:
            batch_input: Batch input texts and options
            
        Returns:
            Batch prediction response
        """
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            predictions = []
            
            for text in batch_input.texts:
                # Create text input for single prediction
                text_input = TextInput(
                    text=text,
                    include_explanation=batch_input.include_explanation,
                    include_preprocessing=batch_input.include_preprocessing
                )
                
                # Make prediction
                prediction = await self.predict_single(text_input)
                predictions.append(prediction)
            
            # Calculate statistics
            harassment_count = sum(1 for p in predictions if p.prediction == 1)
            normal_count = len(predictions) - harassment_count
            processing_time = time.time() - start_time
            
            response = BatchPredictionResponse(
                predictions=predictions,
                total_count=len(predictions),
                harassment_count=harassment_count,
                normal_count=normal_count,
                processing_time=processing_time
            )
            
            logger.info(f"Batch prediction completed in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
    
    async def health_check(self) -> HealthResponse:
        """
        Health check endpoint.
        
        Returns:
            Health status response
        """
        return HealthResponse(
            status="healthy" if self.model_loaded else "unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=self.model_loaded,
            version="1.0.0"
        )
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Model information
        """
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {
            "model_name": "ACE Ensemble (6 ML Models)",
            "version": "1.0.0",
            "components": ["BERT", "CNN", "SVM", "Deep Emoji", "Emotion", "Contextual RoBERTa"],
            "description": "Machine learning ensemble with 6 specialized models for advanced harassment detection",
            "uptime": str(datetime.now() - self.start_time)
        }


# Global server instance - will be initialized in main or lazily
ace_server = None


def create_app() -> FastAPI:
    """Create FastAPI application."""
    global ace_server
    if ace_server is None:
        ace_server = ACEAPIServer()

    app = FastAPI(
        title="ACE Detection API",
        description="Advanced Cyberbullying & Emotion Detection API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    if ace_server.config['security']['enable_cors']:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=ace_server.config['security']['allowed_origins'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
        return response

    # Routes
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_single(text_input: TextInput):
        """Single text prediction endpoint."""
        return await ace_server.predict_single(text_input)

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(batch_input: BatchTextInput):
        """Batch text prediction endpoint."""
        return await ace_server.predict_batch(batch_input)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return await ace_server.health_check()

    @app.get("/model/info")
    async def get_model_info():
        """Get model information."""
        return await ace_server.get_model_info()

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "ACE Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }

    @app.get("/favicon.ico")
    async def favicon():
        """Serve favicon.ico to prevent 404 errors."""
        return Response(content="", media_type="image/x-icon")

    return app


# Create FastAPI app
app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
    """
    uvicorn.run(
        "api.serve:app",
        host=host,
        port=port,
        workers=workers,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ACE Detection API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--config", default="deployment/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize server with config
    ace_server = ACEAPIServer(args.config)
    
    # Run server
    run_server(args.host, args.port, args.workers)
