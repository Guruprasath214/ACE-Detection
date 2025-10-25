"""
ACE Detection Framework - Training Module

This module contains training scripts and evaluation utilities for the ACE framework.
"""

from .train_pipeline import ACETrainer
from .evaluation import ModelEvaluator

__all__ = [
    'ACETrainer',
    'ModelEvaluator'
]
