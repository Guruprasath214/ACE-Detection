"""
ACE Detection Framework - API Module

This module contains the FastAPI server for real-time harassment detection.
"""

from .serve import create_app, ACEAPIServer

__all__ = [
    'create_app',
    'ACEAPIServer'
]
