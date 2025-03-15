"""
Error handling module for the Azure OpenAI Proxy.

This module provides a centralized error handling system including:
- Custom exception types
- Error formatting utilities
- FastAPI exception handlers
"""

from app.errors.exceptions import *
from app.errors.handlers import * 