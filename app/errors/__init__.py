"""
Error handling module for the Azure OpenAI Proxy.

This module provides a centralized error handling system including:
- Custom exception types
- Error formatting utilities
- FastAPI exception handlers
"""

from .exceptions import ProxyError
from .client_errors import (
    ValidationError,
    ConfigurationError,
    ModelNotSupportedError,
    ServiceUnavailableError,
    TokenLimitError,
    InstanceConnectionError
)
from .internal_errors import (
    ResourceNotFoundError,
    InstanceError,
    InstanceNotFoundError,
    RateLimitError
)
from .handlers import *
"""
Error handling module for the Azure OpenAI Proxy.

This module provides a centralized error handling system including:
- Custom exception types
- Error formatting utilities
- FastAPI exception handlers
"""

from app.errors.exceptions import *
from app.errors.handlers import *
