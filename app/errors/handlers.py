"""
FastAPI exception handlers and utilities for error handling.

This module defines FastAPI exception handlers that convert exceptions
to standardized HTTP responses.
"""

import logging
import time
import traceback
from typing import Dict, Any, Callable, Type, Union, Optional

from fastapi import Request, FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors.exceptions import ProxyError

logger = logging.getLogger(__name__)


def format_error_response(
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format a standardized error response.
    
    Args:
        message: Error message
        details: Additional error details
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "status": "error",
        "message": message,
        "timestamp": int(time.time())
    }
    
    if details:
        response["details"] = details
        
    return response


async def proxy_error_handler(request: Request, exc: ProxyError) -> JSONResponse:
    """
    Handle ProxyError exceptions.
    
    Args:
        request: FastAPI request
        exc: ProxyError instance
        
    Returns:
        JSONResponse with standardized error format
    """
    logger.error(f"ProxyError: {exc.message}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle FastAPI RequestValidationError.
    
    Args:
        request: FastAPI request
        exc: RequestValidationError instance
        
    Returns:
        JSONResponse with standardized error format
    """
    errors = exc.errors()
    error_details = []
    
    for error in errors:
        error_detail = {
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        }
        error_details.append(error_detail)
    
    logger.warning(f"Validation error: {error_details}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=format_error_response(
            message="Request validation error",
            details={"errors": error_details}
        )
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle Starlette HTTPException.
    
    Args:
        request: FastAPI request
        exc: StarletteHTTPException instance
        
    Returns:
        JSONResponse with standardized error format
    """
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    # Handle custom detail format
    if isinstance(exc.detail, dict) and "status" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_response(
            message=str(exc.detail)
        )
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle any unhandled exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception instance
        
    Returns:
        JSONResponse with standardized error format
    """
    # Get the full exception traceback
    tb = traceback.format_exc()
    
    # Log the full traceback at ERROR level
    logger.error(f"Unhandled exception: {str(exc)}\n{tb}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=format_error_response(
            message="Internal server error",
            details={
                "type": exc.__class__.__name__,
                "detail": str(exc)
            }
        )
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers with a FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Register handler for ProxyError (and subclasses)
    app.add_exception_handler(ProxyError, proxy_error_handler)
    
    # Register handler for FastAPI's RequestValidationError
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    
    # Register handler for Starlette's HTTPException
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Register handler for generic exceptions (fallback)
    app.add_exception_handler(Exception, generic_exception_handler)


def handle_errors(func: Callable) -> Callable:
    """
    Decorator for handling errors in service methods.
    
    This decorator catches exceptions and either:
    1. Passes through ProxyError exceptions
    2. Wraps other exceptions in appropriate ProxyError subclasses
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that handles errors
    """
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ProxyError:
            # ProxyError exceptions are already properly formatted, re-raise
            raise
        except Exception as e:
            # Log the unexpected error
            logger.error(f"Error in {func.__name__}: {str(e)}")
            
            # Wrap in a generic ProxyError
            raise ProxyError(
                message=f"An error occurred: {str(e)}",
                status_code=500,
                details={"function": func.__name__}
            )
            
    return wrapper 