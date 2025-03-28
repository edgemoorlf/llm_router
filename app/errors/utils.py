"""
Error handling utilities for common operations.

This module provides utility functions for common error handling operations,
making it easier to perform error checks consistently.
"""

import logging
import time
import functools
from typing import Optional, Dict, Any, TypeVar, List, Callable

from fastapi import HTTPException, status

from app.errors.exceptions import (
    ProxyError
)

from app.errors.internal_errors import (
    InstanceNotFoundError,
)

from app.errors.client_errors import (
    ModelNotSupportedError,
    ValidationError,
)

from app.models.instance import InstanceConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


def check_instance_exists(instance: Optional[InstanceConfig], instance_name: str) -> InstanceConfig:
    """
    Check if an instance exists and raise an appropriate error if not.
    
    Args:
        instance: The instance configuration to check
        instance_name: Name of the instance
        
    Returns:
        The instance if it exists
        
    Raises:
        InstanceNotFoundError: If the instance does not exist
    """
    if not instance:
        logger.warning(f"Instance '{instance_name}' not found")
        raise InstanceNotFoundError(instance_name=instance_name)
    
    return instance


def check_model_supported(instance: InstanceConfig, model_name: str) -> bool:
    """
    Check if a model is supported by an instance and raise an error if not.
    
    Args:
        instance: The instance configuration to check
        model_name: Name of the model
        
    Returns:
        True if the model is supported
        
    Raises:
        ModelNotSupportedError: If the model is not supported
    """
    # If no supported models list, assume it can handle any model
    if not instance.supported_models:
        return True
        
    # Check for exact match
    if model_name in instance.supported_models:
        return True
        
    # Check for case-insensitive match
    if model_name.lower() in [m.lower() for m in instance.supported_models]:
        return True
    
    logger.warning(f"Model '{model_name}' is not supported by instance '{instance.name}'")
    raise ModelNotSupportedError(
        model_name=model_name,
        instance_name=instance.name
    )


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """
    Validate that a dictionary contains all required fields.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Raises:
        ValidationError: If any required fields are missing
    """
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        error_message = f"Missing required fields: {', '.join(missing_fields)}"
        logger.warning(error_message)
        
        raise ValidationError(
            message=error_message,
            validation_errors={
                "missing_fields": missing_fields
            }
        )


def handle_router_errors(operation_name: str) -> Callable:
    """
    Decorator for handling common errors in router functions, particularly HTTP 500 errors.
    
    This decorator standardizes error handling across router functions by:
    1. Catching and properly formatting HTTP 500 errors
    2. Ensuring consistent logging
    3. Passing through HTTP exceptions (like 404s) as they are
    
    Args:
        operation_name: Name of the operation being performed (for logging)
        
    Returns:
        Decorated function with standardized error handling
    
    Example:
        @router.get("/my-endpoint")
        @handle_router_errors("retrieving data")
        async def my_endpoint():
            # Your code here - no need for try/except for 500 errors!
            result = await service.get_data()
            return result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Let FastAPI HTTP exceptions pass through as-is
                raise
            except ProxyError:
                # Let our custom proxy errors pass through (they'll be handled by our exception handlers)
                raise
            except Exception as e:
                # Log the error with useful context
                logger.error(f"Error while {operation_name}: {str(e)}", exc_info=True)
                
                # Return a standardized HTTP 500 response
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "status": "error",
                        "message": f"An error occurred while {operation_name}: {str(e)}",
                        "timestamp": int(time.time()),
                        "error_type": e.__class__.__name__
                    }
                )
        return wrapper
    return decorator 


def create_500_error(message: str, additional_details: Optional[Dict[str, Any]] = None) -> HTTPException:
    """
    Create a standardized HTTP 500 error exception.
    
    This helper function creates a properly formatted HTTPException with a 500 status code.
    Use this when you need to directly raise a 500 error without using the decorator.
    
    Args:
        message: Error message explaining what went wrong
        additional_details: Optional additional details to include in the response
        
    Returns:
        A properly formatted HTTPException
    
    Example:
        if fatal_condition:
            raise create_500_error("Something went catastrophically wrong", {"item_id": item_id})
    """
    details = {
        "status": "error",
        "message": message,
        "timestamp": int(time.time())
    }
    
    if additional_details:
        if "details" not in details:
            details["details"] = {}
        details["details"].update(additional_details)
    
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=details
    ) 