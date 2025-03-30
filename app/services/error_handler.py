"""Service for centralized error handling across different LLM providers."""
import logging
import time
from typing import Dict, Any, Tuple, Optional
from fastapi import HTTPException, status

from app.instance.instance_context import instance_manager

logger = logging.getLogger(__name__)

# Common error messages
CONTENT_POLICY_VIOLATION_DETAIL = "The prompt has triggered Content management policy. Please modify the prompt and try again."

class ErrorHandler:
    """Service for handling errors from LLM providers in a consistent way."""
    
    def __init__(self):
        """Initialize the error handler service."""
        logger.info("Initialized Error Handler service")
    
    def handle_special_error(self, error: Exception, instance_name: str) -> bool:
        """
        Handle special errors that require immediate propagation.
        
        Args:
            error: The exception that occurred
            instance_name: Name of the instance where the error occurred
            
        Returns:
            True if a special error was handled, False otherwise
        """
        if isinstance(error, HTTPException):
            if error.status_code == 400:
                logger.warning(f"Bad request (HTTP {error.status_code}) in instance {instance_name}")
                if "content management policy" in error.detail.lower():
                    logger.warning(f"Content policy violation (HTTP {error.status_code}) in instance {instance_name}")
                    raise HTTPException(
                        status_code=error.status_code,
                        detail=CONTENT_POLICY_VIOLATION_DETAIL,
                        headers=error.headers if hasattr(error, "headers") else {}
                    )

            # doing nothing otherwise, hand it over the normal error handler
                
        return False  # Not a special error
    
    def handle_instance_error(self, instance: Dict[str, Any], error: Exception) -> None:
        """
        Centralized handler for instance errors.
        
        Args:
            instance: The instance that encountered an error
            error: The exception that occurred
        """
        instance_name = instance.get("name", "")
        
        if isinstance(error, HTTPException):
            if error.status_code == 429:
                retry_after = None
                if hasattr(error, 'headers') and error.headers and 'retry-after' in error.headers:
                    try:
                        retry_after = int(error.headers['retry-after'])
                    except (ValueError, TypeError):
                        pass
                        
                instance_manager.update_instance_state(
                    instance_name,
                    status="rate_limited",
                    rate_limited_until=time.time() + (retry_after or 60)
                )
            elif error.status_code in [401, 404]:
                # Only under 401,404 we should mark the instance as error
                instance_manager.update_instance_state(
                    instance_name,
                    status="error",
                    last_error=str(error),
                    error_count=instance.get("error_count", 0) + 1
                )
                
        # Otherwise, it's not instance's error, we should not update instance state
    
    def parse_http_error(self, error: Exception) -> Tuple[int, str, Dict[str, str]]:
        """
        Parse HTTP errors to extract status code, message, and headers.
        
        Args:
            error: The exception to parse
            
        Returns:
            Tuple of (status_code, error_message, headers)
        """
        if isinstance(error, HTTPException):
            status_code = error.status_code
            error_message = error.detail
            headers = error.headers if hasattr(error, "headers") else {}
            return status_code, error_message, headers
            
        # For non-HTTP exceptions, use generic server error
        return status.HTTP_500_INTERNAL_SERVER_ERROR, str(error), {}
    
    def update_instance_status(self, instance_name: str, status: str, **kwargs) -> None:
        """
        Centralized method to update instance status.
        
        Args:
            instance_name: The name of the instance to update
            status: The new status value
            **kwargs: Additional status attributes to update
        """
        instance_manager.update_instance_state(instance_name, status=status, **kwargs)

# Create a singleton instance
error_handler = ErrorHandler() 