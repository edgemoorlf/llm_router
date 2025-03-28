"""
Client-facing exceptions that should be returned to the client.
These include errors with status codes 400, 500, 503, and 533.
"""

from .exceptions import ProxyError
from typing import Optional, Dict, Any

class ValidationError(ProxyError):
    """Error raised when request validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if validation_errors:
            details["validation_errors"] = validation_errors
            
        super().__init__(
            message=message,
            status_code=400,
            details=details
        )

class ConfigurationError(ProxyError):
    """Error raised when there's an issue with configuration."""
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if config_section:
            details["config_section"] = config_section
            
        super().__init__(
            message=message,
            status_code=500,
            details=details
        )

class ModelNotSupportedError(ProxyError):
    """Error raised when a model is not supported by an instance."""
    
    def __init__(
        self,
        model_name: str,
        instance_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"Model '{model_name}' is not supported"
        if instance_name:
            message += f" by instance '{instance_name}'"
            
        details = details or {}
        details["model_name"] = model_name
        if instance_name:
            details["instance_name"] = instance_name
            
        super().__init__(
            message=message,
            status_code=400,
            details=details
        )

class ServiceUnavailableError(ProxyError):
    """Error raised when a service is unavailable."""
    
    def __init__(
        self,
        service_name: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"Service '{service_name}' is unavailable"
        if reason:
            message += f": {reason}"
            
        details = details or {}
        details["service_name"] = service_name
        if reason:
            details["reason"] = reason
            
        super().__init__(
            message=message,
            status_code=503,
            details=details
        )

class TokenLimitError(ProxyError):
    """Error raised when request exceeds maximum allowed tokens."""
    
    def __init__(
        self,
        instance_id: str,
        max_tokens: int,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = message or f"Request exceeds maximum allowed tokens ({max_tokens})"
        
        details = details or {}
        details.update({
            "instance_id": instance_id,
            "max_tokens": max_tokens,
            "error_code": "token_limit_exceeded"
        })
            
        super().__init__(
            message=message,
            status_code=533,
            details=details
        )

class InstanceConnectionError(ProxyError):
    """Error raised when connection to an instance fails."""
    
    def __init__(
        self,
        instance_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = message or f"Failed to connect to instance {instance_name}"
        
        details = details or {}
        details["instance_name"] = instance_name
        
        super().__init__(
            message=message,
            status_code=503,
            details=details
        )
