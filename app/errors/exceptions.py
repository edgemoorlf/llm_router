"""
Custom exception classes for domain-specific errors.

This module defines a hierarchy of exception types that represent
different error categories in the application domain.
"""

import time
from typing import Optional, Dict, Any


class ProxyError(Exception):
    """Base exception class for all proxy-related errors."""
    
    def __init__(
        self, 
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new ProxyError.
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = int(time.time())
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a standardized error dictionary."""
        result = {
            "status": "error",
            "message": self.message,
            "timestamp": self.timestamp
        }
        
        if self.details:
            result["details"] = self.details
            
        return result


class ResourceNotFoundError(ProxyError):
    """Error raised when a requested resource cannot be found."""
    
    def __init__(
        self, 
        message: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new ResourceNotFoundError.
        
        Args:
            message: Error message
            resource_type: Type of resource (e.g., "instance", "model")
            resource_id: Identifier of the resource
            details: Additional error details
        """
        details = details or {}
        details.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })
        
        super().__init__(
            message=message,
            status_code=404,
            details=details
        )


class ValidationError(ProxyError):
    """Error raised when request validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new ValidationError.
        
        Args:
            message: Error message
            validation_errors: Specific validation errors
            details: Additional error details
        """
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
        """
        Initialize a new ConfigurationError.
        
        Args:
            message: Error message
            config_section: Section of configuration with issues
            details: Additional error details
        """
        details = details or {}
        if config_section:
            details["config_section"] = config_section
            
        super().__init__(
            message=message,
            status_code=500,
            details=details
        )


class InstanceError(ProxyError):
    """Base class for instance-related errors."""
    
    def __init__(
        self,
        message: str,
        instance_name: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new InstanceError.
        
        Args:
            message: Error message
            instance_name: Name of the instance
            status_code: HTTP status code
            details: Additional error details
        """
        details = details or {}
        details["instance_name"] = instance_name
        
        super().__init__(
            message=message,
            status_code=status_code,
            details=details
        )


class InstanceConnectionError(InstanceError):
    """Error raised when connection to an instance fails."""
    
    def __init__(
        self,
        instance_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new InstanceConnectionError.
        
        Args:
            instance_name: Name of the instance
            message: Error message
            details: Additional error details
        """
        message = message or f"Failed to connect to instance {instance_name}"
        
        super().__init__(
            message=message,
            instance_name=instance_name,
            status_code=503,
            details=details
        )


class InstanceNotFoundError(ResourceNotFoundError):
    """Error raised when an instance cannot be found."""
    
    def __init__(
        self,
        instance_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new InstanceNotFoundError.
        
        Args:
            instance_name: Name of the instance
            message: Error message
            details: Additional error details
        """
        message = message or f"Instance '{instance_name}' not found"
        
        super().__init__(
            message=message,
            resource_type="instance",
            resource_id=instance_name,
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
        """
        Initialize a new ModelNotSupportedError.
        
        Args:
            model_name: Name of the model
            instance_name: Name of the instance
            details: Additional error details
        """
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
        """
        Initialize a new ServiceUnavailableError.
        
        Args:
            service_name: Name of the service
            reason: Reason for unavailability
            details: Additional error details
        """
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


class RateLimitError(ProxyError):
    """Error raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: Optional[str] = None,
        instance_name: Optional[str] = None,
        reset_time: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new RateLimitError.
        
        Args:
            message: Error message
            instance_name: Name of the instance
            reset_time: Time when rate limit resets
            details: Additional error details
        """
        message = message or "Rate limit exceeded"
        
        details = details or {}
        if instance_name:
            details["instance_name"] = instance_name
        if reset_time:
            details["reset_time"] = reset_time
            
        super().__init__(
            message=message,
            status_code=429,
            details=details
        ) 