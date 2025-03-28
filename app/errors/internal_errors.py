"""
Internal exceptions that should not be returned to the client.
These include errors with status codes 404, 429, etc.
"""

from .exceptions import ProxyError
from typing import Optional, Dict, Any

class ResourceNotFoundError(ProxyError):
    """Error raised when a requested resource cannot be found."""
    
    def __init__(
        self, 
        message: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
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

class InstanceError(ProxyError):
    """Base class for instance-related errors."""
    
    def __init__(
        self,
        message: str,
        instance_name: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["instance_name"] = instance_name
        
        super().__init__(
            message=message,
            status_code=status_code,
            details=details
        )

"""
Internal exceptions that should not be returned to the client.
These include errors with status codes 404, 429, etc.
"""

from .exceptions import ProxyError
from typing import Optional, Dict, Any

class ResourceNotFoundError(ProxyError):
    """Error raised when a requested resource cannot be found."""
    
    def __init__(
        self, 
        message: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
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

class InstanceError(ProxyError):
    """Base class for instance-related errors."""
    
    def __init__(
        self,
        message: str,
        instance_name: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["instance_name"] = instance_name
        
        super().__init__(
            message=message,
            status_code=status_code,
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
        message = message or f"Instance '{instance_name}' not found"
        super().__init__(
            message=message,
            resource_type="instance",
            resource_id=instance_name,
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
