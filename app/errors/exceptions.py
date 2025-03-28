"""
Base exception class for all proxy-related errors.
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
