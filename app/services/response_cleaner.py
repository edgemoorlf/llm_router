"""Service for cleaning and normalizing responses from different LLM providers."""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ResponseCleaner:
    """Service for cleaning and normalizing responses from different LLM providers."""
    
    def __init__(self):
        """Initialize the response cleaner service."""
        logger.info("Initialized Response Cleaner service")
    
    def clean_azure_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove Azure-specific fields from the response to match OpenAI format.
        
        Args:
            response: The raw Azure OpenAI API response
            
        Returns:
            Cleaned response compatible with standard OpenAI format
        """
        # Make a copy to avoid modifying the original
        cleaned = response.copy()
        
        # Remove top-level Azure-specific fields
        cleaned.pop("prompt_filter_results", None)
        cleaned.pop("content_filter_results", None)
        
        if "choices" in cleaned:
            for choice in cleaned["choices"]:
                # Remove choice-level Azure fields
                choice.pop("content_filter_results", None)
                
                # Remove refusal field from message
                if "message" in choice:
                    choice["message"].pop("refusal", None)
        
        return cleaned
    
    def normalize_stream_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize streaming response formats between providers.
        
        Args:
            response: A chunk from a streaming response
            
        Returns:
            Normalized chunk in consistent format
        """
        # For now, this just delegates to the Azure cleaner
        # In the future, we could add more normalization logic for different providers
        return self.clean_azure_response(response)
    
    def normalize_error_response(self, status_code: int, detail: str) -> Dict[str, Any]:
        """
        Create a normalized error response matching OpenAI's format.
        
        Args:
            status_code: HTTP status code
            detail: Error message detail
            
        Returns:
            Normalized error response in OpenAI format
        """
        error_type = "server_error"
        if status_code == 400:
            error_type = "invalid_request_error"
        elif status_code == 401:
            error_type = "authentication_error"
        elif status_code == 403:
            error_type = "permission_error"
        elif status_code == 404:
            error_type = "not_found_error"
        elif status_code == 429:
            error_type = "rate_limit_error"
            
        return {
            "error": {
                "message": detail,
                "type": error_type,
                "code": status_code
            }
        }

# Create a singleton instance
response_cleaner = ResponseCleaner()
