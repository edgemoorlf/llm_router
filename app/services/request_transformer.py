"""Service for transforming API requests between different LLM providers."""
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, status

from app.utils.rate_limiter import rate_limiter
from app.utils.token_estimator import estimate_chat_tokens, estimate_completion_tokens

logger = logging.getLogger(__name__)

# Default token limits
DEFAULT_TOKEN_RATE_LIMIT = 30000
DEFAULT_MAX_INPUT_TOKENS_LIMIT = 16384

class RequestTransformer:
    """Service for transforming API requests between different LLM provider formats."""
    
    def __init__(self):
        """Initialize the request transformer service."""
        logger.info("Initialized Request Transformer service")
    
    def transform_openai_to_azure(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform an OpenAI API request to Azure OpenAI format.
        
        Args:
            endpoint: The API endpoint (e.g., '/v1/chat/completions')
            payload: The request payload
            
        Returns:
            Transformed payload for Azure OpenAI
        """
        # Get the model name from the payload
        model_name = payload.get("model")
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model name is required",
            )
            
        # Use exact model name matching only
        exact_model_name = model_name.lower()
        
        # Clone the payload and remove the model field since Azure uses deployment names
        azure_payload = payload.copy()
        azure_payload.pop("model", None)
        
        # Estimate tokens for rate limiting and instance selection
        required_tokens = self.estimate_tokens(endpoint, azure_payload, model_name)
        
        # in Azure OpenAI, at least for S0 level, max_tokens is counted towards the TPM limit
        # per 0318-maxtoken strategy, we remove max_tokens from the payload
        max_tokens = azure_payload.get("max_tokens", None)
        if max_tokens and max_tokens > 5000 and max_tokens > required_tokens + 5000 and '2024-05-13' not in model_name:
            azure_payload["max_tokens"] = required_tokens + 5000
        
        # Check against global rate limiter (if enabled)
        if not azure_payload.get("stream", False) and required_tokens > 0:
            # Only apply global rate limiting if the per-instance rate limiting is not sufficient
            # This gives us a fallback mechanism while still preferring per-instance limits
            allowed, retry_after = rate_limiter.check_capacity(required_tokens)
            if not allowed:
                logger.warning(f"Global rate limit exceeded: required {required_tokens} tokens")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Global rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )
        
        # Add tokens to payload for later use
        azure_payload["required_tokens"] = required_tokens
        logger.debug(f"Transformed request for model '{model_name}' (est. {required_tokens} tokens) input tokens: {required_tokens} max tokens: {azure_payload.get('max_tokens', None)}")
        
        return {
            "original_model": exact_model_name,  # Preserve the original model name for instance selection
            "payload": azure_payload
        }
    
    def estimate_tokens(self, endpoint: str, payload: Dict[str, Any], model_name: str) -> int:
        """
        Estimate tokens for the payload based on endpoint type.
        
        Args:
            endpoint: The API endpoint path
            payload: The request payload
            model_name: The model name
            
        Returns:
            Estimated token count
        """
        # For handling specific endpoints
        if endpoint == "/v1/chat/completions":
            # Estimate tokens for chat completions
            return estimate_chat_tokens(
                payload.get("messages", []),
                payload.get("functions", None),
                model_name,
                "azure"
            )

        # For completions endpoint
        elif endpoint == "/v1/completions":
            # Estimate tokens for standard completions
            prompt = payload.get("prompt", "")
            return estimate_completion_tokens(prompt, model_name, "azure")
        
        # For embeddings endpoint (rough estimate)
        elif endpoint == "/v1/embeddings":
            # Estimate tokens for embeddings
            input_text = payload.get("input", "")
            if isinstance(input_text, str):
                return len(input_text.split()) * 2  # Rough approximation
            elif isinstance(input_text, list):
                return sum(len(text.split()) * 2 for text in input_text if isinstance(text, str))
        
        # For other endpoints, use a minimum token count for rate limiting
        else:
            return 100  # Minimum token count for unknown endpoints

# Create a singleton instance
request_transformer = RequestTransformer() 