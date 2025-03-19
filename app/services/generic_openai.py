"""Generic OpenAI service for forwarding and transforming requests to generic OpenAI-compatible instances."""
import logging
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
import httpx
import asyncio
import time

from app.instance.instance_context import instance_manager
from app.utils.rate_limiter import rate_limiter
from app.utils.token_estimator import estimate_chat_tokens, estimate_completion_tokens

logger = logging.getLogger(__name__)

class GenericOpenAIService:
    """Service for transforming and forwarding requests to generic OpenAI-compatible instances."""
    
    def __init__(self):
        """Initialize the Generic OpenAI service."""
        logger.info("Initialized Generic OpenAI service for non-Azure instances")
    
    async def transform_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform an OpenAI API request for generic OpenAI-compatible services.
        
        Args:
            endpoint: The API endpoint (e.g., '/v1/chat/completions')
            payload: The request payload
            
        Returns:
            Transformed payload for generic OpenAI-compatible services
        """
        # Get the model name from the payload
        model_name = payload.get("model")
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model name is required",
            )
        
        # Normalize the model name
        normalized_model = model_name.lower().split(':')[0]
        
        # Find instances that support this model and are generic provider type
        configs = instance_manager.get_all_configs()
        states = instance_manager.get_all_states()
        model_instances = []
        
        for name, config in configs.items():
            # Check if this is a generic provider and supports the model
            if config.provider_type == "generic" and (
                normalized_model in [m.lower() for m in config.supported_models] or not config.supported_models
            ):
                state = states.get(name)
                if state and state.status == "healthy":
                    model_instances.append((name, config))
                
        logger.debug(f"Found {len(model_instances)} generic instances supporting model '{normalized_model}'")
        
        if not model_instances:
            logger.warning(f"No generic instances found supporting model '{model_name}'")
        
        # For generic providers, we keep the model field as is
        generic_payload = payload.copy()
        logger.debug(f"generic payload: {generic_payload}")
        
        # Estimate tokens for rate limiting and instance selection
        required_tokens = self._estimate_tokens(endpoint, generic_payload, model_name)
        
        # Check against global rate limiter (if enabled)
        if not generic_payload.get("stream", False) and required_tokens > 0:
            # Only apply global rate limiting if the per-instance rate limiting is not sufficient
            # This gives us a fallback mechanism while still preferring per-instance limits
            allowed, retry_after = rate_limiter.check_and_update(required_tokens, instance_id=None)
            if not allowed:
                logger.warning(f"Global rate limit exceeded: required {required_tokens} tokens")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Global rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )
        
        # Log the transformation
        logger.debug(f"Prepared request for generic provider with model '{model_name}' (est. {required_tokens} tokens)")
        
        return {
            "deployment": model_name,  # For generic providers, the deployment is the model name
            "payload": generic_payload,
            "required_tokens": required_tokens
        }
    
    def _estimate_tokens(self, endpoint: str, payload: Dict[str, Any], model_name: str) -> int:
        """Estimate tokens for the request based on endpoint type."""
        # For handling specific endpoints
        if endpoint == "/v1/chat/completions" or endpoint == "/v1/chat/completions/v1/chat/completions":
            # Estimate tokens for chat completions
            return estimate_chat_tokens(
                payload.get("messages", []),
                payload.get("functions", None),
                model_name,
                "generic"
            )
        
        # For completions endpoint
        elif endpoint == "/v1/completions":
            # Estimate tokens for standard completions
            return estimate_completion_tokens(payload)
        
        # For embeddings endpoint (rough estimate)
        elif endpoint == "/v1/embeddings":
            # Estimate tokens for embeddings
            input_text = payload.get("input", "")
            if isinstance(input_text, str):
                return len(input_text.split()) * 2  # Rough approximation
            elif isinstance(input_text, list):
                return sum(len(text.split()) * 2 for text in input_text if isinstance(text, str))
        
        # For other endpoints, use a minimum token count for rate limiting
        return 42  # Minimum token count for unknown endpoints
    
    async def forward_request(
        self, endpoint: str, model_name: str, payload: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        """
        Forward a request to an available generic OpenAI-compatible instance with automatic failover.
        
        Args:
            endpoint: The API endpoint
            model_name: The model name (used for instance selection)
            payload: The request payload
            method: The HTTP method
            
        Returns:
            The API response
        """
        # Get the estimated token requirement from the payload
        required_tokens = payload.pop("required_tokens", 1000)  # Default if not available
        
        # Select the best instance for this request
        instance_name = instance_manager.select_instance_for_request(
            model=model_name,
            tokens=required_tokens,
            provider_type="generic"
        )
        
        if not instance_name:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"No generic instances available that support model '{model_name}'",
            )
            
        # Get instance config and state
        config = instance_manager.get_instance_config(instance_name)
        state = instance_manager.get_instance_state(instance_name)
        
        if not config or not state:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to get configuration or state for instance '{instance_name}'",
            )
            
        # Check rate limit
        allowed, retry_after = instance_manager.check_rate_limit(instance_name, required_tokens)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for instance '{instance_name}'. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )
            
        try:
            # Forward the request
            result = await self._forward_request_to_instance(config, endpoint, payload, method)
            
            # If successful, update the instance stats
            instance_manager.mark_healthy(instance_name)
            
            # Record the request with token usage
            if "usage" in result and "total_tokens" in result["usage"]:
                instance_manager.record_request(
                    name=instance_name,
                    success=True,
                    tokens=result["usage"]["total_tokens"]
                )
            
            logger.debug(f"Request completed using generic instance {instance_name}")
            return result
                
        except Exception as e:
            # Handle errors
            error_message = str(e)
            logger.warning(f"Failed to forward request to generic instance {instance_name}: {error_message}")
            
            # Record the error
            if isinstance(e, HTTPException):
                status_code = e.status_code
                if status_code == 429:
                    # Rate limited
                    instance_manager.mark_rate_limited(instance_name, int(e.headers.get("retry-after", "60")))
                else:
                    instance_manager.record_request(
                        name=instance_name,
                        success=False,
                        error=error_message,
                        status_code=status_code
                    )
            else:
                # Other error
                instance_manager.record_request(
                    name=instance_name,
                    success=False,
                    error=error_message,
                    status_code=500
                )
                instance_manager.update_instance_state(
                    instance_name,
                    status="error",
                    last_error=error_message
                )
            
            raise
        
    async def _forward_request_to_instance(
        self, config: Any, endpoint: str, payload: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        """
        Forward a request directly to an instance.
        
        Args:
            config: The instance configuration to use
            endpoint: The API endpoint
            payload: The request payload
            method: HTTP method
            
        Returns:
            The API response
            
        Raises:
            HTTPException: If the request fails
        """
        # Build the URL
        url = f"{config.api_base.rstrip('/')}{endpoint}"
        
        # Create a new client for this request
        async with httpx.AsyncClient(timeout=httpx.Timeout(config.timeout_seconds)) as client:
            # Set up headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}"
            }
            
            # Make the request with retries
            for attempt in range(config.retry_count + 1):
                try:
                    response = await client.request(
                        method=method,
                        url=url,
                        json=payload,
                        headers=headers
                    )
                    
                    # Check for HTTP errors
                    response.raise_for_status()
                    
                    # Return the JSON response
                    return response.json()
                    
                except httpx.HTTPStatusError as e:
                    # Don't retry on certain status codes
                    if e.response.status_code in [400, 401, 403, 404, 429]:
                        raise HTTPException(
                            status_code=e.response.status_code,
                            detail=str(e),
                            headers=dict(e.response.headers)
                        )
                        
                    # On last attempt, raise the error
                    if attempt == config.retry_count:
                        raise HTTPException(
                            status_code=e.response.status_code,
                            detail=str(e)
                        )
                        
                    # Otherwise wait and retry
                    await asyncio.sleep(2 ** attempt)
                    
                except Exception as e:
                    # On last attempt, raise the error
                    if attempt == config.retry_count:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Request failed: {str(e)}"
                        )
                        
                    # Otherwise wait and retry
                    await asyncio.sleep(2 ** attempt)
                    
        # This should never be reached due to the raises above
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error in request forwarding"
        )

# Create a singleton instance
generic_openai_service = GenericOpenAIService()
