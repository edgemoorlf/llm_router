"""Generic OpenAI service for forwarding and transforming requests to generic OpenAI-compatible instances."""
import logging
from typing import Any, Dict, List
from fastapi import HTTPException, status

from app.instance.manager import instance_manager
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
        model_instances = [
            instance for instance in instance_manager.instances.values()
            if (
                # Must be generic provider type
                instance.provider_type == "generic" and
                (
                    # Instance has this model in its supported_models list
                    normalized_model in [m.lower() for m in instance.supported_models] or
                    # Instance has no model restrictions (empty supported_models means it supports all models)
                    not instance.supported_models
                )
            )
        ]
        
        logger.debug(f"Found {len(model_instances)} generic instances supporting model '{normalized_model}'")
        
        if not model_instances:
            logger.warning(f"No generic instances found supporting model '{model_name}'")
        
        # For generic providers, we keep the model field as is
        generic_payload = payload.copy()
        logger.debug(f"generic payload: {generic_payload}")
        # Estimate tokens for rate limiting and instance selection
        required_tokens = 0
        
        # For handling specific endpoints; somehow new API is sending /v1/chat/completions/v1/chat/completions
        if endpoint == "/v1/chat/completions" or endpoint == "/v1/chat/completions/v1/chat/completions":
            # Estimate tokens for chat completions
            required_tokens = estimate_chat_tokens(
                generic_payload.get("messages", []),
                generic_payload.get("functions", None),
                model_name,
                "generic"
            )
        
        # For completions endpoint
        elif endpoint == "/v1/completions":
            # Estimate tokens for standard completions
            required_tokens = estimate_completion_tokens(generic_payload)
        
        # For embeddings endpoint (rough estimate)
        elif endpoint == "/v1/embeddings":
            # Estimate tokens for embeddings
            input_text = generic_payload.get("input", "")
            if isinstance(input_text, str):
                required_tokens = len(input_text.split()) * 2  # Rough approximation
            elif isinstance(input_text, list):
                required_tokens = sum(len(text.split()) * 2 for text in input_text if isinstance(text, str))
        
        # For other endpoints, use a minimum token count for rate limiting
        else:
            required_tokens = 42  # Minimum token count for unknown endpoints
        
        # Check against global rate limiter (if enabled)
        if not generic_payload.get("stream", False) and required_tokens > 0:
            # Only apply global rate limiting if the per-instance rate limiting is not sufficient
            # This gives us a fallback mechanism while still preferring per-instance limits
            allowed, retry_after = rate_limiter.check_and_update(required_tokens)
            if not allowed:
                logger.warning(f"Global rate limit exceeded: required {required_tokens} tokens")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
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
        
        # Filter instances to only include generic provider types
        generic_instances = [
            instance for instance in instance_manager.instances.values()
            if instance.provider_type == "generic"
        ]
        
        if not generic_instances:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No generic OpenAI-compatible instances available",
            )
        
        # Select an instance based on the routing strategy and model support
        # We need to ensure we only select from generic instances
        primary_instance = None
        for instance in generic_instances:
            # Check if this instance supports the model
            if (
                # Instance has this model in its supported_models list
                model_name.lower() in [m.lower() for m in instance.supported_models] or
                # Instance has no model restrictions (empty supported_models means it supports all models)
                not instance.supported_models
            ):
                # This instance supports the model, use it as primary
                primary_instance = instance
                break
        
        if not primary_instance:
            # If no specific instance was found, use the first generic instance
            primary_instance = generic_instances[0]
            logger.warning(f"No generic instance explicitly supports model '{model_name}', using {primary_instance.name}")
        
        # For generic provider types, we keep the model field
        # No need to modify the payload
        
        # Try to send the request to an available instance with failover
        # We'll only try generic instances
        result, instance = await instance_manager.try_instances(
            endpoint, 
            model_name,  # For generic providers, we use the model name as the deployment
            payload, 
            required_tokens,
            method,
            provider_type="generic"  # Only try generic instances
        )
        
        logger.debug(f"Request completed using generic instance {instance.name}")
        
        return result
        
    async def get_instances_status(self) -> List[Dict[str, Any]]:
        """Get the status of all generic instances."""
        all_stats = instance_manager.get_instance_stats()
        # Filter to only include generic instances
        return [
            stats for stats in all_stats
            if stats["provider_type"] == "generic"
        ]

# Create a singleton instance
generic_openai_service = GenericOpenAIService()
