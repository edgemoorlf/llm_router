"""Generic OpenAI service for forwarding and transforming requests to generic OpenAI-compatible instances."""
import logging
from typing import Any, Dict, List
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
        all_instances = instance_manager.get_all_instances().values()
        model_instances = []
        
        for instance in all_instances:
            provider_type = instance.get("provider_type", "")
            supported_models = instance.get("supported_models", [])
            
            # Check if this is a generic provider and supports the model
            if provider_type == "generic" and (
                normalized_model in [m.lower() for m in supported_models] or not supported_models
            ):
                model_instances.append(instance)
                
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
            instance for instance in instance_manager.get_all_instances().values()
            if instance.get("provider_type", "") == "generic"
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
            supported_models = instance.get("supported_models", [])
            if (
                # Instance has this model in its supported_models list
                model_name.lower() in [m.lower() for m in supported_models] or
                # Instance has no model restrictions (empty supported_models means it supports all models)
                not supported_models
            ):
                # This instance supports the model, use it as primary
                primary_instance = instance
                break
        
        if not primary_instance:
            # If no specific instance was found, use the first generic instance
            primary_instance = generic_instances[0]
            logger.warning(f"No generic instance explicitly supports model '{model_name}', using {primary_instance.get('name', '')}")
        
        # Try to forward the request to each instance until one succeeds
        errors = []
        for instance in generic_instances:
            try:
                # Try to forward the request to this instance
                instance_name = instance.get("name", "")
                logger.debug(f"Trying to forward request to generic instance: {instance_name}")
                
                # Forward the request
                result = await self._forward_request_to_instance(instance, endpoint, payload, method)
                
                # If successful, update the instance stats and return the result
                instance_manager.update_instance_state(instance_name, status="healthy")
                
                # If we have usage information, update the TPM
                if "usage" in result and "total_tokens" in result["usage"]:
                    current_stats = instance.get("instance_stats", {})
                    current_tpm = current_stats.get("current_tpm", 0)
                    new_tpm = current_tpm + result["usage"]["total_tokens"]
                    instance_manager.update_instance_state(
                        instance_name,
                        instance_stats={"current_tpm": new_tpm}
                    )
                
                logger.debug(f"Request completed using generic instance {instance_name}")
                return result
                
            except Exception as e:
                # Log the error and try the next instance
                error_message = str(e)
                logger.warning(f"Failed to forward request to generic instance {instance.get('name', '')}: {error_message}")
                errors.append(f"{instance.get('name', '')}: {error_message}")
                
                # Update the instance state
                instance_name = instance.get("name", "")
                if isinstance(e, HTTPException) and getattr(e, "status_code", 0) == 429:
                    # Rate limited
                    instance_manager.update_instance_state(instance_name, status="rate_limited")
                else:
                    # Other error
                    instance_manager.update_instance_state(
                        instance_name,
                        status="error",
                        last_error=error_message
                    )
                
                # Continue to the next instance
                continue
                
        # If we're here, all instances failed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"All generic instances failed. Errors: {', '.join(errors)}"
        )
        
    async def _forward_request_to_instance(
        self, instance: Dict[str, Any], endpoint: str, payload: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        """
        Forward a request directly to an instance.
        
        Args:
            instance: The instance dictionary to use
            endpoint: The API endpoint
            payload: The request payload
            method: HTTP method
            
        Returns:
            The API response
            
        Raises:
            HTTPException: If the request fails
        """
        # Get instance properties
        instance_name = instance.get("name", "")
        provider_type = instance.get("provider_type", "generic")
        api_base = instance.get("api_base", "")
        api_key = instance.get("api_key", "")
        timeout_seconds = instance.get("timeout_seconds", 60.0)
        retry_count = instance.get("retry_count", 3)
        
        # Generate a request ID for tracking
        request_id = f"{instance_name}-{time.time()}"
        
        # Build the URL (for generic instances, we use the endpoint directly)
        url = f"{api_base}{endpoint}"
            
        logger.debug(f"[{request_id}] Forwarding request to generic instance {instance_name}: {url}")
        
        # Set up HTTP client with appropriate timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            # Set headers for generic OpenAI-compatible APIs
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
                
            # Make the request with retries
            response = None
            last_error = None
            
            for retry in range(retry_count + 1):  # +1 for the initial attempt
                try:
                    if method == "GET":
                        response = await client.get(url, headers=headers)
                    elif method == "POST":
                        response = await client.post(url, json=payload, headers=headers)
                    elif method == "PUT":
                        response = await client.put(url, json=payload, headers=headers)
                    elif method == "DELETE":
                        response = await client.delete(url, headers=headers)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                        
                    # Check for HTTP errors
                    response.raise_for_status()
                    
                    # Return the JSON response
                    return response.json()
                        
                except httpx.HTTPStatusError as e:
                    last_error = e
                    
                    # Don't retry on certain status codes
                    if e.response.status_code in [400, 401, 403, 404]:
                        break
                        
                    # For other errors, retry with backoff
                    if retry < retry_count:
                        backoff_seconds = (2 ** retry) * 0.5  # Exponential backoff
                        logger.warning(f"[{request_id}] Request failed with HTTP {e.response.status_code}, retrying in {backoff_seconds:.2f}s ({retry+1}/{retry_count})")
                        await asyncio.sleep(backoff_seconds)
                    
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                    last_error = e
                    
                    # Retry connection errors
                    if retry < retry_count:
                        backoff_seconds = (2 ** retry) * 0.5  # Exponential backoff
                        logger.warning(f"[{request_id}] Connection error: {str(e)}, retrying in {backoff_seconds:.2f}s ({retry+1}/{retry_count})")
                        await asyncio.sleep(backoff_seconds)
                
                except Exception as e:
                    # Don't retry other exceptions
                    logger.error(f"[{request_id}] Unexpected error: {str(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Unexpected error forwarding request: {str(e)}"
                    )
            
            # If we're here, all retries failed
            if isinstance(last_error, httpx.HTTPStatusError):
                # Try to parse error response
                try:
                    error_json = last_error.response.json()
                    error_message = error_json.get("error", {}).get("message", str(last_error))
                except Exception:
                    error_message = last_error.response.text or str(last_error)
                    
                status_code = last_error.response.status_code
                
                # Create headers dict if it doesn't exist
                headers = {}
                if hasattr(last_error.response, "headers"):
                    for key, value in last_error.response.headers.items():
                        headers[key.lower()] = value
                
                raise HTTPException(
                    status_code=status_code,
                    detail=error_message,
                    headers=headers
                )
            else:
                # For connection errors
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Service unavailable: {str(last_error)}"
                )
    
    async def get_instances_status(self) -> List[Dict[str, Any]]:
        """Get the status of all generic instances."""
        all_stats = instance_manager.get_instance_stats()
        # Filter to only include generic instances
        return [
            stats for stats in all_stats
            if stats.get("provider_type", "") == "generic"
        ]

# Create a singleton instance
generic_openai_service = GenericOpenAIService()
