"""Azure OpenAI service for forwarding and transforming requests to Azure OpenAI instances using DRY request handling."""
import logging
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
import random
import time
import httpx
import asyncio
import os

# Import from instance context
from app.instance.instance_context import instance_manager, instance_router
from app.utils.rate_limiter import rate_limiter
from app.utils.token_estimator import estimate_chat_tokens, estimate_completion_tokens
from app.utils.url_builder import build_instance_url

logger = logging.getLogger(__name__)

# Default token rate limit (tokens per minute)
DEFAULT_TOKEN_RATE_LIMIT = 30000
DEFAULT_MAX_INPUT_TOKENS_LIMIT = 16384

class AzureOpenAIService:
    """Service for Azure OpenAI requests with streamlined endpoint methods."""
    
    def __init__(self):
        """Initialize the Azure OpenAI service."""
        logger.info("Initialized Azure OpenAI service for Azure-specific instances")
    
    async def transform_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
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
        
        # Verify there are Azure instances that support this model
        azure_instances = self._get_azure_instances_for_model(exact_model_name)
        
        if not azure_instances:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No Azure instances found that support model '{model_name}'",
            )
        
        # Clone the payload and remove the model field since Azure uses deployment names
        azure_payload = payload.copy()
        azure_payload.pop("model", None)
        
        # Estimate tokens for rate limiting and instance selection
        required_tokens = self._estimate_tokens(endpoint, azure_payload, model_name)
        
        # Check against global rate limiter (if enabled)
        if not azure_payload.get("stream", False) and required_tokens > 0:
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
        
        # Add tokens to payload for later use
        azure_payload["required_tokens"] = required_tokens
        logger.debug(f"Transformed request for model '{model_name}' (est. {required_tokens} tokens)")
        
        return {
            "original_model": exact_model_name,  # Preserve the original model name for instance selection
            "payload": azure_payload
        }
    
    def _estimate_tokens(self, endpoint: str, payload: Dict[str, Any], model_name: str) -> int:
        """Estimate tokens for the payload based on endpoint type."""
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
        else:
            return 100  # Minimum token count for unknown endpoints
        
    def _clean_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Remove Azure-specific fields from the response to match OpenAI format."""
        # Remove top-level Azure-specific fields
        response.pop("prompt_filter_results", None)
        response.pop("content_filter_results", None)
        
        if "choices" in response:
            for choice in response["choices"]:
                # Remove choice-level Azure fields
                choice.pop("content_filter_results", None)
                choice.pop("logprobs", None)
                
                # Remove refusal field from message
                if "message" in choice:
                    choice["message"].pop("refusal", None)
        
        return response

    async def forward_request(
        self, endpoint: str, payload: Dict[str, Any], original_model: str, method: str = "POST"
    ) -> Dict[str, Any]:
        """Forward request to Azure instances with failover handling."""
        required_tokens = payload.pop("required_tokens", 1000)
        
        # Use the shared instance selection method
        available_instances = self.select_instances_for_model(original_model, required_tokens)
        
        if not available_instances:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail=f"No Azure instances available that support model '{original_model}'"
            )
            
        # The first instance is the primary instance based on our selection logic
        primary_instance = available_instances[0]
            
        return await self._execute_request(endpoint, original_model, payload, required_tokens, method, primary_instance)

    def _get_azure_instances(self) -> list:
        """Get all available Azure instances."""
        instances = instance_manager.get_all_instances().values()
        return [i for i in instances if i.get("provider_type", "") == "azure"]
        
    def _get_azure_instances_for_model(self, model_name: str) -> list:
        """Get Azure instances that support a specific model."""
        return [
            instance for instance in self._get_azure_instances()
            if (
                # Strict model compatibility check - exact match in supported_models only
                model_name.lower() in [m.lower() for m in instance.get("supported_models", [])]
            )
        ]

    def _select_primary_instance(self, instances: list, tokens: int, model_name: str) -> Optional[object]:
        """
        Select primary instance based on routing strategy with strict model compatibility checking.
        
        Args:
            instances: List of available Azure instances that support the model
            tokens: Required tokens for the request
            model_name: The model name to use
        """
        # Filter instances by token capacity and health
        eligible_instances = []
        for instance in instances:
            # Get instance-specific rate limiter
            instance_name = instance.get("name", "unknown")
            
            # Get current usage from instance state
            current_usage = instance.get("current_tpm", 0)
            max_tpm = instance.get("max_tpm", DEFAULT_TOKEN_RATE_LIMIT)
            
            if current_usage + tokens <= max_tpm:
                # Verify instance is healthy (not in error or rate limited state)
                status = instance.get("status", "")
                if status == "healthy":
                    eligible_instances.append(instance)
                
        # If we found eligible instances, select one based on the lowest load
        if eligible_instances:
            # Choose the instance with the lowest current usage
            selected_instance = min(eligible_instances, 
                                 key=lambda i: i.get("current_tpm", 0))
            logger.debug(f"Selected instance '{selected_instance.get('name', '')}' for model '{model_name}'")
            # Store the model used for selection to ensure consistent fallback selection
            selected_instance["_original_model_for_selection"] = model_name
            return selected_instance
            
        # If no eligible instances found with capacity, try using the instance router's selection
        instance_name = instance_router.select_instance(required_tokens=tokens, model_name=model_name)
        if instance_name:
            instance = instance_manager.get_instance(instance_name)
            if instance and instance.get("provider_type", "") == "azure" and model_name.lower() in [
                m.lower() for m in instance.get("supported_models", [])
            ]:
                logger.debug(f"Selected instance '{instance.get('name', '')}' via instance router for model '{model_name}'")
                # Store the model used for selection to ensure consistent fallback selection
                instance["_original_model_for_selection"] = model_name
            return instance
            
        # If no instances are available that meet our criteria, return None
        logger.error(f"No instances available with capacity for {tokens} tokens for model '{model_name}'")
        return None

    async def _execute_request(
        self, endpoint: str, model_name: str, payload: Dict[str, Any], required_tokens: int, method: str, instance: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute the request with failover handling."""
        # We already check for None in forward_request, but add an assertion for safety
        assert instance is not None, "Instance should not be None at this point"
        
        # Preserve the original model for consistent instance selection in fallbacks
        original_model = instance.get("_original_model_for_selection", model_name)
        
        # Ensure the model name is in the payload for deployment name resolution
        payload_with_model = payload.copy()
        payload_with_model["model"] = original_model
            
        try:
            # Use the already selected instance directly
            instance_name = instance.get("name", "")
            logger.debug(f"Executing request using Azure instance {instance_name} for model {model_name}")
            
            # Forward the request directly to the selected instance
            result = await self._forward_request_to_instance(instance, endpoint, payload_with_model, method)
            
            # Mark instance as healthy
            instance_manager.update_instance_state(instance_name, status="healthy")
            
            logger.debug(f"Request completed using Azure instance {instance.get('name', '')} for model {model_name}")
            return self._clean_response(result)
        except HTTPException as e:
            # Check for content management policy errors
            if "content management policy" in e.detail.lower():
                logger.warning(f"Content policy violation (HTTP {e.status_code}) - prompt: {payload_with_model.get('messages', '')}")
                # Ensure we're preserving the original error (which should be 400) for content policy violations
                raise HTTPException(
                    status_code=e.status_code,  # Preserve original status code (should be 400)
                    detail=e.detail,
                    headers=e.headers
                )
                
            # If there's an error with this instance, then try other instances as fallback
            logger.warning(f"Primary instance {instance.get('name', '')} failed with error {e.status_code}: {e.detail}, falling back to other instances")
            
            # Update instance status based on error
            instance_name = instance.get("name", "")
            if e.status_code == 429:
                retry_after = None
                if hasattr(e, 'headers') and e.headers and 'retry-after' in e.headers:
                    try:
                        retry_after = int(e.headers['retry-after'])
                    except (ValueError, TypeError):
                        pass
                instance_manager.update_instance_state(
                    instance_name,
                    status="rate_limited",
                    rate_limited_until=time.time() + (retry_after or 60)
                )
            else:
                instance_manager.update_instance_state(
                    instance_name,
                    status="error",
                    last_error=str(e),
                    error_count=instance.get("error_count", 0) + 1
                )
            
            # Get all eligible Azure instances that support this model (excluding the failed one)
            azure_instances = self._get_azure_instances_for_model(original_model)
            eligible_instances = []
            
            for candidate in azure_instances:
                # Don't include the failed instance
                if candidate.get('name', '') == instance.get('name', ''):
                    continue
                    
                # Verify token capacity
                current_tpm = candidate.get("instance_stats", {}).get("current_tpm", 0) 
                max_tpm = candidate.get("max_tpm", 0)
                token_capacity_sufficient = current_tpm + required_tokens <= max_tpm
                
                # Check instance is healthy (not rate limited or in error state)
                is_healthy = candidate.get("status", "") == "healthy"
                
                if token_capacity_sufficient and is_healthy:
                    eligible_instances.append(candidate)
            
            # Log the number of eligible instances
            logger.debug(f"Found {len(eligible_instances)} eligible fallback instances for model '{original_model}'")
            
            # Try all eligible instances
            if eligible_instances:
                fallback_errors = []
                
                for fallback_instance in eligible_instances:
                    try:
                        # Forward the request to the fallback instance
                        fallback_name = fallback_instance.get("name", "")
                        logger.debug(f"Trying fallback instance {fallback_name} for model {original_model}")
                        
                        result = await self._forward_request_to_instance(fallback_instance, endpoint, payload_with_model, method)
                        
                        # Mark instance as healthy and update TPM
                        fallback_name = fallback_instance.get("name", "")
                        instance_manager.update_instance_state(fallback_name, status="healthy")
                        
                        if "usage" in result and "total_tokens" in result["usage"]:
                            # Update TPM directly through the instance manager
                            current_stats = fallback_instance.get("instance_stats", {})
                            current_tpm = current_stats.get("current_tpm", 0)
                            new_tpm = current_tpm + result["usage"]["total_tokens"]
                            instance_manager.update_instance_state(
                                fallback_name,
                                instance_stats={"current_tpm": new_tpm}
                            )
                            
                        logger.debug(f"Request completed using fallback Azure instance {fallback_name}")
                        return self._clean_response(result)
                    except Exception as fallback_e:
                        error_msg = str(fallback_e)
                        logger.warning(f"Fallback instance {fallback_instance.get('name', '')} failed with error: {error_msg}")
                        fallback_errors.append(f"{fallback_instance.get('name', '')}: {error_msg}")
                        
                        # Update fallback instance status
                        fallback_name = fallback_instance.get("name", "")
                        if isinstance(fallback_e, HTTPException) and fallback_e.status_code == 429:
                            retry_after = None
                            if hasattr(fallback_e, 'headers') and fallback_e.headers and 'retry-after' in fallback_e.headers:
                                try:
                                    retry_after = int(fallback_e.headers['retry-after'])
                                except (ValueError, TypeError):
                                    pass
                            instance_manager.update_instance_state(
                                fallback_name,
                                status="rate_limited",
                                rate_limited_until=time.time() + (retry_after or 60)
                            )
                        else:
                            instance_manager.update_instance_state(
                                fallback_name,
                                status="error",
                                last_error=error_msg,
                                error_count=fallback_instance.get("error_count", 0) + 1
                            )
                
                # If we get here, all fallbacks failed
                error_detail = f"All eligible fallback instances failed. Primary error: {e.detail}. Fallback errors: {', '.join(fallback_errors)}"
                logger.error(error_detail)
                raise HTTPException(
                    status_code=e.status_code,
                    detail=error_detail,
                    headers=e.headers if hasattr(e, "headers") else {}
                )
            else:
                logger.error(f"No eligible fallback instances found with capacity for {required_tokens} tokens for model '{original_model}'")
                raise e
                
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
        provider_type = instance.get("provider_type", "azure")
        api_base = instance.get("api_base", "")
        api_key = instance.get("api_key", "")
        api_version = instance.get("api_version", "")
        model_deployments = instance.get("model_deployments", {})
        timeout_seconds = instance.get("timeout_seconds", 60.0)
        retry_count = instance.get("retry_count", 3)
        
        # Extract model from payload
        model_name = payload.get("model", "").lower() if payload and "model" in payload else ""
        
        # Determine deployment name based on the model
        deployment = ""
        if model_name and provider_type == "azure" and model_deployments:
            # Look up the deployment name for this model in this specific instance
            deployment = model_deployments.get(model_name, "")
            if deployment:
                logger.debug(f"Resolved deployment '{deployment}' for model '{model_name}' in instance '{instance_name}'")
            else:
                logger.warning(f"No deployment mapping found for model '{model_name}' in instance '{instance_name}'")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No deployment mapping found for model '{model_name}' in instance '{instance_name}'"
                )
                
        # Generate a request ID for tracking
        request_id = f"{instance_name}-{time.time()}"
        
        # Add current model to instance for URL building
        instance = instance.copy()  # Make a copy to avoid modifying the original
        instance["_current_model"] = model_name
        
        # Build the URL using shared logic
        url = build_instance_url(instance, endpoint)
        if not url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not build URL for instance '{instance.get('name', '')}'"
            )
        
        logger.debug(f"[{request_id}] Forwarding request to instance {instance_name} ({provider_type}): {url}")
        
        # Set up HTTP client with appropriate timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            # Set headers based on provider_type
            headers = {"Content-Type": "application/json"}
            
            if provider_type == "azure":
                headers["api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
                
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
        """Get the status of all Azure instances."""
        all_stats = instance_manager.get_instance_stats()
        # Filter to only include Azure instances
        return [
            stats for stats in all_stats
            if stats.get("provider_type", "") == "azure"
        ]

    def select_instances_for_model(self, model_name: str, required_tokens: int = 0, exclude_instance_name: str = None) -> list:
        """
        Select Azure instances suitable for a specific model, ordered by suitability.
        This method can be shared with other services like streaming to ensure consistent selection logic.
        
        Args:
            model_name: The model name to use
            required_tokens: Required tokens for the request (for capacity checking)
            exclude_instance_name: Optional instance name to exclude (e.g., after a failure)
            
        Returns:
            List of instances ordered by suitability (primary instance first, then fallbacks)
        """
        # Get Azure instances that support this model
        azure_instances = self._get_azure_instances_for_model(model_name)
        
        if not azure_instances:
            return []
            
        # Select the best instance based on capacity and health
        primary_instance = self._select_primary_instance(azure_instances, required_tokens, model_name)
        
        available_instances = []
        if primary_instance:
            # Don't include if it matches the exclude name
            if not exclude_instance_name or primary_instance.get("name") != exclude_instance_name:
                available_instances = [primary_instance]  # Start with the best instance
            
            # Add other eligible instances as fallbacks
            for instance in azure_instances:
                instance_name = instance.get("name", "")
                # Skip if it's the primary instance or excluded instance
                if instance_name == primary_instance.get("name") or instance_name == exclude_instance_name:
                    continue
                
                # Only include healthy instances
                if instance.get("status") == "healthy":
                    available_instances.append(instance)
        else:
            # If no primary instance was found, use all healthy instances that support the model
            available_instances = [
                instance for instance in azure_instances 
                if instance.get("status") == "healthy" and 
                   (not exclude_instance_name or instance.get("name") != exclude_instance_name)
            ]
            
        # Sort by priority (lowest number = highest priority)
        available_instances.sort(key=lambda x: x.get("priority", 100))
        
        return available_instances

# Create a singleton instance
azure_openai_service = AzureOpenAIService()
