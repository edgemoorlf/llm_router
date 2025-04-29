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
from app.instance.instance_context import instance_manager
from app.utils.url_builder import build_instance_url
from app.services.instance_selector import instance_selector
from app.services.error_handler import error_handler
from app.services.request_transformer import request_transformer
from app.services.response_cleaner import response_cleaner

logger = logging.getLogger(__name__)

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
        # Delegate to the request transformer service
        transformed = request_transformer.transform_openai_to_azure(endpoint, payload)
        
        # Verify there are Azure instances that support this model
        azure_instances = instance_selector.get_instances_for_model(
            transformed["original_model"], 
            provider_type="azure"
        )
        
        if not azure_instances:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No instances found that support model '{payload.get('model')}'",
            )
            
        return transformed

    async def forward_request(
        self, endpoint: str, payload: Dict[str, Any], original_model: str, method: str = "POST"
    ) -> Dict[str, Any]:
        """Forward request to Azure instances with retry and fallback handling."""
        required_tokens = payload.pop("required_tokens", 1000)
        exclude_instances = set()  # Track excluded instances
        max_retries = 3
        retry_count = 0

        request_id = f"req-{time.time()}"
        
        logger.debug(f"[{request_id}] Forwarding request to Azure instances. Model: {original_model}, Tokens: {required_tokens}")
        
        while retry_count < max_retries:
            try:
                # Get available instances, excluding previously failed ones
                available_instances = instance_selector.select_instances_for_model(
                    original_model, 
                    required_tokens,
                    provider_type="azure",
                    exclude_instance_names=exclude_instances
                )
                
                if available_instances:
                    instance_names = [inst.get("name", "") for inst in available_instances]
                    logger.debug(f"[{request_id}] Selected {len(available_instances)} instances for request: {', '.join(instance_names)}")
                else:
                    logger.warning(f"[{request_id}] No instances available for model '{original_model}' with {required_tokens} tokens")
                
                if not available_instances:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                        detail=f"[{request_id}] No instances available that support model '{original_model}'"
                    )
                
                # Execute request with fallback handling
                result = await self._execute_with_fallbacks(
                    request_id=request_id,
                    endpoint=endpoint,
                    model_name=original_model,
                    payload=payload,
                    required_tokens=required_tokens,
                    method=method,
                    available_instances=available_instances,
                    exclude_instances=exclude_instances
                )
                
                # Clean the response using the response cleaner service
                return response_cleaner.clean_azure_response(result)
                
            except HTTPException as e:
                # Check for special errors that should propagate immediately
                error_handler.handle_special_error(e, "global")
                
                # If this was the last retry, re-raise
                if retry_count >= max_retries - 1:
                    logger.error(f"[{request_id}] All retries failed for model '{original_model}'. Last error: {str(e)}")
                    raise
                
                # Prepare for next retry
                retry_count += 1
                logger.warning(f"[{request_id}] All instances failed, retry {retry_count}/{max_retries}")
                await asyncio.sleep(0.5 * retry_count)  # Increasing backoff

    async def _execute_with_fallbacks(
        self, 
        request_id: str,
        endpoint: str, 
        model_name: str, 
        payload: Dict[str, Any], 
        required_tokens: int,
        method: str, 
        available_instances: List[Dict[str, Any]],
        exclude_instances: set
    ) -> Dict[str, Any]:
        """Execute request with a list of available instances, trying fallbacks as needed."""
        fallback_errors = []
        
        # Shuffle the instances to randomize the order
        random.shuffle(available_instances)

        # Try each instance in sequence until success or all fail
        for instance in available_instances:
            instance_name = instance.get("name", "")
            
            # Skip already excluded instances
            if instance_name in exclude_instances:
                continue
                
            # Prepare payload with model
            payload_with_model = payload.copy()
            payload_with_model["model"] = model_name
            
            try:
                logger.debug(f"[{request_id}] Attempting request with instance {instance_name}")
                
                # Forward the request directly to this instance
                result = await self._forward_request_to_instance(
                    request_id, instance, endpoint, payload_with_model, method
                )
                
                # Update instance metrics on success
                instance_manager.mark_healthy(instance_name)
                
                # Record the request with token usage
                if "usage" in result and "total_tokens" in result["usage"]:
                    tokens = result["usage"]["total_tokens"]
                    
                    # Record basic stats first
                    instance_manager.record_request(
                        name=instance_name,
                        success=True,
                        tokens=0  # Don't count tokens here to avoid double counting
                    )
                    
                    # Update token usage using the rate limiter
                    instance_manager.update_token_usage(instance_name, tokens)
                else:
                    # If no token usage information is available, just record the request
                    instance_manager.record_request(
                        name=instance_name,
                        success=True,
                        tokens=0
                    )
                    
                logger.debug(f"[{request_id}] Request completed using Azure instance {instance_name}")
                return result
                
            except HTTPException as e:
                # Handle special errors that should propagate immediately
                error_handler.handle_special_error(e, instance_name)
                    
                # Update instance status and track the error
                error_handler.handle_instance_error(instance, e)
                exclude_instances.add(instance_name)
                fallback_errors.append(f"{instance_name}: {str(e)}")
                
                # Log failure and continue to next instance
                logger.warning(f"[{request_id}] Instance {instance_name} failed with error {str(e)}, trying next instance")
        
        # If we get here, all instances failed
        error_detail = f"[{request_id}] All instances failed for model '{model_name}'. Errors: {', '.join(fallback_errors)}"
        logger.error(error_detail)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_detail
        )

    async def _forward_request_to_instance(
        self, request_id: str, instance: Dict[str, Any], endpoint: str, payload: Dict[str, Any], method: str = "POST"
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
        api_key = instance.get("api_key", "")
        model_deployments = instance.get("model_deployments", {})
        timeout_seconds = instance.get("timeout_seconds", 60.0)
        retry_count = instance.get("retry_count", 5)
        
        # Extract model from payload
        model_name = payload.get("model", "").lower() if payload and "model" in payload else ""
        
        # Determine deployment name based on the model
        deployment = ""
        if model_name and provider_type == "azure" and model_deployments:
            # Look up the deployment name for this model in this specific instance
            deployment = model_deployments.get(model_name, "")
            if deployment:
                logger.debug(f"[{request_id}] Resolved deployment '{deployment}' for model '{model_name}' in instance '{instance_name}'")
            else:
                logger.warning(f"[{request_id}] No deployment mapping found for model '{model_name}' in instance '{instance_name}'")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"[{request_id}] No deployment mapping found for model '{model_name}' in instance '{instance_name}'"
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
                detail=f"[{request_id}] Could not build URL for instance '{instance.get('name', '')}'"
            )
        
        logger.debug(f"[{request_id}] Forwarding request to instance {instance_name} ({provider_type}): {url}")
        
        # Set up HTTP client with appropriate timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), proxies={"http://": instance.get("proxy_url")}) as client:
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
                    if e.response.status_code in [400, 401, 403, 404, 429]:
                        # For 429, we should break to let the higher level fallback logic handle it
                        # rather than retrying the same rate-limited instance
                        if e.response.status_code == 429:
                            logger.warning(f"[{request_id}] Instance {instance_name} is rate limited (HTTP 429). Breaking retry loop to try other instances.")
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

    def select_instances_for_model(self, model_name: str, required_tokens: int = 0, exclude_instance_name: str = None, exclude_instance_names: set = None) -> list:
        """
        Select Azure instances suitable for a specific model, ordered by suitability.
        
        This method is now a wrapper around the dedicated instance_selector service to maintain backward compatibility
        while reducing code duplication.
        
        Args:
            model_name: The model name to use
            required_tokens: Required tokens for the request (for capacity checking)
            exclude_instance_name: Optional instance name to exclude (e.g., after a failure) - DEPRECATED
            exclude_instance_names: Optional set of instance names to exclude (e.g., after failures)
            
        Returns:
            List of instances ordered by suitability (primary instance first, then fallbacks)
        """
        # For backward compatibility
        excluded = set()
        if exclude_instance_names:
            excluded = exclude_instance_names
        elif exclude_instance_name:
            excluded.add(exclude_instance_name)
            
        # Delegate to the instance selector service
        return instance_selector.select_instances_for_model(
            model_name,
            required_tokens,
            provider_type="azure",
            exclude_instance_names=excluded
        )

# Create a singleton instance
azure_openai_service = AzureOpenAIService()
