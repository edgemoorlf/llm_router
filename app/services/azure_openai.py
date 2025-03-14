"""Azure OpenAI service for forwarding and transforming requests to Azure OpenAI instances using DRY request handling."""
import logging
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from app.instance.manager import instance_manager
from app.utils.model_mappings import model_mapper
from app.utils.rate_limiter import rate_limiter
from app.utils.token_estimator import estimate_chat_tokens, estimate_completion_tokens

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
            allowed, retry_after = rate_limiter.check_and_update(required_tokens)
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
        
        # Get Azure instances that support this model
        azure_instances = self._get_azure_instances_for_model(original_model)
        
        if not azure_instances:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail=f"No Azure instances available that support model '{original_model}'"
            )
            
        # Select the best instance based on capacity and health
        primary_instance = self._select_primary_instance(azure_instances, required_tokens, original_model)
        
        # If no instance is found that can handle this request with its token limits, return 503
        if not primary_instance:
            error_detail = f"No instances available that can handle {required_tokens} tokens for model '{original_model}'"
            logger.error(error_detail)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_detail
            )
            
        return await self._execute_request(endpoint, original_model, payload, required_tokens, method, primary_instance)

    def _get_azure_instances(self) -> list:
        """Get all available Azure instances."""
        return [i for i in instance_manager.get_all_instances().values() if i.provider_type == "azure"]
        
    def _get_azure_instances_for_model(self, model_name: str) -> list:
        """Get Azure instances that support a specific model."""
        return [
            instance for instance in self._get_azure_instances()
            if (
                # Strict model compatibility check - exact match in supported_models only
                model_name.lower() in [m.lower() for m in instance.supported_models]
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
            # Verify token capacity
            if instance.instance_stats.current_tpm + tokens <= instance.max_tpm:
                # Verify instance is healthy
                if instance.status == "healthy":
                    eligible_instances.append(instance)
                
        # If we found eligible instances, select one based on the lowest load
        if eligible_instances:
            # Choose the instance with the lowest load
            selected_instance = min(eligible_instances, key=lambda i: i.instance_stats.current_tpm)
            logger.debug(f"Selected instance '{selected_instance.name}' for model '{model_name}'")
            # Store the model used for selection to ensure consistent fallback selection
            setattr(selected_instance, "_original_model_for_selection", model_name)
            return selected_instance
            
        # If no eligible instances found with capacity, try using the instance manager's selection
        # This is a fallback mechanism that might use more complex routing rules
        instance = instance_manager.select_instance(tokens, model_name)
        if instance and instance.provider_type == "azure" and model_name.lower() in [m.lower() for m in instance.supported_models]:
            logger.debug(f"Selected instance '{instance.name}' via instance manager for model '{model_name}'")
            # Store the model used for selection to ensure consistent fallback selection
            setattr(instance, "_original_model_for_selection", model_name)
            return instance
            
        # If no instances are available that meet our criteria, return None
        logger.error(f"No instances available with capacity for {tokens} tokens for model '{model_name}'")
        return None
        
    async def _execute_request(
        self, endpoint: str, model_name: str, payload: Dict[str, Any], required_tokens: int, method: str, instance: Optional[object]
    ) -> Dict[str, Any]:
        """Execute the request with failover handling."""
        # We already check for None in forward_request, but add an assertion for safety
        assert instance is not None, "Instance should not be None at this point"
        
        # Preserve the original model for consistent instance selection in fallbacks
        original_model = getattr(instance, "_original_model_for_selection", model_name)
        
        # Ensure the model name is in the payload for deployment name resolution
        payload_with_model = payload.copy()
        payload_with_model["model"] = original_model
            
        try:
            # Use the already selected instance directly
            logger.debug(f"Executing request using Azure instance {instance.name} for model {model_name}")
            
            # Forward the request directly to the selected instance
            result = await instance_manager.forwarder.forward_request(
                instance, endpoint, payload_with_model, method
            )
            
            # Mark instance as healthy and update TPM
            instance.mark_healthy()
            if "usage" in result and "total_tokens" in result["usage"]:
                instance.update_tpm_usage(result["usage"]["total_tokens"])
                
            logger.debug(f"Request completed using Azure instance {instance.name}")
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
            logger.warning(f"Primary instance {instance.name} failed with error {e.status_code}: {e.detail}, falling back to other instances")
            
            # Update instance status based on error
            if e.status_code == 429:
                retry_after = None
                if hasattr(e, 'headers') and e.headers and 'retry-after' in e.headers:
                    try:
                        retry_after = int(e.headers['retry-after'])
                    except (ValueError, TypeError):
                        pass
                instance.mark_rate_limited(retry_after)
            else:
                instance.mark_error(str(e))
            
            # Get all eligible Azure instances that support this model (excluding the failed one)
            azure_instances = self._get_azure_instances_for_model(original_model)
            eligible_instances = []
            
            for candidate in azure_instances:
                # Don't include the failed instance
                if candidate.name == instance.name:
                    continue
                    
                # Verify token capacity
                token_capacity_sufficient = candidate.instance_stats.current_tpm + required_tokens <= candidate.max_tpm
                
                # Check instance is healthy (not rate limited or in error state)
                is_healthy = candidate.status == "healthy"
                
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
                        logger.debug(f"Trying fallback instance {fallback_instance.name} for model {original_model}")
                        
                        result = await instance_manager.forwarder.forward_request(
                            fallback_instance, endpoint, payload_with_model, method
                        )
                        
                        # Mark instance as healthy and update TPM
                        fallback_instance.mark_healthy()
                        if "usage" in result and "total_tokens" in result["usage"]:
                            fallback_instance.update_tpm_usage(result["usage"]["total_tokens"])
                            
                        logger.debug(f"Request completed using fallback Azure instance {fallback_instance.name}")
                        return self._clean_response(result)
                    except Exception as fallback_e:
                        error_msg = str(fallback_e)
                        logger.warning(f"Fallback instance {fallback_instance.name} failed with error: {error_msg}")
                        fallback_errors.append(f"{fallback_instance.name}: {error_msg}")
                        
                        # Update fallback instance status
                        if isinstance(fallback_e, HTTPException) and fallback_e.status_code == 429:
                            retry_after = None
                            if hasattr(fallback_e, 'headers') and fallback_e.headers and 'retry-after' in fallback_e.headers:
                                try:
                                    retry_after = int(fallback_e.headers['retry-after'])
                                except (ValueError, TypeError):
                                    pass
                            fallback_instance.mark_rate_limited(retry_after)
                        else:
                            fallback_instance.mark_error(error_msg)
                
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

    async def get_instances_status(self) -> List[Dict[str, Any]]:
        """Get the status of all Azure instances."""
        all_stats = instance_manager.get_instance_stats()
        # Filter to only include Azure instances
        return [
            stats for stats in all_stats
            if stats["provider_type"] == "azure"
        ]

# Create a singleton instance
azure_openai_service = AzureOpenAIService()
