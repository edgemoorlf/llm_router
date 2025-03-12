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
        
        # Find Azure instances that support this model with strict compatibility checking
        azure_instances = [
            instance for instance in instance_manager.instances.values()
            if (
                # Must be Azure provider type
                instance.provider_type == "azure" and
                # Strict model compatibility check - exact match in supported_models only
                exact_model_name in [m.lower() for m in instance.supported_models]
            )
        ]
        
        logger.debug(f"Found {len(azure_instances)} Azure instances with strict model compatibility for '{model_name}'")
        
        if not azure_instances:
            logger.warning(f"No Azure instances found with strict model compatibility for '{model_name}'")
        
        # Get the deployment name from the first instance that has a mapping for this model
        deployment_name = None
        for instance in azure_instances:
            # Use only exact model name matching
            if exact_model_name in instance.model_deployments:
                deployment_name = instance.model_deployments[exact_model_name]
                logger.debug(f"Found deployment mapping for model '{model_name}' in Azure instance '{instance.name}': {deployment_name}")
                break
        
        # If no instance has a mapping, fall back to the global model mapper
        if not deployment_name:
            deployment_name = model_mapper.get_deployment_name(model_name)
            logger.debug(f"Using global model mapper for model '{model_name}': {deployment_name}")
        
        if not deployment_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No deployment mapped for model '{model_name}' in any Azure instance",
            )
        
        # Clone the payload and remove the model field since Azure uses deployment names
        azure_payload = payload.copy()
        azure_payload.pop("model", None)
        
        # Estimate tokens for rate limiting and instance selection
        required_tokens = 0
        
        # For handling specific endpoints
        if endpoint == "/v1/chat/completions":
            # Estimate tokens for chat completions
            required_tokens = estimate_chat_tokens(
                azure_payload.get("messages", []),
                azure_payload.get("functions", None),
                model_name,
                "azure"
            )

        # For completions endpoint
        elif endpoint == "/v1/completions":
            # Estimate tokens for standard completions
            required_tokens = estimate_completion_tokens(azure_payload)
        
        # For embeddings endpoint (rough estimate)
        elif endpoint == "/v1/embeddings":
            # Estimate tokens for embeddings
            input_text = azure_payload.get("input", "")
            if isinstance(input_text, str):
                required_tokens = len(input_text.split()) * 2  # Rough approximation
            elif isinstance(input_text, list):
                required_tokens = sum(len(text.split()) * 2 for text in input_text if isinstance(text, str))
        
        # For other endpoints, use a minimum token count for rate limiting
        else:
            required_tokens = 100  # Minimum token count for unknown endpoints
        
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
        
        # Log the transformation
        logger.debug(f"Transformed request for model '{model_name}' to Azure deployment '{deployment_name}' (est. {required_tokens} tokens)")
        
        azure_payload["required_tokens"] = required_tokens
        logger.debug(f"azure_payload required_tokens {required_tokens}")
        return {
            "azure_deployment": deployment_name,
            "original_model": model_name,  # Preserve the original model name for instance matching
            "payload": azure_payload
        }
        
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
        self, endpoint: str, azure_deployment: str, payload: Dict[str, Any], original_model: str = None, method: str = "POST"
    ) -> Dict[str, Any]:
        """Forward request to Azure instances with failover handling."""
        required_tokens = payload.pop("required_tokens", 1000)
        azure_instances = self._get_azure_instances()
        
        if not azure_instances:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="No Azure instances available"
            )
            
        # Use the original model name for instance selection when available
        model_for_selection = original_model if original_model else azure_deployment
        
        # Log which model name is being used for selection
        if original_model:
            logger.debug(f"Using original model '{original_model}' for instance selection")
        else:
            logger.warning(f"Original model name not available, using deployment '{azure_deployment}' for instance selection")
            
        primary_instance = self._select_primary_instance(azure_instances, required_tokens, model_for_selection)
        
        # If no instance is found that can handle this request with its token limits, return 503
        if not primary_instance:
            error_detail = f"No instances available that can handle {required_tokens} tokens for model '{model_for_selection}'"
            logger.error(error_detail)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_detail
            )
            
        return await self._execute_request(endpoint, azure_deployment, payload, required_tokens, method, primary_instance)

    def _get_azure_instances(self) -> list:
        """Get all available Azure instances."""
        return [i for i in instance_manager.instances.values() if i.provider_type == "azure"]

    def _select_primary_instance(self, instances: list, tokens: int, model_or_deployment: str) -> Optional[object]:
        """
        Select primary instance based on routing strategy with strict model compatibility checking.
        
        Args:
            instances: List of available Azure instances
            tokens: Required tokens for the request
            model_or_deployment: The original model name (preferred) or azure deployment name
        """
        # Filter instances to only those that explicitly support this model
        eligible_instances = []
        for instance in instances:
            # Strict model compatibility check - exact match in supported_models only
            if model_or_deployment in instance.supported_models:
                # Verify token capacity
                if instance.instance_stats.current_tpm + tokens <= instance.max_tpm:
                    # Verify instance is healthy
                    if instance.status == "healthy":
                        eligible_instances.append(instance)
                
        # If we found eligible instances, select one based on the routing strategy
        if eligible_instances:
            # Choose the instance with the lowest load
            selected_instance = min(eligible_instances, key=lambda i: i.instance_stats.current_tpm)
            logger.debug(f"Selected instance '{selected_instance.name}' with strict model compatibility for '{model_or_deployment}'")
            # Store the model used for selection to ensure consistent fallback selection
            setattr(selected_instance, "_original_model_for_selection", model_or_deployment)
            return selected_instance
            
        # If no eligible instances found, try using the instance manager's selection
        # This is a fallback mechanism that might use more complex routing rules
        instance = instance_manager.select_instance(tokens, model_or_deployment)
        if instance and instance.provider_type == "azure" and model_or_deployment in instance.supported_models:
            logger.debug(f"Selected instance '{instance.name}' via instance manager with strict model compatibility for '{model_or_deployment}'")
            # Store the model used for selection to ensure consistent fallback selection
            setattr(instance, "_original_model_for_selection", model_or_deployment)
            return instance
            
        # If no instances are available that meet our criteria, return None
        logger.error(f"No instances available with strict model compatibility for '{model_or_deployment}' and capacity for {tokens} tokens")
        return None
        
    async def _execute_request(
        self, endpoint: str, deployment: str, payload: Dict[str, Any], required_tokens: int, method: str, instance: Optional[object]
    ) -> Dict[str, Any]:
        """Execute the request with failover handling."""
        # We already check for None in forward_request, but add an assertion for safety
        assert instance is not None, "Instance should not be None at this point"
        
        # Preserve the original model for consistent instance selection in fallbacks
        original_model = getattr(instance, "_original_model_for_selection", deployment)
            
        try:
            # Use the already selected instance directly instead of redoing selection
            logger.debug(f"Executing request using Azure instance {instance.name} for deployment {deployment}")
            
            # Forward the request directly to the selected instance
            result = await instance_manager.forwarder.forward_request(
                instance, endpoint, deployment, payload, method
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
                logger.warning(f"Content policy violation (HTTP {e.status_code}) - prompt: {payload.get('messages', '')}")
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
            
            # Get all eligible Azure instances that support this model
            # This ensures we use the same selection criteria as the primary selection
            azure_instances = self._get_azure_instances()
            eligible_instances = []
            
            for candidate in azure_instances:
                # Don't include the failed instance
                if candidate.name == instance.name:
                    continue
                    
                # Strict model compatibility check - exact match in supported_models only
                model_supported = original_model in candidate.supported_models
                    
                # Verify token capacity
                token_capacity_sufficient = candidate.instance_stats.current_tpm + required_tokens <= candidate.max_tpm
                
                # Check instance is healthy (not rate limited or in error state)
                is_healthy = candidate.status == "healthy"
                
                if model_supported and token_capacity_sufficient and is_healthy:
                    eligible_instances.append(candidate)
            
            # Log the number of eligible instances with model support information
            logger.debug(f"Found {len(eligible_instances)} eligible fallback instances with strict model support for '{original_model}'")
            
            # Try all eligible instances
            if eligible_instances:
                fallback_errors = []
                
                for fallback_instance in eligible_instances:
                    try:
                        # Forward the request to the fallback instance
                        logger.debug(f"Trying fallback instance {fallback_instance.name} for model {original_model}")
                        
                        result = await instance_manager.forwarder.forward_request(
                            fallback_instance, endpoint, deployment, payload, method
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
                logger.error(f"No eligible fallback instances found with strict model compatibility for '{original_model}' and capacity for {required_tokens} tokens")
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
