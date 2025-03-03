"""Azure OpenAI service for forwarding and transforming requests to Azure OpenAI instances."""
import logging
from typing import Any, Dict, List
from fastapi import HTTPException, status

from app.utils.model_mappings import model_mapper
from app.utils.rate_limiter import rate_limiter
from app.utils.instance_manager import instance_manager
from app.utils.token_estimator import estimate_chat_tokens, estimate_completion_tokens

logger = logging.getLogger(__name__)

class AzureOpenAIService:
    """Service for transforming and forwarding requests specifically to Azure OpenAI instances."""
    
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
        
        # Normalize the model name
        normalized_model = model_name.lower().split(':')[0]
        
        # Find Azure instances that support this model
        azure_instances = [
            instance for instance in instance_manager.instances.values()
            if (
                # Must be Azure provider type
                instance.provider_type == "azure" and
                (
                    # Instance has this model in its supported_models list
                    normalized_model in [m.lower() for m in instance.supported_models] or
                    # Instance has a deployment mapping for this model
                    normalized_model in instance.model_deployments or
                    # Instance has no model restrictions (empty supported_models means it supports all models)
                    not instance.supported_models
                )
            )
        ]
        
        logger.debug(f"Found {len(azure_instances)} Azure instances supporting model '{normalized_model}'")
        
        if not azure_instances:
            logger.warning(f"No Azure instances found supporting model '{model_name}'")
        
        # Get the deployment name from the first instance that has a mapping for this model
        deployment_name = None
        for instance in azure_instances:
            if normalized_model in instance.model_deployments:
                deployment_name = instance.model_deployments[normalized_model]
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
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Global rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )
        
        # Log the transformation
        logger.debug(f"Transformed request for model '{model_name}' to Azure deployment '{deployment_name}' (est. {required_tokens} tokens)")
        
        return {
            "azure_deployment": deployment_name,
            "payload": azure_payload,
            "required_tokens": required_tokens
        }
        
    async def forward_request(
        self, endpoint: str, azure_deployment: str, payload: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        """
        Forward a request to an available Azure OpenAI instance with automatic failover.
        
        Args:
            endpoint: The API endpoint
            azure_deployment: The Azure deployment name
            payload: The request payload
            method: The HTTP method
            
        Returns:
            The API response
        """
        # Get the estimated token requirement from the payload
        required_tokens = payload.pop("required_tokens", 1000)  # Default if not available
        
        # Filter instances to only include Azure provider types
        azure_instances = [
            instance for instance in instance_manager.instances.values()
            if instance.provider_type == "azure"
        ]
        
        if not azure_instances:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No Azure OpenAI instances available",
            )
        
        # Select an Azure instance based on the routing strategy
        primary_instance = instance_manager.select_instance(required_tokens, azure_deployment)
        
        # Ensure we only selected an Azure instance
        if primary_instance and primary_instance.provider_type != "azure":
            logger.warning(f"Non-Azure instance {primary_instance.name} selected, filtering to Azure only")
            primary_instance = None
            
            # Try to find a suitable Azure instance
            for instance in azure_instances:
                if azure_deployment in instance.model_deployments.values():
                    primary_instance = instance
                    break
            
            # If still no instance found, use the first Azure instance
            if not primary_instance and azure_instances:
                primary_instance = azure_instances[0]
        
        # Try to send the request to an available Azure instance with failover
        # We'll only try Azure instances
        result, instance = await instance_manager.try_instances(
            endpoint, 
            azure_deployment, 
            payload, 
            required_tokens,
            method,
            provider_type="azure"  # Only try Azure instances
        )
        
        logger.debug(f"Request completed using Azure instance {instance.name}")
        
        return result
        
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
