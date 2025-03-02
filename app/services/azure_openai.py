"""Azure OpenAI service for forwarding and transforming requests to multiple instances."""
import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import httpx
from fastapi import HTTPException, status

from app.utils.model_mappings import model_mapper
from app.utils.rate_limiter import rate_limiter, TokenUsage
from app.utils.instance_manager import instance_manager, AzureOpenAIInstance

logger = logging.getLogger(__name__)

class AzureOpenAIService:
    """Service for transforming and forwarding requests to Azure OpenAI instances."""
    
    def __init__(self):
        """Initialize the Azure OpenAI service."""
        logger.info("Initialized Azure OpenAI service with multi-instance support")
    
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
        
        # Look up the Azure deployment name
        deployment_name = model_mapper.get_deployment_name(model_name)
        if not deployment_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No Azure deployment mapped for model '{model_name}'",
            )
        
        # Clone the payload and remove the model field
        azure_payload = payload.copy()
        azure_payload.pop("model", None)
        
        # Estimate tokens for rate limiting and instance selection
        required_tokens = 0
        
        # For handling specific endpoints
        if endpoint == "/v1/chat/completions":
            # Estimate tokens for chat completions
            required_tokens = self._estimate_chat_tokens(azure_payload)
        
        # For completions endpoint
        elif endpoint == "/v1/completions":
            # Estimate tokens for standard completions
            required_tokens = self._estimate_completion_tokens(azure_payload)
        
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
    
    def _estimate_chat_tokens(self, payload: Dict[str, Any]) -> int:
        """
        Estimate the number of tokens in a chat completion request.
        
        Args:
            payload: The chat completion request payload
            
        Returns:
            Estimated token count
        """
        messages = payload.get("messages", [])
        model = payload.get("model", "gpt-3.5-turbo")
        max_tokens = payload.get("max_tokens", 256)  # Default if not specified
        
        # Estimate tokens for each message
        message_tokens = 0
        for message in messages:
            # Count tokens in the content
            content = message.get("content", "")
            if isinstance(content, str):
                message_tokens += rate_limiter.estimate_tokens(content, model)
            elif isinstance(content, list):  # Handle multi-modal content (list of objects)
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        message_tokens += rate_limiter.estimate_tokens(item.get("text", ""), model)
            
            # Add tokens for role and metadata (rough estimate)
            message_tokens += 4  # ~4 tokens per message overhead
        
        # Add tokens for completion (max_tokens)
        total_tokens = message_tokens + max_tokens
        
        return total_tokens
    
    def _estimate_completion_tokens(self, payload: Dict[str, Any]) -> int:
        """
        Estimate the number of tokens in a completion request.
        
        Args:
            payload: The completion request payload
            
        Returns:
            Estimated token count
        """
        prompt = payload.get("prompt", "")
        model = payload.get("model", "text-davinci-003")
        max_tokens = payload.get("max_tokens", 256)  # Default if not specified
        
        # Estimate tokens for the prompt
        if isinstance(prompt, str):
            prompt_tokens = rate_limiter.estimate_tokens(prompt, model)
        elif isinstance(prompt, list):
            prompt_tokens = sum(rate_limiter.estimate_tokens(p, model) for p in prompt if isinstance(p, str))
        else:
            prompt_tokens = 0
        
        # Add tokens for completion (max_tokens)
        total_tokens = prompt_tokens + max_tokens
        
        return total_tokens
    
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
        
        # Try to send the request to an available instance with failover
        result, instance = await instance_manager.try_instances(
            endpoint, 
            azure_deployment, 
            payload, 
            required_tokens,
            method
        )
        
        logger.debug(f"Request completed using instance {instance.name}")
        
        return result
        
    async def get_instances_status(self) -> List[Dict[str, Any]]:
        """Get the status of all instances."""
        return instance_manager.get_instance_stats()

# Create a singleton instance
azure_openai_service = AzureOpenAIService()
