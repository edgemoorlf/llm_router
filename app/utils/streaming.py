"""Utility functions for handling streaming requests to OpenAI-compatible services."""
import json
import logging
import httpx
from typing import Dict, Any
from fastapi import HTTPException, status
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse

from app.instance.manager import instance_manager
from app.instance.api_instance import InstanceStatus
from app.utils.token_estimator import estimate_chat_tokens, estimate_completion_tokens

logger = logging.getLogger(__name__)

async def handle_streaming_request(endpoint: str, deployment: str, payload: Dict[str, Any], provider_type: str = "azure", original_model: str = None) -> StreamingResponse:
    """
    Handle streaming requests by forwarding them to the API and streaming the response back.
    Supports multiple API instances with automatic failover.
    
    Args:
        endpoint: The API endpoint
        deployment: The deployment name
        payload: The request payload
        provider_type: The provider type ("azure" or "generic")
        original_model: The original model name from the client request
        
    Returns:
        A streaming response
    """
    # Ensure streaming is enabled
    payload["stream"] = True
    payload.pop("required_tokens", 1000)
    
    # Save the original model for adding to response chunks
    original_model = original_model or payload.get("model", "unknown")
    
    # Estimate tokens for rate limiting and instance selection
    required_tokens = 0
    if endpoint == "/v1/chat/completions":
        # Calculate precise token count using improved estimator
        required_tokens = estimate_chat_tokens(
            messages=payload.get("messages", []),
            functions=payload.get("functions"),
            model=payload.get("model", ""),
            provider=provider_type
        )
        
        # Validate against max input tokens before proceeding
        max_input_tokens = next((inst.max_input_tokens for inst in instance_manager.instances.values() 
                               if inst.max_input_tokens > 0), None)
        logger.debug(f"required tokens {required_tokens} vs max input tokens {max_input_tokens}")        
        if max_input_tokens and required_tokens > max_input_tokens:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Request exceeds maximum input tokens ({required_tokens} > {max_input_tokens})"
            )
        
        # Enforce strict token limits before instance selection
        if required_tokens <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token count estimation"
            )
    elif endpoint == "/v1/completions":
        required_tokens = estimate_completion_tokens(
            payload.get("prompt", ""),
            payload.get("model", ""),
            provider_type
        )
    else:
        required_tokens = 500  # Default for unknown endpoints
    
    # Extract model name from payload if available
    model_name = payload.get("model")
    
    # Use the original model name for instance selection when available
    model_for_selection = original_model if original_model else payload.get("model")
    
    # Select an instance based on the routing strategy and model support
    # Use the original model name to ensure proper matching against supported_models and model_deployments
    primary_instance = instance_manager.select_instance(required_tokens, model_for_selection)
    logger.debug(f"Selected primary instance for streaming: {primary_instance.name if primary_instance else 'None'} for model: {model_for_selection} max input tokens: {primary_instance.max_input_tokens if primary_instance else 'N/A'}")
    
    # If no instance was found that can handle this request with its token limits, return 503
    if not primary_instance:
        error_detail = f"No instances available that can handle {required_tokens} tokens for model '{model_for_selection}'"
        logger.error(error_detail)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_detail
        )
    
    # For Azure provider types, remove the model field since it uses deployment names
    # For generic provider types, keep the model field since it's required
    if provider_type == "azure":
        # Remove the model field for Azure provider types
        payload.pop("model", None)
        logger.debug(f"Removed model field for Azure provider type")
    else:
        logger.debug(f"Keeping model field for generic provider type")
    
    # Filter out instances in error state
    available_instances = [instance for instance in instance_manager.instances.values() 
                           if instance.status != InstanceStatus.ERROR]
    
    if not available_instances:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No available API instances for streaming request",
        )
    
    # Try primary instance first, then others in priority order
    if primary_instance:
        available_instances = [primary_instance] + [i for i in available_instances if i.name != primary_instance.name]
    
    # Sort remaining instances by priority
    available_instances.sort(key=lambda x: x.priority)
    
    # Try each instance until one succeeds
    last_error = None
    for instance in available_instances:
        try:
            # Initialize the instance if needed
            instance.initialize_client()
            
            # Build the URL for this instance based on provider type
            if provider_type == "azure":
                # Azure uses deployment names in the path
                url = instance.build_url(deployment_name=deployment, endpoint=endpoint)
            else:
                # Generic providers use model parameter
                url = instance.build_url(endpoint).replace("{model}", deployment)
                
            logger.debug(f"Constructed {provider_type} URL: {url}")
            
            logger.debug(f"Streaming request via instance {instance.name} to {url}")
            
            # Create a new client for streaming (we can't reuse the existing client)
            client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
            client.headers.update({"api-key": instance.api_key})
            
            # Make the request to the API
            response = await client.post(url, json=payload, headers={"Accept": "application/json"})
            
            # Check for HTTP errors
            response.raise_for_status()
    
            # Mark the instance as healthy
            instance.mark_healthy()
            
            if endpoint == "/v1/chat/completions":
                # For chat completions, we need to process each chunk to ensure it matches OpenAI format
                async def process_stream():
                    async for line in response.aiter_lines():
                        if not line or line.strip() == "":
                            continue
                            
                        if line.startswith("data:"):
                            line = line[5:].strip()
                            
                        # Handle the "[DONE]" message that indicates the end of the stream
                        if line == "[DONE]":
                            yield f"data: [DONE]\n\n"
                            continue
                            
                        try:
                            chunk = json.loads(line)
                            
                            # Ensure all required fields exist in the chunk
                            if "choices" not in chunk:
                                chunk["choices"] = []
                                
                            # If we have choices, make sure they have the right format
                            for choice in chunk["choices"]:
                                if "delta" not in choice and "text" in choice:
                                    # Convert from completions format to chat format if needed
                                    choice["delta"] = {"content": choice["text"]}
                                    choice.pop("text", None)
                                elif "delta" in choice and isinstance(choice["delta"], str):
                                    # Handle the case where delta might be a string
                                    choice["delta"] = {"content": choice["delta"]}
                                    
                            # Add model if it's missing
                            if "model" not in chunk:
                                chunk["model"] = original_model
                                
                            # Ensure correct object type
                            if "object" not in chunk:
                                chunk["object"] = "chat.completion.chunk"
                                
                            # Format the chunk as a server-sent event
                            yield f"data: {json.dumps(chunk)}\n\n"
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding streaming response JSON: {str(e)} - Line: {line}")
                            # Pass through the line as-is
                            yield f"data: {line}\n\n"
                        except Exception as e:
                            logger.error(f"Error processing streaming chunk: {str(e)}")
                            
                return StreamingResponse(
                    process_stream(),
                    media_type="text/event-stream",
                    background=BackgroundTask(client.aclose),
                )
            else:
                # For other endpoints, return the raw stream
                return StreamingResponse(
                    response.aiter_bytes(),
                    media_type="application/json",
                    background=BackgroundTask(client.aclose),
                )
                
        except httpx.HTTPStatusError as e:
            # Handle API errors
            error_detail = "Unknown error"
            
            try:
                error_response = e.response.json()
                error_detail = error_response.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            
            status_code = e.response.status_code
            
            # Handle rate limiting
            if status_code == 429:
                retry_after = int(e.response.headers.get("retry-after", "60"))
                instance.mark_rate_limited(retry_after)
                logger.warning(f"Instance {instance.name} rate limited, retrying with next instance")
                last_error = f"Rate limit exceeded: {error_detail}"
                continue
            
            # Handle other errors
            instance.mark_error(f"HTTP {status_code}: {error_detail}")
            logger.error(f"Instance {instance.name} returned error {status_code}: {error_detail}")
            last_error = f"API error: {error_detail}"
            continue
            
        except (httpx.RequestError, httpx.TimeoutException) as e:
            # Handle connection errors
            instance.mark_error(f"Connection error: {str(e)}")
            logger.error(f"Connection error to instance {instance.name}: {str(e)}")
            last_error = f"Connection error: {str(e)}"
            continue
            
        except Exception as e:
            # Handle unexpected errors
            instance.mark_error(f"Unexpected error: {str(e)}")
            logger.exception(f"Unexpected error with instance {instance.name}")
            last_error = f"Unexpected error: {str(e)}"
            continue
    
    # If we get here, all instances failed
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"All API instances failed for streaming request. Last error: {last_error}",
    )
