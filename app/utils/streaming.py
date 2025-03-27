"""Utility functions for handling streaming requests to OpenAI-compatible services."""
import json
import logging
import httpx
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, status, Response
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse

from app.instance.instance_context import instance_manager
from app.services.instance_selector import instance_selector
from app.utils.url_builder import build_instance_url

logger = logging.getLogger(__name__)

async def handle_streaming_request(endpoint: str, payload: Dict[str, Any], provider_type: str = "azure", original_model: str = None) -> StreamingResponse:
    """
    Handle streaming requests by forwarding them to the API and streaming the response back.
    Supports multiple API instances with automatic failover.
    
    Args:
        endpoint: The API endpoint
        payload: The request payload
        provider_type: The provider type ("azure" or "generic")
        original_model: The original model name from the client request
        
    Returns:
        A streaming response
    """
    # Ensure streaming is enabled
    payload["stream"] = True
    
    # Save the original model for adding to response chunks
    model_name = original_model or payload.get("model", "unknown")
    
    # Ensure model_name is in the payload for Azure deployment mapping
    if "model" not in payload and model_name:
        payload["model"] = model_name

    # Estimate tokens in the request if it's a chat completion
    required_tokens = payload.pop("required_tokens", 64)

    # Get all eligible instances for this model/request directly from the instance selector
    instances = instance_selector.select_instances_for_model(
        model_name=model_name,
        required_tokens=required_tokens,
        provider_type=provider_type
    )
    
    if not instances:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No instances available that support model '{model_name}' for streaming request",
        )
    
    # Extract instance names for easier handling
    eligible_instance_names = [instance.get("name") for instance in instances]
    
    # Keep track of tried instances to avoid retrying failed ones
    tried_instances = set()
    last_error = None
    
    # Try instances until one succeeds or all fail
    for instance_name in eligible_instance_names:
        if instance_name in tried_instances:
            continue
            
        tried_instances.add(instance_name)
        
        # Get instance config and state
        config = instance_manager.get_instance_config(instance_name)
        state = instance_manager.get_instance_state(instance_name)
        
        if not config or not state:
            logger.warning(f"Failed to get configuration or state for instance '{instance_name}', trying next instance")
            continue
            
        # Check max input tokens
        max_input_tokens = config.max_input_tokens
        if required_tokens > max_input_tokens:
            logger.warning(f"Input tokens ({required_tokens}) exceed maximum allowed ({max_input_tokens}) for instance '{instance_name}', trying next instance")
            continue
            
        # Check rate limit
        allowed = instance_manager.check_rate_limit(instance_name, required_tokens)
        if not allowed:
            logger.warning(f"Rate limit exceeded for instance '{instance_name}', trying next instance")
            continue
        
        try:
            # Combine config and state for URL building
            instance = config.dict()
            instance.update(state.dict())
            instance["_current_model"] = model_name
            
            # Build the URL using shared logic
            url = build_instance_url(instance, endpoint)
            if not url:
                logger.warning(f"Failed to build URL for instance '{instance_name}', trying next instance")
                continue
            
            logger.debug(f"Streaming request via instance {instance_name} to {url}")
            
            # Create a new client for streaming
            client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
            
            # Set appropriate headers based on provider type
            if config.provider_type == "azure":
                client.headers.update({"api-key": config.api_key})
            else:
                client.headers.update({"Authorization": f"Bearer {config.api_key}"})
            
            # Make the request to the API
            response = await client.post(url, json=payload, headers={"Accept": "application/json"})
            
            # Check for HTTP errors
            response.raise_for_status()

            # Mark the instance as healthy
            instance_manager.mark_healthy(instance_name)
            
            if endpoint == "/v1/chat/completions":
                # Process chat completions stream
                async def process_stream():
                    try:
                        async for line in response.aiter_lines():
                            if not line or line.strip() == "":
                                continue
                                
                            if line.startswith("data:"):
                                line = line[5:].strip()
                                
                            if line == "[DONE]":
                                yield f"data: [DONE]\n\n"
                                continue
                                
                            try:
                                chunk = json.loads(line)
                                
                                # Ensure all required fields exist
                                if "choices" not in chunk:
                                    chunk["choices"] = []
                                    
                                # Format choices correctly
                                for choice in chunk["choices"]:
                                    if "delta" not in choice and "text" in choice:
                                        choice["delta"] = {"content": choice["text"]}
                                        choice.pop("text", None)
                                    elif "delta" in choice and isinstance(choice["delta"], str):
                                        choice["delta"] = {"content": choice["delta"]}
                                        
                                # Add model if missing
                                if "model" not in chunk:
                                    chunk["model"] = model_name
                                    
                                # Ensure correct object type
                                if "object" not in chunk:
                                    chunk["object"] = "chat.completion.chunk"
                                    
                                yield f"data: {json.dumps(chunk)}\n\n"
                            except json.JSONDecodeError as e:
                                logger.error(f"Error decoding streaming response JSON: {str(e)} - Line: {line}")
                                yield f"data: {line}\n\n"
                            except Exception as e:
                                logger.error(f"Error processing streaming chunk: {str(e)}")
                    finally:
                        # Update instance metrics after streaming completes
                        try:
                            instance_manager.record_request(
                                name=instance_name,
                                success=True,
                                tokens=0  # Don't count tokens here to avoid double counting
                            )
                            
                            # Update token usage separately
                            instance_manager.update_token_usage(instance_name, required_tokens)
                        except Exception as e:
                            logger.error(f"Error updating metrics for {instance_name}: {str(e)}")
                        
                return StreamingResponse(
                    process_stream(),
                    media_type="text/event-stream",
                    background=BackgroundTask(client.aclose),
                )
            else:
                # Process other endpoints stream
                async def process_stream():
                    try:
                        async for line in response.aiter_lines():
                            if not line or line.strip() == "":
                                continue
                                
                            if line.startswith("data:"):
                                yield f"{line}\n\n"
                                continue
                                
                            if line == "[DONE]":
                                yield f"data: [DONE]\n\n"
                                continue
                                
                            try:
                                chunk = json.loads(line)
                                
                                if "choices" not in chunk:
                                    chunk["choices"] = []
                                    
                                if "model" not in chunk:
                                    chunk["model"] = model_name
                                    
                                if "object" not in chunk:
                                    chunk["object"] = "text_completion.chunk"
                                    
                                yield f"data: {json.dumps(chunk)}\n\n"
                            except json.JSONDecodeError as e:
                                logger.error(f"Error decoding streaming response JSON: {str(e)} - Line: {line}")
                                yield f"data: {line}\n\n"
                            except Exception as e:
                                logger.error(f"Error processing streaming chunk: {str(e)}")
                    finally:
                        # Update instance metrics after streaming completes
                        try:
                            instance_manager.record_request(
                                name=instance_name,
                                success=True,
                                tokens=0  # Don't count tokens here to avoid double counting
                            )
                            
                            # Update token usage separately
                            instance_manager.update_token_usage(instance_name, required_tokens)
                        except Exception as e:
                            logger.error(f"Error updating metrics for {instance_name}: {str(e)}")
                
                return StreamingResponse(
                    process_stream(),
                    media_type="text/event-stream",
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
            
            # Record the error and update instance state
            try:
                instance_manager.record_request(
                    name=instance_name,
                    success=False,
                    error=error_detail
                )
            except Exception as record_err:
                logger.error(f"Error recording request failure for {instance_name}: {str(record_err)}")
            
            # Handle rate limiting
            if status_code == 429:
                retry_after = int(e.response.headers.get("retry-after", "60"))
                instance_manager.mark_rate_limited(instance_name, retry_after)
            else:
                instance_manager.update_instance_state(
                    instance_name,
                    status="error",
                    last_error=error_detail
                )
            
            # Save the error and try the next instance
            last_error = HTTPException(
                status_code=status_code,
                detail=error_detail,
                headers=e.response.headers
            )
            logger.warning(f"Instance {instance_name} failed with {status_code}: {error_detail}, trying next instance")
            continue
            
        except (httpx.RequestError, httpx.TimeoutException) as e:
            # Handle connection errors
            error_detail = f"Connection error: {str(e)}"
            try:
                instance_manager.record_request(
                    name=instance_name,
                    success=False,
                    error=error_detail
                )
            except Exception as record_err:
                logger.error(f"Error recording request failure for {instance_name}: {str(record_err)}")
                
            instance_manager.update_instance_state(
                instance_name,
                status="error",
                last_error=error_detail
            )
            
            # Save the error and try the next instance
            last_error = HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_detail
            )
            logger.warning(f"Instance {instance_name} connection error: {error_detail}, trying next instance")
            continue
            
        except Exception as e:
            # Handle unexpected errors
            error_detail = f"Unexpected error: {str(e)}"
            try:
                instance_manager.record_request(
                    name=instance_name,
                    success=False,
                    error=error_detail
                )
            except Exception as record_err:
                logger.error(f"Error recording request failure for {instance_name}: {str(record_err)}")
                
            instance_manager.update_instance_state(
                instance_name,
                status="error",
                last_error=error_detail
            )
            
            # Save the error and try the next instance
            last_error = HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_detail
            )
            logger.warning(f"Instance {instance_name} unexpected error: {error_detail}, trying next instance")
            continue
    
    # If we've tried all instances and all failed, raise the last error
    if last_error:
        logger.error(f"All instances failed for streaming request. Last error: {last_error.detail}")
        raise last_error
        
    # Should never reach here, but just in case
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"No instances available for streaming request"
    )
