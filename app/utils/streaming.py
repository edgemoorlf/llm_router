"""Utility functions for handling streaming requests to OpenAI-compatible services."""
import json
import logging
import asyncio
import re
import time
import httpx
from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Response
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse

from app.instance.instance_context import instance_manager
from app.instance.api_instance import InstanceStatus
from app.utils.token_estimator import estimate_chat_tokens, estimate_completion_tokens
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
    payload.pop("required_tokens", None)
    
    # Save the original model for adding to response chunks
    model_name = original_model or payload.get("model", "unknown")
    
    # Ensure model_name is in the payload for Azure deployment mapping
    if "model" not in payload and model_name:
        payload["model"] = model_name
    
    # Get max input tokens from instances
    max_input_tokens = next((inst.get("max_input_tokens", 0) for inst in instance_manager.get_all_instances().values()
                           if inst.get("max_input_tokens", 0) > 0), None)

    if max_input_tokens:
        # Estimate tokens in the request
        if endpoint == "/v1/chat/completions":
            token_count = estimate_chat_tokens(
                payload.get("messages", []),
                payload.get("functions", None),
                original_model or payload.get("model", ""),
                provider=provider_type
            )
            logger.debug(f"token count: {token_count}")
            
            if token_count > max_input_tokens:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input tokens ({token_count}) exceed maximum allowed ({max_input_tokens})"
                )

    # Filter out instances in error state
    available_instances = [instance for instance in instance_manager.get_all_instances().values() 
                         if instance.get("status", "") != "error"]
    
    if not available_instances:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No available API instances for streaming request",
        )

    # Sort instances by priority
    available_instances.sort(key=lambda x: x.get("priority", 100))
    
    # Try each instance until one succeeds
    last_error = None
    for instance in available_instances:
        try:
            # Initialize the instance if needed
            instance_name = instance.get("name", "")
            logger.debug(f"Trying to stream using instance: {instance_name}")
            
            # For each instance, make sure the model name is in the payload for deployment mapping
            instance_payload = payload.copy()
            
            # Determine deployment name based on the model (for Azure instances)
            deployment = ""
            curr_model_name = instance_payload.get("model", "").lower()
            if curr_model_name and instance.get("provider_type", "") == "azure" and instance.get("model_deployments", {}):
                # Look up the deployment name for this model in this specific instance
                model_deployments = instance.get("model_deployments", {})
                deployment = model_deployments.get(curr_model_name, "")
                if deployment:
                    logger.debug(f"Resolved deployment '{deployment}' for model '{curr_model_name}' in instance '{instance_name}'")
                else:
                    logger.warning(f"No deployment mapping found for model '{curr_model_name}' in instance '{instance_name}'")
                    
                    # Skip this instance if it doesn't have a deployment mapping for this model
                    if instance.get("provider_type", "") == "azure":
                        logger.warning(f"Skipping Azure instance {instance_name} that lacks deployment mapping for model {curr_model_name}")
                        continue
            
            # Add current model to instance for URL building
            instance = instance.copy()  # Make a copy to avoid modifying the original
            instance["_current_model"] = curr_model_name
            
            # Build the URL using shared logic
            url = build_instance_url(instance, endpoint)
            if not url:
                continue  # Skip this instance if URL building failed
            
            logger.debug(f"Streaming request via instance {instance_name} to {url}")
            
            # Create a new client for streaming (we can't reuse the existing client)
            client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
            
            # Set appropriate headers based on provider type
            api_key = instance.get("api_key", "")
            if instance.get("provider_type", "") == "azure":
                client.headers.update({"api-key": api_key})
            else:
                client.headers.update({"Authorization": f"Bearer {api_key}"})
            
            # Make the request to the API
            response = await client.post(url, json=instance_payload, headers={"Accept": "application/json"})
            
            # Check for HTTP errors
            response.raise_for_status()
    
            # Mark the instance as healthy
            instance_manager.update_instance_state(instance_name, status="healthy")
            
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
                                chunk["model"] = model_name
                                
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
                # For other endpoints like completions, we still need to ensure SSE format
                async def process_stream():
                    async for line in response.aiter_lines():
                        if not line or line.strip() == "":
                            continue
                            
                        # If the line already starts with data:, just yield it
                        if line.startswith("data:"):
                            yield f"{line}\n\n"
                            continue
                            
                        # Handle the "[DONE]" message
                        if line == "[DONE]":
                            yield f"data: [DONE]\n\n"
                            continue
                            
                        try:
                            # Try to parse as JSON
                            chunk = json.loads(line)
                            
                            # Ensure all required fields exist
                            if "choices" not in chunk:
                                chunk["choices"] = []
                                
                            # Add model if it's missing
                            if "model" not in chunk:
                                chunk["model"] = model_name
                                
                            # Ensure correct object type for completions
                            if "object" not in chunk:
                                chunk["object"] = "text_completion.chunk"
                                
                            # Format as SSE
                            yield f"data: {json.dumps(chunk)}\n\n"
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding streaming response JSON: {str(e)} - Line: {line}")
                            # Pass through as SSE anyway
                            yield f"data: {line}\n\n"
                        except Exception as e:
                            logger.error(f"Error processing streaming chunk: {str(e)}")
                
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
            
            # Handle rate limiting
            if status_code == 429:
                retry_after = int(e.response.headers.get("retry-after", "60"))
                instance_manager.update_instance_state(instance_name, status="rate_limited", retry_after=retry_after)
                logger.warning(f"Instance {instance_name} rate limited, retrying with next instance")
                last_error = f"Rate limit exceeded: {error_detail}"
                continue
            
            # Handle other errors
            instance_manager.update_instance_state(instance_name, status="error", error_detail=error_detail)
            logger.error(f"Instance {instance_name} returned error {status_code}: {error_detail}")
            last_error = f"API error: {error_detail}"
            continue
            
        except (httpx.RequestError, httpx.TimeoutException) as e:
            # Handle connection errors
            instance_manager.update_instance_state(instance_name, status="error", error_detail=str(e))
            logger.error(f"Connection error to instance {instance_name}: {str(e)}")
            last_error = f"Connection error: {str(e)}"
            continue
            
        except Exception as e:
            # Handle unexpected errors
            instance_manager.update_instance_state(instance_name, status="error", error_detail=str(e))
            logger.exception(f"Unexpected error with instance {instance_name}")
            last_error = f"Unexpected error: {str(e)}"
            continue
    
    # If we get here, all instances failed
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"All API instances failed for streaming request. Last error: {last_error}",
    )
