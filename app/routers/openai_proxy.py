"""Router for proxying OpenAI API requests to OpenAI-compatible services."""
import json
import logging
import time
import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse

from app.instance.manager import instance_manager
from app.services.azure_openai import azure_openai_service
from app.services.generic_openai import generic_openai_service
from app.utils.token_estimator import estimate_chat_tokens, estimate_completion_tokens

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])

def determine_service_by_model(model_name: str) -> Tuple[Any, str]:
    """
    Determine which service to use based on the model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        A tuple of (service, provider_type)
    """
    if not model_name:
        logger.debug("No model name provided, using Azure service by default")
        return azure_openai_service, "azure"
    
    # Normalize the model name
    model_name = model_name.lower()
    
    # Get available instances for each provider type
    azure_instances = instance_manager.router.get_available_instances_for_model(
        instance_manager.instances, 
        model_name,
        provider_type="azure"
    )
    
    generic_instances = instance_manager.router.get_available_instances_for_model(
        instance_manager.instances, 
        model_name,
        provider_type="generic"
    )
    
    # Log what we found for debugging
    if azure_instances:
        logger.debug(f"Found {len(azure_instances)} Azure instances supporting model '{model_name}'")
    
    if generic_instances:
        logger.debug(f"Found {len(generic_instances)} Generic instances supporting model '{model_name}'")
    
    # Decide which service to use based on available instances
    if generic_instances and not azure_instances:
        # Only generic instances support this model
        logger.debug(f"Using generic service for model '{model_name}' (only generic instances available)")
        return generic_openai_service, "generic"
    elif azure_instances and not generic_instances:
        # Only Azure instances support this model
        logger.debug(f"Using Azure service for model '{model_name}' (only Azure instances available)")
        return azure_openai_service, "azure"
    elif generic_instances and azure_instances:
        # Both types support this model, prefer Azure by default
        # This could be made configurable in the future
        logger.debug(f"Using Azure service for model '{model_name}' (both types available, preferring Azure)")
        return azure_openai_service, "azure"
    else:
        # No instances explicitly support this model, default to Azure
        logger.warning(f"No instances explicitly support model '{model_name}', defaulting to Azure service")
        return azure_openai_service, "azure"

@router.post("/chat/completions")
async def chat_completions(request: Request) -> Any:
    """
    Proxy for OpenAI /v1/chat/completions endpoint.
    
    Forwards chat completion requests to OpenAI-compatible services with appropriate transformations.
    Supports both streaming and non-streaming responses.
    """
    try:
        # Parse the request body
        payload = await request.json()
        
        # Check if streaming is requested
        stream = payload.get("stream", False)
        
        # Get the model name from the payload
        model_name = payload.get("model")
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model name is required",
            )
        
        # Determine which service to use based on the model name
        service, provider_type = determine_service_by_model(model_name)
        
        # Transform request for the appropriate service
        transformed = await service.transform_request("/v1/chat/completions", payload)
        
        if stream:
            # For streaming, we need to process the response as a stream
            return await handle_streaming_request(
                "/v1/chat/completions",
                transformed["azure_deployment" if provider_type == "azure" else "deployment"],
                transformed["payload"],
                provider_type
            )
        else:
            # For regular requests, we can forward and return directly
            response = await service.forward_request(
                "/v1/chat/completions",
                transformed["azure_deployment" if provider_type == "azure" else "deployment"],
                transformed["payload"]
            )
            
            # Ensure the response matches OpenAI's expected format
            # Check if response has the expected structure
            if "choices" not in response or not isinstance(response["choices"], list) or not response["choices"]:
                logger.error(f"Unexpected API response format: {response}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="The API response does not contain expected 'choices' field",
                )
                
            # Ensure each choice has a message with role and content for non-streaming
            for choice in response["choices"]:
                if "message" not in choice:
                    # If the API returns a different format, attempt to fix it
                    if "text" in choice:
                        # Some versions might return text directly
                        choice["message"] = {
                            "role": "assistant",
                            "content": choice["text"]
                        }
                        
            # Add any missing standard fields
            if "model" not in response:
                response["model"] = payload.get("model", "unknown")
                
            # Ensure correct object type
            if "object" not in response:
                response["object"] = "chat.completion"
                
            return response
    
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.exception("Error processing chat completions request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )

@router.post("/completions")
async def completions(request: Request) -> Any:
    """
    Proxy for OpenAI /v1/completions endpoint.
    
    Forwards completion requests to OpenAI-compatible services with appropriate transformations.
    Supports both streaming and non-streaming responses.
    """
    try:
        # Parse the request body
        payload = await request.json()
        
        # Check if streaming is requested
        stream = payload.get("stream", False)
        
        # Get the model name from the payload
        model_name = payload.get("model")
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model name is required",
            )
        
        # Determine which service to use based on the model name
        service, provider_type = determine_service_by_model(model_name)
        
        # Transform request for the appropriate service
        transformed = await service.transform_request("/v1/completions", payload)
        
        if stream:
            # For streaming, we need to process the response as a stream
            return await handle_streaming_request(
                "/v1/completions",
                transformed["azure_deployment" if provider_type == "azure" else "deployment"],
                transformed["payload"],
                provider_type
            )
        else:
            # For regular requests, we can forward and return directly
            response = await service.forward_request(
                "/v1/completions",
                transformed["azure_deployment" if provider_type == "azure" else "deployment"],
                transformed["payload"]
            )
            
            # Ensure the response matches OpenAI's expected format
            # Check if response has the expected structure
            if "choices" not in response or not isinstance(response["choices"], list) or not response["choices"]:
                logger.error(f"Unexpected Azure API response format: {response}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="The Azure API response does not contain expected 'choices' field",
                )
                
            # Ensure each choice has the required fields
            for choice in response["choices"]:
                if "text" not in choice:
                    # Some Azure versions might return differently
                    if "message" in choice and "content" in choice["message"]:
                        choice["text"] = choice["message"]["content"]
                    elif "content" in choice:
                        choice["text"] = choice["content"]
                        
            # Add any missing standard fields  
            if "model" not in response:
                response["model"] = payload.get("model", "unknown")
                
            # Ensure correct object type
            if "object" not in response:
                response["object"] = "text_completion"
                
            return response
    
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.exception("Error processing completions request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )

@router.post("/embeddings")
async def embeddings(request: Request) -> Any:
    """
    Proxy for OpenAI /v1/embeddings endpoint.
    
    Forwards embedding requests to OpenAI-compatible services with appropriate transformations.
    """
    try:
        # Parse the request body
        payload = await request.json()
        
        # Get the model name from the payload
        model_name = payload.get("model")
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model name is required",
            )
        
        # Determine which service to use based on the model name
        service, provider_type = determine_service_by_model(model_name)
        
        # Transform request for the appropriate service
        transformed = await service.transform_request("/v1/embeddings", payload)
        
        # Forward request to the API
        response = await service.forward_request(
            "/v1/embeddings",
            transformed["azure_deployment" if provider_type == "azure" else "deployment"],
            transformed["payload"]
        )
        
        # Ensure the response matches OpenAI's expected format
        # Check if response has the expected structure
        if "data" not in response or not isinstance(response["data"], list):
            logger.error(f"Unexpected API response format for embeddings: {response}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The API response does not contain expected 'data' field",
            )
            
        # Ensure each embedding item has the required fields
        for item in response["data"]:
            if "embedding" not in item:
                logger.error(f"Missing embedding in response data item: {item}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="The API response contains invalid embedding data",
                )
        
        # Add any missing standard fields
        if "model" not in response:
            response["model"] = payload.get("model", "unknown")
            
        # Ensure correct object type
        if "object" not in response:
            response["object"] = "list"
            
        # Ensure usage information exists
        if "usage" not in response:
            # Calculate estimated usage based on input and output
            input_tokens = 0
            input_text = payload.get("input", "")
            
            # Handle both string and list inputs
            if isinstance(input_text, str):
                input_tokens = len(input_text.split()) * 2  # Rough estimate
            elif isinstance(input_text, list):
                for text in input_text:
                    if isinstance(text, str):
                        input_tokens += len(text.split()) * 2  # Rough estimate
            
            response["usage"] = {
                "prompt_tokens": input_tokens,
                "total_tokens": input_tokens
            }
            
        return response
    
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.exception("Error processing embeddings request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path: str) -> Any:
    """
    Catch-all route for handling any other OpenAI API endpoints.
    
    Attempts to map the request to the appropriate API endpoint.
    """
    try:
        # Get the full path
        full_path = f"/v1/{path}"
        
        # Parse the request body for non-GET requests
        if request.method != "GET":
            try:
                payload = await request.json()
            except json.JSONDecodeError:
                payload = {}
        else:
            # Convert query parameters to dict for GET requests
            payload = dict(request.query_params)
        
        # Check if we can handle this endpoint
        if "model" in payload:
            # Get the model name from the payload
            model_name = payload.get("model")
            
            # Determine which service to use based on the model name
            service, provider_type = determine_service_by_model(model_name)
            
            # If there's a model parameter, we can try to transform and forward
            transformed = await service.transform_request(full_path, payload)
            
            # Forward the request to the API
            response = await service.forward_request(
                full_path,
                transformed["azure_deployment" if provider_type == "azure" else "deployment"],
                transformed["payload"],
                method=request.method,
            )
            
            # Ensure the response includes the model information
            if "model" not in response:
                response["model"] = payload.get("model", "unknown")
                
            # Add some basic verification based on the endpoint pattern
            if "completions" in full_path:
                # Make sure we have choices
                if "choices" not in response:
                    response["choices"] = []
                    
                # Set appropriate object type
                if "object" not in response:
                    if "chat" in full_path:
                        response["object"] = "chat.completion"
                    else:
                        response["object"] = "text_completion"
            elif "embeddings" in full_path:
                # Make sure we have data array
                if "data" not in response:
                    response["data"] = []
                    
                # Set appropriate object type    
                if "object" not in response:
                    response["object"] = "list"
                    
            # Log the response for debugging
            logger.debug(f"Processed response for {full_path}")
            
            return response
        else:
            # We can't handle this request
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported endpoint or missing model parameter: {full_path}",
            )
    
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Error processing request to {full_path}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )

async def handle_streaming_request(endpoint: str, deployment: str, payload: Dict[str, Any], provider_type: str = "azure") -> StreamingResponse:
    """
    Handle streaming requests by forwarding them to the API and streaming the response back.
    Supports multiple API instances with automatic failover.
    
    Args:
        endpoint: The API endpoint
        deployment: The deployment name
        payload: The request payload
        provider_type: The provider type ("azure" or "generic")
        
    Returns:
        A streaming response
    """
    import httpx
    import json
    import asyncio
    from starlette.background import BackgroundTask
    from app.utils.instance_manager import instance_manager, InstanceStatus
    
    # Ensure streaming is enabled
    payload["stream"] = True
    
    # Save the original model for adding to response chunks
    original_model = payload.get("model", "unknown")
    
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
    
    # Select an instance based on the routing strategy and model support
    primary_instance = instance_manager.select_instance(required_tokens, model_name)
    logger.debug(f"Selected primary instance for streaming: {primary_instance.name if primary_instance else 'None'} for model: {model_name} max input tokens: {primary_instance.max_input_tokens}")
    
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
                url = instance.build_url(deployment, endpoint)
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
