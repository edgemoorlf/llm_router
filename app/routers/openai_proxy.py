"""Router for proxying OpenAI API requests to Azure OpenAI."""
import json
import logging
import time
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse

from app.services.azure_openai import azure_openai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])

@router.get("/instances/status")
async def get_instances_status() -> JSONResponse:
    """
    Get the status of all Azure OpenAI instances.
    
    Returns detailed information about each instance, including:
    - Health status (healthy, rate limited, error)
    - Current TPM usage
    - TPM limits
    - Priority and weight settings
    - Error information if applicable
    """
    try:
        instances = await azure_openai_service.get_instances_status()
        return JSONResponse(
            content={
                "status": "success",
                "timestamp": int(time.time()),
                "instances": instances,
                "total_instances": len(instances),
                "healthy_instances": len([i for i in instances if i["status"] == "healthy"]),
                "routing_strategy": os.getenv("AZURE_ROUTING_STRATEGY", "failover")
            }
        )
    except Exception as e:
        logger.exception("Error getting instances status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting instances status: {str(e)}",
        )

@router.post("/chat/completions")
async def chat_completions(request: Request) -> Any:
    """
    Proxy for OpenAI /v1/chat/completions endpoint.
    
    Forwards chat completion requests to Azure OpenAI with appropriate transformations.
    Supports both streaming and non-streaming responses.
    """
    try:
        # Parse the request body
        payload = await request.json()
        
        # Check if streaming is requested
        stream = payload.get("stream", False)
        
        # Transform request for Azure OpenAI
        transformed = await azure_openai_service.transform_request("/v1/chat/completions", payload)
        
        if stream:
            # For streaming, we need to process the response as a stream
            return await handle_streaming_request(
                "/v1/chat/completions",
                transformed["azure_deployment"],
                transformed["payload"]
            )
        else:
            # For regular requests, we can forward and return directly
            response = await azure_openai_service.forward_request(
                "/v1/chat/completions",
                transformed["azure_deployment"],
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
                
            # Ensure each choice has a message with role and content for non-streaming
            for choice in response["choices"]:
                if "message" not in choice:
                    # If the Azure API returns a different format, attempt to fix it
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
    
    Forwards completion requests to Azure OpenAI with appropriate transformations.
    Supports both streaming and non-streaming responses.
    """
    try:
        # Parse the request body
        payload = await request.json()
        
        # Check if streaming is requested
        stream = payload.get("stream", False)
        
        # Transform request for Azure OpenAI
        transformed = await azure_openai_service.transform_request("/v1/completions", payload)
        
        if stream:
            # For streaming, we need to process the response as a stream
            return await handle_streaming_request(
                "/v1/completions",
                transformed["azure_deployment"],
                transformed["payload"]
            )
        else:
            # For regular requests, we can forward and return directly
            response = await azure_openai_service.forward_request(
                "/v1/completions",
                transformed["azure_deployment"],
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
    
    Forwards embedding requests to Azure OpenAI with appropriate transformations.
    """
    try:
        # Parse the request body
        payload = await request.json()
        
        # Transform request for Azure OpenAI
        transformed = await azure_openai_service.transform_request("/v1/embeddings", payload)
        
        # Forward request to Azure OpenAI
        response = await azure_openai_service.forward_request(
            "/v1/embeddings",
            transformed["azure_deployment"],
            transformed["payload"]
        )
        
        # Ensure the response matches OpenAI's expected format
        # Check if response has the expected structure
        if "data" not in response or not isinstance(response["data"], list):
            logger.error(f"Unexpected Azure API response format for embeddings: {response}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The Azure API response does not contain expected 'data' field",
            )
            
        # Ensure each embedding item has the required fields
        for item in response["data"]:
            if "embedding" not in item:
                logger.error(f"Missing embedding in response data item: {item}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="The Azure API response contains invalid embedding data",
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
    
    Attempts to map the request to the appropriate Azure OpenAI endpoint.
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
            # If there's a model parameter, we can try to transform and forward
            transformed = await azure_openai_service.transform_request(full_path, payload)
            
            # Forward the request to Azure OpenAI
            response = await azure_openai_service.forward_request(
                full_path,
                transformed["azure_deployment"],
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

async def handle_streaming_request(endpoint: str, deployment: str, payload: Dict[str, Any]) -> StreamingResponse:
    """
    Handle streaming requests by forwarding them to Azure OpenAI and streaming the response back.
    Supports multiple Azure OpenAI instances with automatic failover.
    
    Args:
        endpoint: The API endpoint
        deployment: The Azure deployment name
        payload: The request payload
        
    Returns:
        A streaming response
    """
    import httpx
    import json
    import asyncio
    from starlette.background import BackgroundTask
    from app.utils.instance_manager import instance_manager
    
    # Ensure streaming is enabled
    payload["stream"] = True
    
    # Save the original model for adding to response chunks
    original_model = payload.get("model", "unknown")
    
    # Estimate tokens for rate limiting and instance selection
    required_tokens = 0
    if endpoint == "/v1/chat/completions":
        required_tokens = azure_openai_service._estimate_chat_tokens(payload)
    elif endpoint == "/v1/completions":
        required_tokens = azure_openai_service._estimate_completion_tokens(payload)
    else:
        required_tokens = 500  # Default for unknown endpoints
    
    # Select an instance using our instance manager
    instance = instance_manager.select_instance(required_tokens)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No available Azure OpenAI instances with capacity for streaming request",
        )
    
    # Initialize the instance if needed
    instance.initialize_client()
    
    # Build the Azure URL for this instance
    url = instance.build_azure_url(endpoint, deployment)
    
    logger.debug(f"Streaming request via instance {instance.name} to {url}")
    
    # Create a new client for streaming (we can't reuse the existing client)
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    client.headers.update({"api-key": instance.api_key})
    
    # Make the request to Azure OpenAI
    response = await client.post(url, json=payload, headers={"Accept": "application/json"})
    
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
