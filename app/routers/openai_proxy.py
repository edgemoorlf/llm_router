"""Router for proxying OpenAI API requests to OpenAI-compatible services."""
import json
import logging
from typing import Any, Tuple, Dict, Optional
import uuid
import time
import traceback

from fastapi import APIRouter, Depends, HTTPException, Request, status, Body, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.instance.manager import instance_manager
from app.services.azure_openai import azure_openai_service
from app.services.generic_openai import generic_openai_service
from app.utils.streaming import handle_streaming_request
from app.errors.utils import handle_router_errors
from app.instance.service_stats import service_stats

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
        instance_manager.get_all_instances(), 
        model_name,
        provider_type="azure"
    )
    
    generic_instances = instance_manager.router.get_available_instances_for_model(
        instance_manager.get_all_instances(), 
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

@router.post("/v1/chat/completions")
@handle_router_errors("processing chat completions request")
async def chat_completion(
    request: Request, 
    body: Dict[str, Any] = Body(...),  # Use Body for accessing raw request body
    x_ms_client_id: Optional[str] = Header(None),  # Optional Azure client ID
    x_ms_trace_id: Optional[str] = Header(None),   # Optional Azure trace ID
) -> Dict[str, Any]:
    """
    Handle chat completions API requests.
    
    This is a specialized endpoint for the chat completions format with message sequence.
    """
    request_id = str(uuid.uuid4())
    request_start_time = time.time()
    
    # Set a logger prefix for this request
    log_prefix = f"[{request_id}]"
    
    # Keep the original model name for metadata
    original_model = body.get("model", "")
    
    # Extract stream parameter
    stream = body.get("stream", False)
    
    # Determine the service to use based on the model
    service, provider_type = determine_service_by_model(original_model)
    
    # Save a copy of the original payload
    payload = {**body}
    
    # Transform the request for the appropriate backend
    logger.debug(f"{log_prefix} Transforming request for provider type: {provider_type}")
    transformed = await service.transform_request("/v1/chat/completions", payload)
    
    if stream:
        # For streaming, we need to process the response as a stream
        return await handle_streaming_request(
            "/v1/chat/completions",
            transformed["payload"],
            provider_type,
            original_model=transformed.get("original_model")
        )
    else:
        # For regular requests, we can forward and return directly
        response = await service.forward_request(
            "/v1/chat/completions",
            transformed["payload"],
            transformed.get("original_model")
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
            if "message" not in choice or "role" not in choice["message"] or "content" not in choice["message"]:
                # Azure OpenAI might not always include content in case of content filtering
                if "message" in choice and "role" in choice["message"] and "content" not in choice["message"]:
                    choice["message"]["content"] = ""
                    
        # Add the model info to the response if not present
        if "model" not in response:
            response["model"] = original_model or "Unknown"
            
        # Ensure finish_reason is always included
        for choice in response["choices"]:
            if "finish_reason" not in choice:
                choice["finish_reason"] = "stop"  # Set a default
                
        # Track request completion
        duration_ms = int((time.time() - request_start_time) * 1000)
        logger.info(f"{log_prefix} Request completed in {duration_ms}ms")
                
        return response

@router.post("/v1/completions")
@handle_router_errors("processing completions request")
async def completion(
    request: Request, 
    body: Dict[str, Any] = Body(...),
    x_ms_client_id: Optional[str] = Header(None),  # Optional Azure client ID
    x_ms_trace_id: Optional[str] = Header(None),   # Optional Azure trace ID
) -> Dict[str, Any]:
    """
    Handle standard completions API requests.
    
    This is the legacy non-chat API for text completion.
    """
    request_id = str(uuid.uuid4())
    request_start_time = time.time()
    
    # Set a logger prefix for this request
    log_prefix = f"[{request_id}]"
    
    # Keep the original model name for metadata
    original_model = body.get("model", "")
    
    # Extract stream parameter
    stream = body.get("stream", False)
    
    # Determine the service to use based on the model
    service, provider_type = determine_service_by_model(original_model)
    
    # Save a copy of the original payload
    payload = {**body}
    
    # Transform the request for the appropriate backend
    logger.debug(f"{log_prefix} Transforming request for provider type: {provider_type}")
    transformed = await service.transform_request("/v1/completions", payload)
    
    if stream:
        # For streaming, we need to process the response as a stream
        return await handle_streaming_request(
            "/v1/completions",
            transformed["payload"],
            provider_type,
            original_model=transformed.get("original_model")
        )
    else:
        # For regular requests, we can forward and return directly
        response = await service.forward_request(
            "/v1/completions",
            transformed["payload"],
            transformed.get("original_model")
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
            
        # Track request completion
        duration_ms = int((time.time() - request_start_time) * 1000)
        logger.info(f"{log_prefix} Request completed in {duration_ms}ms")
        
        return response

@router.post("/v1/embeddings")
@handle_router_errors("processing embeddings request")
async def embeddings(request: Request) -> Any:
    """
    Proxy for OpenAI /v1/embeddings endpoint.
    
    Forwards embedding requests to OpenAI-compatible services with appropriate transformations.
    """
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
        transformed["payload"],
        transformed.get("original_model")
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

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
@handle_router_errors("processing API request")
async def catch_all(request: Request, path: str) -> Any:
    """
    Catch-all route for handling any other OpenAI API endpoints.
    
    Attempts to map the request to the appropriate API endpoint.
    """
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
        
        # Extract stream parameter if it exists
        stream = payload.get("stream", False)
        
        # Determine which service to use based on the model name
        service, provider_type = determine_service_by_model(model_name)
        
        # If there's a model parameter, we can try to transform and forward
        transformed = await service.transform_request(full_path, payload)
        
        # Handle streaming requests differently
        if stream:
            # Process as a streaming request
            return await handle_streaming_request(
                full_path,
                transformed["payload"],
                provider_type,
                original_model=transformed.get("original_model")
            )
        else:
            # For regular requests, forward and return directly
            response = await service.forward_request(
                full_path,
                transformed["payload"],
                transformed.get("original_model"),
                method=request.method
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
