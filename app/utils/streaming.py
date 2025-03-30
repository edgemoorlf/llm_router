"""Utility functions for handling streaming requests to OpenAI-compatible services."""
import logging
import time
from typing import Dict, Any
from fastapi import HTTPException, status
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse

from app.services.error_handler import error_handler
from app.instance.instance_context import instance_manager
from app.services.instance_selector import instance_selector
from app.utils.streaming_handler.http_client import create_http_client
from app.utils.streaming_handler.stream_processor import (
    process_chat_stream,
    process_text_stream,
)
from app.utils.streaming_handler.instance_processor import (
    validate_and_prepare_payload,
    handle_instance_processing,
)


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
    model_name = original_model or payload.get("model", "unknown")
    payload = await validate_and_prepare_payload(payload, model_name)
    required_tokens = payload.pop("required_tokens", 64)
    request_id = f"streaming-{time.time()}"
    
    instances = instance_selector.select_instances_for_model(
        model_name=model_name,
        required_tokens=required_tokens,
        provider_type=provider_type
    )
    

    logger.info(f"{request_id}] Selected {len(instances)} instances for streaming request: {','.join(i['name'] for i in instances)}")
    if not instances:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"[{request_id}] No instances available that support model '{model_name}'"
        )

    tried_instances = set()
    last_error = None
    
    for instance_name in [i.get("name") for i in instances]:
        if instance_name in tried_instances:
            continue
        logger.info(f"{request_id}] trying {instance_name} ...")            
        tried_instances.add(instance_name)
        
        try:
            instance, url = await handle_instance_processing(
                instance_name,
                model_name,
                required_tokens,
                endpoint
            )
            if not instance or not url:
                continue

            client = create_http_client(instance)
            response = await client.post(url, json=payload, headers={"Accept": "application/json"})
            if not response.is_success:
                raise HTTPException(status_code=response.status_code)
                        
            instance_manager.mark_healthy(instance_name)
            
            processor = process_chat_stream if endpoint == "/v1/chat/completions" else process_text_stream
            logger.info(f"{request_id}] ... {instance_name} done")   
            return StreamingResponse(
                processor(response, instance_name, model_name, required_tokens),
                media_type="text/event-stream",
                background=BackgroundTask(client.aclose)
            )
            
        except HTTPException as e:
            error_handler.handle_special_error(e, instance_name)
            error_handler.handle_instance_error(instance, e)
            last_error = e
            logger.warning(f"[{request_id}] Instance {instance_name} failed with HTTP error: {str(e)}")
            continue
    
    if last_error:
        logger.error(f"[{request_id}] All instances failed. Last error: {last_error.detail}")
        
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="No available instances for streaming request"
    )
