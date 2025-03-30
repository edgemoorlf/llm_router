import json
import logging
from typing import AsyncGenerator, Dict, Any
from app.instance.instance_context import instance_manager

logger = logging.getLogger(__name__)

async def process_chat_stream(response, instance_name: str, model_name: str, required_tokens: int) -> AsyncGenerator[str, None]:
    """Process chat completion stream response."""
    try:
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            
            line = line[5:].strip() if line.startswith("data:") else line
            if line == "[DONE]":
                yield "data: [DONE]\n\n"
                continue

            try:
                chunk = json.loads(line)
                chunk.setdefault("choices", [])
                
                for choice in chunk["choices"]:
                    if "text" in choice:
                        choice["delta"] = {"content": choice.pop("text")}
                    elif isinstance(choice.get("delta"), str):
                        choice["delta"] = {"content": choice["delta"]}
                
                chunk.setdefault("model", model_name)
                chunk.setdefault("object", "chat.completion.chunk")
                yield f"data: {json.dumps(chunk)}\n\n"
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e} - Line: {line}")
                yield f"data: {line}\n\n"
    finally:
        update_instance_metrics(instance_name, required_tokens)

async def process_text_stream(response, instance_name: str, model_name: str, required_tokens: int) -> AsyncGenerator[str, None]:
    """Process text completion stream response."""
    try:
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            
            if line.startswith("data:"):
                yield f"{line}\n\n"
            elif line == "[DONE]":
                yield "data: [DONE]\n\n"
            else:
                try:
                    chunk = json.loads(line)
                    chunk.setdefault("choices", [])
                    chunk.setdefault("model", model_name)
                    chunk.setdefault("object", "text_completion.chunk")
                    yield f"data: {json.dumps(chunk)}\n\n"
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e} - Line: {line}")
                    yield f"data: {line}\n\n"
    finally:
        update_instance_metrics(instance_name, required_tokens)

def update_instance_metrics(instance_name: str, tokens: int):
    """Update instance metrics after stream processing."""
    try:
        instance_manager.record_request(instance_name, success=True)
        instance_manager.update_token_usage(instance_name, tokens)
    except Exception as e:
        logger.error(f"Error updating metrics for {instance_name}: {str(e)}")
