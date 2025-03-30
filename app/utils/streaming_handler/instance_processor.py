import logging
from typing import Dict, Any, Optional, Tuple
from app.instance.instance_context import instance_manager
from app.utils.url_builder import build_instance_url

logger = logging.getLogger(__name__)

async def validate_and_prepare_payload(payload: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Validate payload and ensure required fields exist."""
    payload["stream"] = True
    if "model" not in payload and model_name:
        payload["model"] = model_name
    return payload

async def handle_instance_processing(
    instance_name: str,
    model_name: str,
    required_tokens: int,
    endpoint: str
) -> Tuple[Optional[Dict], Optional[str]]:
    """Validate and prepare an instance for handling the request."""
    config = instance_manager.get_instance_config(instance_name)
    state = instance_manager.get_instance_state(instance_name)
    
    if not config or not state:
        logger.warning(f"Missing config/state for {instance_name}")
        return None, None

    if required_tokens > config.max_input_tokens:
        logger.warning(f"Token limit exceeded for {instance_name}")
        return None, None

    if not instance_manager.check_rate_limit(instance_name, required_tokens):
        logger.warning(f"Rate limit exceeded for {instance_name}")
        return None, None

    instance = {**config.dict(), **state.dict(), "_current_model": model_name}
    url = build_instance_url(instance, endpoint)
    if not url:
        logger.warning(f"Failed to build URL for {instance_name}")
        return None, None

    return instance, url
