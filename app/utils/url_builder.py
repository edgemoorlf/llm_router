import logging

logger = logging.getLogger(__name__)

def build_instance_url(instance: dict, endpoint: str) -> str:
    """
    Build the appropriate URL for an instance based on its provider type.
    
    Args:
        instance: Instance configuration dictionary
        endpoint: The API endpoint (e.g., '/v1/chat/completions')
        
    Returns:
        Constructed URL for the instance
    """
    api_base = instance.get("api_base", "")
    provider_type = instance.get("provider_type", "")
    
    if provider_type == "azure":
        # Get Azure-specific configuration
        api_version = instance.get("api_version", "")
        model_deployments = instance.get("model_deployments", {})
        model_name = instance.get("_current_model", "").lower()  # Set by the caller
        
        # Get deployment name
        deployment = model_deployments.get(model_name, "")
        if not deployment:
            logger.warning(f"No deployment mapping found for model '{model_name}' in instance '{instance.get('name', '')}'")
            return None
            
        # Remove /v1 from the endpoint since Azure uses a different URL structure
        endpoint_without_v1 = endpoint.replace("/v1/", "/")
        if endpoint_without_v1.startswith("/"):
            endpoint_without_v1 = endpoint_without_v1[1:]
            
        # Build Azure URL
        url = f"{api_base}/openai/deployments/{deployment}/{endpoint_without_v1}"
        url = f"{url}?api-version={api_version}"
        
    else:
        # Generic OpenAI URL format
        url = f"{api_base}{endpoint}"
        
    return url 