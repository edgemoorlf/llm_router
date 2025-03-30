import httpx
from typing import Dict, Any

def create_http_client(instance: Dict[str, Any]) -> httpx.AsyncClient:
    """Create and configure an HTTP client for the instance."""
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),
        proxies={"http://": instance.get("proxy_url")}
    )
    
    if instance.get("provider_type") == "azure":
        client.headers.update({"api-key": instance["api_key"]})
    else:
        client.headers.update({"Authorization": f"Bearer {instance['api_key']}"})
    
    return client
