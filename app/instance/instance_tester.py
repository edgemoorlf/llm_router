"""API 实例测试功能"""
from typing import Dict, Any
import time
import logging
import httpx
import traceback

from app.instance.manager import instance_manager

logger = logging.getLogger(__name__)

async def perform_instance_test(instance_name: str) -> dict:
    """
    Test an instance and return the result.
    
    Args:
        instance_name: Name of the instance to test
        
    Returns:
        Dictionary with test results
    """
    if not instance_name:
        return {"success": False, "error": "Empty instance name"}
        
    try:
        # Get the instance from the manager using the encapsulated method
        instance = instance_manager.get_instance(instance_name)
        if not instance:
            return {"success": False, "error": f"Instance {instance_name} not found"}
            
        # Prepare a lightweight test request with minimal tokens
        logger.info(f"Testing instance {instance_name} with a lightweight completion request")
        
        # Create a minimal chat completion request payload
        try:
            # Use the first supported model, or a common fallback
            model = None
            if instance.supported_models:
                model = instance.supported_models[0]
            else:
                # Common fallbacks by provider
                if instance.provider_type == "azure":
                    model = "gpt-4o-mini"
                else:
                    model = "gpt-3.5-turbo"
                    
            payload = {
                "messages": [
                    {"role": "user", "content": "Why is the sky blue? Answer in one sentence."}
                ],
                "max_tokens": 20,      # Minimal response size
                "temperature": 0,      # Deterministic for testing
                "model": model
            }
        except Exception as e:
            return {"success": False, "error": f"Error creating payload: {str(e)}"}
        
        # Get deployment name if needed for Azure
        deployment = None
        try:
            if instance.provider_type == "azure" and payload.get("model") in instance.model_deployments:
                deployment = instance.model_deployments[payload["model"]]
            else:
                deployment = payload["model"]
        except Exception as e:
            return {"success": False, "error": f"Error getting deployment: {str(e)}"}
        
        try:
            # Construct URL based on provider type
            if instance.provider_type == "azure":
                # For Azure OpenAI
                endpoint = "/chat/completions"
                url = instance.build_url(endpoint, deployment)
                headers = {"api-key": instance.api_key, "Content-Type": "application/json"}
            else:
                # For generic providers
                url = f"{instance.api_base}/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {instance.api_key}",
                    "Content-Type": "application/json"
                }
                
            # Send test request
            logger.debug(f"Sending test request to {url}")
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
            response_time = round((time.time() - start_time) * 1000)  # in ms
            
            # Process response
            if response.status_code == 200:
                logger.info(f"Test request to {instance_name} successful ({response_time}ms)")
                
                # Update instance state
                instance.last_used = time.time()
                instance.mark_healthy()
                
                # Return success result
                return {
                    "success": True,
                    "status_code": 200,
                    "response_time_ms": response_time,
                    "timestamp": int(time.time())
                }
            else:
                logger.warning(f"Test request to {instance_name} failed with status {response.status_code}")
                
                # Update instance state based on response
                if response.status_code == 429:
                    instance.mark_rate_limited()
                    status = "rate_limited"
                else:
                    error_message = f"Health check failed: {response.status_code}"
                    try:
                        error_content = response.json()
                        error_message += f" - {str(error_content)[:100]}"
                    except:
                        if response.text:
                            error_message += f" - {response.text[:100]}"
                    instance.mark_error(error_message)
                    status = "error"
                
                # Return error result
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "error": f"Request failed with status {response.status_code}",
                    "status": status,
                    "timestamp": int(time.time())
                }
        except Exception as e:
            logger.error(f"Error making test request to {instance_name}: {str(e)}")
            
            # Mark the instance as having an error
            try:
                instance.mark_error(f"Test request error: {str(e)}")
            except:
                pass
                
            return {
                "success": False,
                "error": f"Request error: {str(e)}",
                "status": "error",
                "timestamp": int(time.time())
            }
    except Exception as e:
        logger.error(f"Error in perform_instance_test for {instance_name}: {str(e)}")
        return {
            "success": False, 
            "error": str(e),
            "timestamp": int(time.time())
        }
