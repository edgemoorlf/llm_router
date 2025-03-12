import logging
import asyncio
import httpx
from typing import Dict, Any, List, Optional, Tuple, Union
from fastapi import HTTPException, status
import traceback

from app.config import InstanceConfig
from app.instance.manager import instance_manager
from app.instance.api_instance import APIInstance
from app.config import config_loader

logger = logging.getLogger(__name__)

class InstanceService:
    """Service for managing API instances at runtime."""
    
    async def add_instance(self, instance_config: InstanceConfig) -> Dict[str, Any]:
        """
        Add a new API instance at runtime.
        
        Args:
            instance_config: Instance configuration
            
        Returns:
            Dictionary with status and details
        """
        # Check if instance with this name already exists
        if instance_config.name in instance_manager.instances:
            return {
                "status": "error",
                "message": f"Instance with name '{instance_config.name}' already exists"
            }
        
        # Basic validation
        if not instance_config.api_key or not instance_config.api_base:
            return {
                "status": "error", 
                "message": "API key and API base URL are required"
            }
        
        try:
            # Create the API instance
            instance = APIInstance(
                name=instance_config.name,
                provider_type=instance_config.provider_type,
                api_key=instance_config.api_key,
                api_base=instance_config.api_base,
                api_version=instance_config.api_version,
                proxy_url=instance_config.proxy_url,
                priority=instance_config.priority,
                weight=instance_config.weight,
                max_tpm=instance_config.max_tpm,
                max_input_tokens=instance_config.max_input_tokens,
                supported_models=instance_config.supported_models,
                model_deployments=instance_config.model_deployments
            )
            
            # Initialize the client for this instance
            instance.initialize_client()
            
            # Add the instance to the manager
            with instance_manager.instances_lock:
                instance_manager.instances[instance_config.name] = instance
                # The _save_state_to_file method will be called by the add_instance method
                instance_manager._save_state_to_file()
            
            # Set initial RPM window
            config = config_loader.get_config()
            if config and config.monitoring and config.monitoring.stats_window_minutes > 0:
                instance.set_rpm_window(config.monitoring.stats_window_minutes)
            
            # Convert InstanceConfig to dict with Pydantic v1/v2 compatibility
            instance_dict = {}
            try:
                # Try Pydantic v2 approach first
                if hasattr(instance_config, 'model_dump'):
                    instance_dict = instance_config.model_dump()
                # Fall back to Pydantic v1
                else:
                    instance_dict = instance_config.dict()
                
                # Redact API key for security
                if "api_key" in instance_dict:
                    instance_dict["api_key"] = "********"

            except Exception as e:
                logger.warning(f"Error converting instance config to dict: {str(e)}")
                # Fallback to manual dictionary creation
                instance_dict = {
                    "name": instance_config.name,
                    "provider_type": instance_config.provider_type,
                    "api_base": instance_config.api_base,
                    "api_version": instance_config.api_version,
                    "priority": instance_config.priority,
                    "weight": instance_config.weight,
                    "max_tpm": instance_config.max_tpm,
                    "max_input_tokens": instance_config.max_input_tokens,
                    "supported_models": instance_config.supported_models,
                    "model_deployments": instance_config.model_deployments,
                    "api_key": "********"  # Redacted
                }
            
            return {
                "status": "success",
                "message": f"Added instance {instance_config.name}",
                "instance": instance_dict
            }
        except Exception as e:
            logger.error(f"Error adding instance {instance_config.name}: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"Error adding instance: {str(e)}"
            }
    
    async def add_many_instances(self, instances: List[InstanceConfig]) -> Dict[str, Any]:
        """
        Add multiple API instances at runtime.
        
        Args:
            instances: List of instance configurations
            
        Returns:
            Dictionary with status and details
        """
        if not instances:
            return {
                "status": "error",
                "message": "No instances provided"
            }
        
        added = []
        skipped = []
        failed = []
        
        for instance_config in instances:
            try:
                # Check if instance with this name already exists
                if instance_config.name in instance_manager.instances:
                    skipped.append({
                        "name": instance_config.name,
                        "reason": "Instance with this name already exists"
                    })
                    continue
                
                # Basic validation
                if not instance_config.api_key or not instance_config.api_base:
                    failed.append({
                        "name": instance_config.name,
                        "reason": "API key and API base URL are required"
                    })
                    continue
                
                # Create the API instance
                instance = APIInstance(
                    name=instance_config.name,
                    provider_type=instance_config.provider_type,
                    api_key=instance_config.api_key,
                    api_base=instance_config.api_base,
                    api_version=instance_config.api_version,
                    proxy_url=instance_config.proxy_url,
                    priority=instance_config.priority,
                    weight=instance_config.weight,
                    max_tpm=instance_config.max_tpm,
                    max_input_tokens=instance_config.max_input_tokens,
                    supported_models=instance_config.supported_models,
                    model_deployments=instance_config.model_deployments
                )
                
                # Initialize the client for this instance
                instance.initialize_client()
                
                # Add the instance to the manager
                with instance_manager.instances_lock:
                    instance_manager.instances[instance_config.name] = instance
                
                # Set initial RPM window
                config = config_loader.get_config()
                if config and config.monitoring and config.monitoring.stats_window_minutes > 0:
                    instance.set_rpm_window(config.monitoring.stats_window_minutes)
                
                added.append({
                    "name": instance_config.name,
                    "provider_type": instance_config.provider_type,
                    "api_base": instance_config.api_base
                })
            except Exception as e:
                failed.append({
                    "name": instance_config.name,
                    "reason": f"Error: {str(e)}"
                })
        
        # Save state to file after adding all instances
        instance_manager._save_state_to_file()
        
        return {
            "status": "success" if added else "warning" if skipped and not added else "error",
            "message": f"Added {len(added)} instances, skipped {len(skipped)}, failed {len(failed)}",
            "added": added,
            "skipped": skipped,
            "failed": failed
        }
    
    async def test_instance(self, instance_config_or_name: Union[InstanceConfig, str]) -> Dict[str, Any]:
        """
        Test an instance configuration or existing instance by name.
        
        Args:
            instance_config_or_name: Either a full InstanceConfig object for a new instance,
                                    or a string name of an existing instance in instance_manager
            
        Returns:
            Status information about the test
        """
        # Check if we're testing an existing instance or a new config
        if isinstance(instance_config_or_name, str):
            # We're testing an existing instance by name
            instance_name = instance_config_or_name
            if instance_name not in instance_manager.instances:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Instance '{instance_name}' not found"
                )
            instance = instance_manager.instances[instance_name]
        else:
            # We're testing a new instance config
            instance_config = instance_config_or_name
            # Basic validation
            if not instance_config.api_key or not instance_config.api_base:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="API key and API base URL are required"
                )
            
            # Create a temporary API instance
            instance = APIInstance(
                name=instance_config.name,
                provider_type=instance_config.provider_type,
                api_key=instance_config.api_key,
                api_base=instance_config.api_base,
                api_version=instance_config.api_version,
                proxy_url=instance_config.proxy_url,
                priority=instance_config.priority,
                weight=instance_config.weight,
                max_tpm=instance_config.max_tpm,
                max_input_tokens=instance_config.max_input_tokens,
                supported_models=instance_config.supported_models,
                model_deployments=instance_config.model_deployments
            )
            
            # Initialize the client for this instance
            instance.initialize_client()
        
        # Test the connection - we'll use a simple GET request to the API base URL
        start_time = asyncio.get_event_loop().time()
        
        # Execute the test based on provider type
        if instance.provider_type == "azure":
            return await self._test_azure_instance(instance, start_time)
        else:
            return await self._test_generic_instance(instance, start_time)
    
    async def _test_azure_instance(self, instance: APIInstance, start_time: float) -> Dict[str, Any]:
        """Test an Azure OpenAI instance by making a request to list models."""
        try:
            # Use the instance's build_url method with the /openai/models endpoint
            # This endpoint doesn't require a deployment name, so we pass an empty string
            url = instance.build_url("/openai/models", "")
            headers = {"api-key": instance.api_key}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)
                
            response.raise_for_status()
            
            # Parse models if available
            models = []
            response_data = response.json()
            if "data" in response_data:
                models = [model.get("id", "") for model in response_data.get("data", [])]
            
            # Successful test
            end_time = asyncio.get_event_loop().time()
            return {
                "status": "success",
                "message": "Successfully connected to the instance API",
                "response_time_ms": int((end_time - start_time) * 1000),
                "provider_type": instance.provider_type,
                "api_base": instance.api_base,
                "models_available": len(models),
                "models": models[:10] if len(models) > 10 else models  # Limit to 10 models to avoid large responses
            }
        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e, start_time)
        except Exception as e:
            return self._handle_connection_error(e, start_time)
    
    async def _test_generic_instance(self, instance: APIInstance, start_time: float) -> Dict[str, Any]:
        """Test a generic OpenAI instance by making a request to list models."""
        try:
            # Use the instance's build_url method instead of directly constructing the URL
            url = instance.build_url("/v1/models", "")
            headers = {"Authorization": f"Bearer {instance.api_key}"}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)
                
            response.raise_for_status()
            
            # Parse models if available
            models = []
            response_data = response.json()
            if "data" in response_data:
                models = [model.get("id", "") for model in response_data.get("data", [])]
            
            # Successful test
            end_time = asyncio.get_event_loop().time()
            return {
                "status": "success",
                "message": "Successfully connected to the instance API",
                "response_time_ms": int((end_time - start_time) * 1000),
                "provider_type": instance.provider_type,
                "api_base": instance.api_base,
                "models": models,
                "model_count": len(models)
            }
        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e, start_time)
        except Exception as e:
            return self._handle_connection_error(e, start_time)
    
    def _handle_http_error(self, e: httpx.HTTPStatusError, start_time: float) -> Dict[str, Any]:
        """Handle HTTP errors from API requests during testing."""
        end_time = asyncio.get_event_loop().time()
        error_message = str(e)
        try:
            error_json = e.response.json()
            if "error" in error_json:
                error_message = error_json["error"].get("message", str(e))
        except:
            pass
        
        return {
            "status": "error",
            "message": f"API error: {error_message}",
            "response_time_ms": int((end_time - start_time) * 1000),
            "error_code": e.response.status_code
        }
    
    def _handle_connection_error(self, e: Exception, start_time: float) -> Dict[str, Any]:
        """Handle connection errors during testing."""
        end_time = asyncio.get_event_loop().time()
        return {
            "status": "error",
            "message": f"Connection error: {str(e)}",
            "response_time_ms": int((end_time - start_time) * 1000),
        }
    
    async def remove_instance(self, instance_name: str) -> Dict[str, Any]:
        """
        Remove an instance from the proxy service at runtime.
        
        Args:
            instance_name: The name of the instance to remove
            
        Returns:
            Status information about the removal
        """
        # Check if instance exists
        if instance_name not in instance_manager.instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance '{instance_name}' not found"
            )
        
        # Remove the instance
        with instance_manager.instances_lock:
            # Get instance details for the response
            instance = instance_manager.instances[instance_name]
            instance_details = {
                "name": instance.name,
                "provider_type": instance.provider_type,
                "api_base": instance.api_base
            }
            
            # Remove the instance
            del instance_manager.instances[instance_name]
        
        logger.info(f"Removed instance '{instance_name}' at runtime")
        
        return {
            "status": "success",
            "message": f"Instance '{instance_name}' removed successfully",
            "instance": instance_details
        }

# Create a singleton instance
instance_service = InstanceService() 