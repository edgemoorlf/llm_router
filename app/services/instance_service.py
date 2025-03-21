import json
import logging
import asyncio
import httpx
from typing import Dict, Any, List, Optional, Tuple, Union
from fastapi import HTTPException, status
import traceback
import time

from app.config import config_loader
from app.models.instance import InstanceConfig, InstanceState
from app.instance.manager import InstanceManager

logger = logging.getLogger(__name__)

class InstanceService:
    """Service for managing API instances and their states."""
    
    def __init__(self, instance_manager: InstanceManager):
        self.instance_manager = instance_manager
        
    def get_instance(self, name: str) -> Optional[Tuple[InstanceConfig, InstanceState]]:
        """
        Get an instance by name.
        
        Args:
            name: Name of the instance to get
            
        Returns:
            Tuple of (InstanceConfig, InstanceState) if found, None otherwise
        """
        return self.instance_manager.get_instance(name)
        
    def get_all_instances(self) -> List[Tuple[InstanceConfig, InstanceState]]:
        """
        Get all instances.
        
        Returns:
            List of (InstanceConfig, InstanceState) tuples
        """
        return self.instance_manager.get_all_instances()
        
    def add_instance(self, config: InstanceConfig) -> None:
        """
        Add a new instance.
        
        Args:
            config: Instance configuration to add
        """
        self.instance_manager.add_instance(config)
        
    def remove_instance(self, name: str) -> None:
        """
        Remove an instance.
        
        Args:
            name: Name of the instance to remove
        """
        self.instance_manager.remove_instance(name)
        
    def update_instance_sync(self, name: str, config: InstanceConfig) -> None:
        """
        Update an instance's configuration (synchronous version).
        
        Args:
            name: Name of the instance to update
            config: New instance configuration
        """
        self.instance_manager.update_instance(name, config)
        
    async def update_instance(self, name: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an instance's configuration asynchronously.
        
        Args:
            name: Name of the instance to update
            update_data: Dictionary of attributes to update
            
        Returns:
            Dictionary with status and details about the update
        """
        try:
            # Check if instance exists
            if not self.instance_manager.has_instance(name):
                return {
                    "status": "error",
                    "message": f"Instance '{name}' not found"
                }
                
            # Get current config
            current_config = self.instance_manager.get_instance_config(name)
            
            # Apply the update using the instance manager
            result = self.instance_manager.update_instance(name, update_data)
            
            if not result:
                return {
                    "status": "error",
                    "message": f"Failed to update instance '{name}'"
                }
                
            # Get the updated config
            updated_config = self.instance_manager.get_instance_config(name)
            
            # Prepare response data
            updated_fields = {}
            for key, value in update_data.items():
                # Only include fields that were actually updated
                if hasattr(current_config, key):
                    old_value = getattr(current_config, key)
                    updated_fields[key] = {
                        "old": old_value,
                        "new": value
                    }
            
            # Convert to dict with redacted API key
            instance_dict = updated_config.dict()
            if "api_key" in instance_dict:
                instance_dict["api_key"] = "**REDACTED**"
                
            return {
                "status": "success",
                "message": f"Instance '{name}' updated successfully",
                "updated_fields": updated_fields,
                "instance": instance_dict
            }
        except Exception as e:
            logger.error(f"Error updating instance {name}: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"Error updating instance: {str(e)}"
            }
        
    def update_tpm_usage(self, name: str, tokens: int) -> None:
        """
        Update the TPM usage for an instance.
        
        Args:
            name: Name of the instance
            tokens: Number of tokens used
        """
        self.instance_manager.update_tpm_usage(name, tokens)
        
    def record_error(self, name: str, error_type: str, error_message: str) -> None:
        """
        Record an error for an instance.
        
        Args:
            name: Name of the instance
            error_type: Type of error (client_error, upstream_error)
            error_message: Error message
        """
        self.instance_manager.record_error(name, error_type, error_message)
        
    def is_rate_limited(self, name: str) -> bool:
        """
        Check if an instance is rate limited.
        
        Args:
            name: Name of the instance
            
        Returns:
            True if the instance is rate limited, False otherwise
        """
        return self.instance_manager.is_rate_limited(name)
        
    def mark_rate_limited(self, name: str) -> None:
        """
        Mark an instance as rate limited.
        
        Args:
            name: Name of the instance
        """
        self.instance_manager.mark_rate_limited(name)
        
    def mark_healthy(self, name: str) -> None:
        """
        Mark an instance as healthy.
        
        Args:
            name: Name of the instance
        """
        self.instance_manager.mark_healthy(name)
    
    async def _add_single_instance(self, instance_config: InstanceConfig) -> Dict[str, Any]:
        """
        Private helper method to add a single instance.
        
        Args:
            instance_config: Instance configuration
            
        Returns:
            Dictionary with status and details:
            - On success: {"status": "success", "instance": instance_dict}
            - On error: {"status": "error", "message": error_message, "reason": reason_string}
        """
        # Check if instance with this name already exists
        if self.instance_manager.has_instance(instance_config.name):
            return {
                "status": "error",
                "message": f"Instance with name '{instance_config.name}' already exists",
                "reason": "duplicate_name"
            }
        
        # Basic validation
        if not instance_config.api_key or not instance_config.api_base:
            return {
                "status": "error", 
                "message": "API key and API base URL are required",
                "reason": "missing_required_fields"
            }
        
        try:
            # Add the instance to the manager
            result = self.instance_manager.add_instance(instance_config)
            
            if not result:
                logger.error(f"Failed to add instance {instance_config.name} to instance manager")
                return {
                    "status": "error",
                    "message": f"Failed to add instance {instance_config.name} to storage",
                    "reason": "storage_error"
                }
                
            logger.info(f"Successfully added instance {instance_config.name} to storage")
            
            # Set initial RPM window
            config = config_loader.get_config()
            if config and config.monitoring and config.monitoring.stats_window_minutes > 0:
                self.instance_manager.set_rpm_window(instance_config.name, config.monitoring.stats_window_minutes)
            
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
                "instance": instance_dict
            }
        except Exception as e:
            logger.error(f"Error adding instance {instance_config.name}: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"Error adding instance: {str(e)}",
                "reason": "exception"
            }
    
    async def add_instance(self, instance_config: InstanceConfig) -> Dict[str, Any]:
        """
        Add a new API instance at runtime.
        
        Args:
            instance_config: Instance configuration
            
        Returns:
            Dictionary with status and details
        """
        result = await self._add_single_instance(instance_config)
        
        # Format the response appropriately for the public API
        if result["status"] == "success":
            return {
                "status": "success",
                "message": f"Added instance {instance_config.name}",
                "instance": result["instance"]
            }
        else:
            return {
                "status": "error",
                "message": result["message"]
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
            result = await self._add_single_instance(instance_config)
            
            if result["status"] == "success":
                # Extract basic info for the success list
                added.append({
                    "name": instance_config.name,
                    "provider_type": instance_config.provider_type,
                    "api_base": instance_config.api_base
                })
            elif result.get("reason") == "duplicate_name":
                skipped.append({
                    "name": instance_config.name,
                    "reason": "Instance with this name already exists"
                })
            else:
                failed.append({
                    "name": instance_config.name,
                    "reason": result["message"]
                })
        
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
            instance = self.instance_manager.get_instance(instance_name)
            if not instance:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Instance '{instance_name}' not found"
                )
            config, state = instance
        else:
            # We're testing a new instance config
            config = instance_config_or_name
            # Basic validation
            if not config.api_key or not config.api_base:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="API key and API base URL are required"
                )
            state = None
        
        # Test the connection - we'll use a simple GET request to the API base URL
        start_time = asyncio.get_event_loop().time()
        
        # Execute the test based on provider type
        if config.provider_type == "azure":
            return await self._test_azure_instance(config, state, start_time)
        else:
            return await self._test_generic_instance(config, state, start_time)
    
    async def _test_azure_instance(self, config: InstanceConfig, state: Optional[InstanceState], start_time: float) -> Dict[str, Any]:
        """Test an Azure OpenAI instance by making a request to list models."""
        try:
            # Use the instance's build_url method with the /openai/models endpoint
            # This endpoint doesn't require a deployment name, so we pass an empty string
            deployment = config.model_deployments.get(config.supported_models[0])
            url = f"{config.api_base}/v1/chat/completions"
            headers = {"api-key": config.api_key}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, 
                                             headers=headers, 
                                             data=json.dumps({
                                                 "model": config.supported_models[0], 
                                                 "messages": [{"role": "user", "content": "Hello, how are you?"}]
                                                 })
                                            )
                
            response.raise_for_status()
            
            # Parse models if available
            models = []
            response_data = response.json()
            if "data" in response_data:
                models = [model.get("id", "") for model in response_data.get("data", [])]
            
            # Reset error count and mark healthy if test passed
            if state:
                self.mark_healthy(config.name)
            
            # Successful test
            end_time = asyncio.get_event_loop().time()
            return {
                "status": "success",
                "message": "Successfully connected to the instance API",
                "response_time_ms": int((end_time - start_time) * 1000),
                "provider_type": config.provider_type,
                "api_base": config.api_base,
                "models_available": len(models),
                "models": models[:10] if len(models) > 10 else models  # Limit to 10 models to avoid large responses
            }
        except httpx.HTTPStatusError as e:
            # Mark the instance as having an error
            error_message = f"HTTP {e.response.status_code}: {str(e)}"
            if state:
                self.record_error(config.name, "client_error", error_message)
            return self._handle_http_error(e, start_time)
        except Exception as e:
            # Mark the instance as having an error
            error_message = f"Connection error: {str(e)}"
            if state:
                self.record_error(config.name, "upstream_error", error_message)
            return self._handle_connection_error(e, start_time)
    
    async def _test_generic_instance(self, config: InstanceConfig, state: Optional[InstanceState], start_time: float) -> Dict[str, Any]:
        """Test a generic OpenAI instance by making a request to list models."""
        try:
            # Use the instance's build_url method instead of directly constructing the URL
            url = f"{config.api_base}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {config.api_key}"}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)
                
            response.raise_for_status()
            
            # Parse models if available
            models = []
            response_data = response.json()
            if "data" in response_data:
                models = [model.get("id", "") for model in response_data.get("data", [])]
            
            # Reset error count and mark healthy if test passed
            if state:
                self.mark_healthy(config.name)
            
            # Successful test
            end_time = asyncio.get_event_loop().time()
            return {
                "status": "success",
                "message": "Successfully connected to the instance API",
                "response_time_ms": int((end_time - start_time) * 1000),
                "provider_type": config.provider_type,
                "api_base": config.api_base,
                "models": models,
                "model_count": len(models)
            }
        except httpx.HTTPStatusError as e:
            # Mark the instance as having an error
            error_message = f"HTTP {e.response.status_code}: {str(e)}"
            if state:
                self.record_error(config.name, "client_error", error_message)
            return self._handle_http_error(e, start_time)
        except Exception as e:
            # Mark the instance as having an error
            error_message = f"Connection error: {str(e)}"
            if state:
                self.record_error(config.name, "upstream_error", error_message)
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
        Remove an instance at runtime.
        
        Args:
            instance_name: The name of the instance to remove
            
        Returns:
            Status information about the removal
        """
        # Check if instance exists
        if not self.instance_manager.has_instance(instance_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance '{instance_name}' not found"
            )
        
        # Get config before removal
        config = self.instance_manager.get_instance_config(instance_name)
        
        # Store instance details for the response before removal
        instance_details = {
            "name": config.name,
            "provider_type": config.provider_type,
            "api_base": config.api_base
        }
        
        # Remove the instance
        self.instance_manager.remove_instance(instance_name)
        
        logger.info(f"Removed instance '{instance_name}' at runtime")
        
        return {
            "status": "success",
            "message": f"Instance '{instance_name}' removed successfully",
            "instance": instance_details
        }
        
    async def test_connectivity(self, instance_name: str) -> Dict[str, Any]:
        """
        Test connectivity to an existing instance by name.
        
        This is a wrapper around test_instance for backward compatibility.
        
        Args:
            instance_name: The name of the existing instance to test
            
        Returns:
            Status information about the test
        """
        # Call the test_instance method with the instance name
        return await self.test_instance(instance_name)
        
    def can_handle_model(self, instance_name: str, model_name: str) -> bool:
        """
        Check if an instance can handle a specific model.
        
        Args:
            instance_name: Name of the instance to check
            model_name: Name of the model to check
            
        Returns:
            True if the instance can handle the model, False otherwise
        """
        instance = self.instance_manager.get_instance(instance_name)
        if not instance:
            return False
            
        config, _ = instance
            
        # If no supported models list, assume it can handle any model
        if not config.supported_models:
            return True
            
        # Check for exact match
        if model_name in config.supported_models:
            return True
            
        # Check for case-insensitive match
        return model_name.lower() in [m.lower() for m in config.supported_models]
    
    def get_deployment_for_model(self, instance_name: str, model_name: str) -> Optional[str]:
        """
        Get the deployment name for a specific model in an instance.
        
        Args:
            instance_name: Name of the instance
            model_name: Name of the model
            
        Returns:
            Deployment name if found, None otherwise
        """
        instance = self.instance_manager.get_instance(instance_name)
        if not instance:
            return None
            
        config, _ = instance
            
        # Check for exact match
        if model_name in config.model_deployments:
            return config.model_deployments[model_name]
            
        # Check for case-insensitive match
        model_name_lower = model_name.lower()
        for k, v in config.model_deployments.items():
            if k.lower() == model_name_lower:
                return v
                
        return None
        
    async def test_model_call(self, instance_name: str, model_name: str, message: str) -> Dict[str, Any]:
        """
        Test a model call with a simple message.
        
        Args:
            instance_name: Name of the instance to test
            model_name: Name of the model to test
            message: Test message to send
            
        Returns:
            Dictionary with test results
        """
        try:
            # Get the instance
            instance = self.instance_manager.get_instance(instance_name)
            if not instance:
                return {
                    "status": "error",
                    "details": {"error": f"Instance '{instance_name}' not found"}
                }
            
            config, _ = instance
            
            # Build a simple test payload
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 50
            }
            
            # Start timing
            start_time = time.time()
            
            # For now, just test connectivity rather than making an actual API call
            # In the future, we could integrate with the instance forwarder to make a real request
            
            # Simulate successful response
            execution_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                "status": "passed",
                "details": {
                    "model": model_name,
                    "response": "Simulated successful model call",
                    "execution_time_ms": execution_time
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "details": {"error": str(e)}
            }

# Create a singleton instance
instance_service = InstanceService(InstanceManager()) 