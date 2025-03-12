"""Configuration API router."""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Body, Query
from pydantic import BaseModel, Field
import time

from app.config import config_loader, InstanceConfig
from app.instance.manager import instance_manager
from app.services.instance_service import instance_service
from app.instance.forwarder import instance_forwarder

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/config",
    tags=["Configuration"],
    responses={404: {"description": "Not found"}},
)

class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration."""
    routing_strategy: str = None
    stats_window_minutes: int = None

@router.get("/")
async def get_config() -> Dict[str, Any]:
    """Get the current configuration (excluding secrets)."""
    try:
        return config_loader.to_dict()
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving configuration: {str(e)}"
        )

@router.get("/instances")
async def get_instances_config() -> Dict[str, Any]:
    """Get the current instance configuration (excluding secrets)."""
    try:
        config_dict = config_loader.to_dict()
        return {
            "instances": config_dict.get("instances", []),
            "total": len(config_dict.get("instances", []))
        }
    except Exception as e:
        logger.error(f"Error getting instance configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving instance configuration: {str(e)}"
        )

@router.post("/reload")
async def reload_config() -> Dict[str, Any]:
    """Reload configuration from disk."""
    try:
        # Reload configuration
        config = config_loader.reload()
        if not config:
            raise ValueError("Failed to reload configuration")
        
        # Reload instance manager
        instance_manager.reload_config()
        
        return {
            "status": "success",
            "message": "Configuration reloaded successfully",
            "instances": len(instance_manager.instances)
        }
    except Exception as e:
        logger.error(f"Error reloading configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading configuration: {str(e)}"
        )

@router.post("/update")
async def update_config(update_request: ConfigUpdateRequest) -> Dict[str, Any]:
    """Update configuration values."""
    try:
        updated_fields = {}
        
        # Update routing strategy if provided
        if update_request.routing_strategy is not None:
            valid_strategies = ["priority", "round_robin", "weighted_random"]
            if update_request.routing_strategy not in valid_strategies:
                raise ValueError(f"Invalid routing strategy. Must be one of: {', '.join(valid_strategies)}")
            
            config_loader.config.routing_strategy = update_request.routing_strategy
            updated_fields["routing_strategy"] = update_request.routing_strategy
        
        # Update stats window if provided
        if update_request.stats_window_minutes is not None:
            if update_request.stats_window_minutes < 1 or update_request.stats_window_minutes > 1440:
                raise ValueError("Stats window must be between 1 and 1440 minutes")
            
            config_loader.config.stats_window_minutes = update_request.stats_window_minutes
            updated_fields["stats_window_minutes"] = update_request.stats_window_minutes
        
        # If no fields were updated, return an error
        if not updated_fields:
            raise ValueError("No valid fields provided for update")
        
        # Save the updated configuration
        config_loader.save()
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "updated_fields": updated_fields
        }
    except ValueError as e:
        logger.error(f"Invalid configuration update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating configuration: {str(e)}"
        )

@router.post("/instances/add")
async def add_instance(instance_config: InstanceConfig) -> Dict[str, Any]:
    """
    Add a new instance to the proxy service at runtime.
    
    Note: This change is not persisted to the configuration file and will be lost on service restart.
    """
    try:
        return await instance_service.add_instance(instance_config)
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error adding instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding instance: {str(e)}"
        )

@router.post("/instances/add-many")
async def add_many_instances(instances: List[InstanceConfig]) -> Dict[str, Any]:
    """
    Add multiple instances to the proxy service at runtime.
    
    Note: These changes are not persisted to the configuration file and will be lost on service restart.
    """
    try:
        return await instance_service.add_many_instances(instances)
    except Exception as e:
        logger.error(f"Error adding instances: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding instances: {str(e)}"
        )

@router.post("/instances/test")
async def test_instance(instance_config: InstanceConfig) -> Dict[str, Any]:
    """
    Test an instance configuration before adding it to the proxy service.
    
    This endpoint creates a temporary instance and tests the connection to the API.
    The instance is not added to the proxy service.
    """
    try:
        return await instance_service.test_instance(instance_config)
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error testing instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing instance: {str(e)}"
        )

@router.delete("/instances/{instance_name}")
async def remove_instance(instance_name: str) -> Dict[str, Any]:
    """
    Remove an instance from the proxy service at runtime.
    
    Note: This change is not persisted to the configuration file and will be lost on service restart.
    """
    try:
        return await instance_service.remove_instance(instance_name)
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error removing instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing instance: {str(e)}"
        )

@router.get("/instances/{instance_name}/stats")
async def get_instance_stats(instance_name: str) -> Dict[str, Any]:
    """
    Get detailed statistics for a specific instance.
    
    This endpoint is useful for monitoring the health and performance of 
    specific instances, particularly newly deployed ones.
    
    Args:
        instance_name: The name of the instance to get statistics for
        
    Returns:
        Detailed statistics about the specified instance
        
    Raises:
        HTTPException: If the instance doesn't exist
    """
    try:
        # Get all instance stats
        all_stats = instance_manager.get_instance_stats()
        
        # Find the specific instance
        instance_stats = next((stats for stats in all_stats if stats["name"] == instance_name), None)
        
        if not instance_stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance '{instance_name}' not found"
            )
        
        return {
            "status": "success",
            "instance": instance_stats
        }
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error getting instance stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting instance stats: {str(e)}"
        )

@router.get("/instances/{instance_name}/health")
async def check_instance_health(
    instance_name: str, 
    test_connection: bool = Query(False, description="Test the connection to the instance API")
) -> Dict[str, Any]:
    """
    Check the health of a specific instance.
    
    This endpoint provides a quick health check for a specific instance, 
    with an option to test the connection to the instance API.
    
    Args:
        instance_name: The name of the instance to check
        test_connection: Whether to test the connection to the instance API
        
    Returns:
        Health information about the specified instance
        
    Raises:
        HTTPException: If the instance doesn't exist
    """
    try:
        # First check if the instance exists
        if instance_name not in instance_manager.instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance '{instance_name}' not found"
            )
            
        # Get basic health info
        instance = instance_manager.instances[instance_name]
        
        health_info = {
            "name": instance.name,
            "status": instance.status,
            "provider_type": instance.provider_type,
            "api_base": instance.api_base,
            "error_count": instance.error_count,
            "last_error": instance.last_error,
            "rate_limited_until": instance.rate_limited_until,
            "current_tpm": instance.instance_stats.current_tpm,
            "max_tpm": instance.max_tpm,
            "tpm_usage_percent": round((instance.instance_stats.current_tpm / instance.max_tpm) * 100, 2) if instance.max_tpm > 0 else 0,
            "last_used": instance.last_used
        }
        
        # If requested, test the connection
        if test_connection:
            # Pass the instance name directly to test_instance
            test_result = await instance_service.test_instance(instance_name)
            health_info["connection_test"] = test_result
        
        return {
            "status": "success",
            "health": health_info
        }
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error checking instance health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking instance health: {str(e)}"
        )

@router.get("/instances/health")
async def get_all_instances_health(
    status_filter: Optional[str] = Query(None, description="Filter instances by status: healthy, error, rate_limited"),
    provider_type: Optional[str] = Query(None, description="Filter instances by provider type: azure, generic"),
    test_newly_added: bool = Query(False, description="Test newly added instances that have not been used yet"),
    last_used_threshold_minutes: int = Query(60, description="Threshold in minutes for considering an instance as newly added")
) -> Dict[str, Any]:
    """
    Get health information for all instances with filtering options.
    
    This endpoint provides health information for all instances, with options to filter
    by status, provider type, and to test newly added instances.
    
    Args:
        status_filter: Filter instances by status (healthy, error, rate_limited)
        provider_type: Filter instances by provider type (azure, generic)
        test_newly_added: Whether to test instances that have not been used yet
        last_used_threshold_minutes: Threshold in minutes for considering an instance as newly added
        
    Returns:
        Health information for all instances, with filtering and test results
    """
    try:
        # Get all instances
        instances = instance_manager.instances
        
        # Filter instances by status if requested
        filtered_instances = {}
        for name, instance in instances.items():
            include = True
            
            # Apply status filter if specified
            if status_filter and instance.status != status_filter:
                include = False
                
            # Apply provider type filter if specified
            if provider_type and instance.provider_type != provider_type:
                include = False
                
            if include:
                filtered_instances[name] = instance
                
        # Prepare basic health information for each instance
        health_info = []
        current_time = time.time()
        threshold_time = current_time - (last_used_threshold_minutes * 60)
        
        for name, instance in filtered_instances.items():
            instance_health = {
                "name": instance.name,
                "status": instance.status,
                "provider_type": instance.provider_type,
                "api_base": instance.api_base,
                "error_count": instance.error_count,
                "last_error": instance.last_error,
                "rate_limited_until": instance.rate_limited_until,
                "current_tpm": instance.instance_stats.current_tpm,
                "max_tpm": instance.max_tpm,
                "tpm_usage_percent": round((instance.instance_stats.current_tpm / instance.max_tpm) * 100, 2) if instance.max_tpm > 0 else 0,
                "last_used": instance.last_used,
                "is_newly_added": instance.last_used is None or (isinstance(instance.last_used, (int, float)) and instance.last_used < threshold_time)
            }
            
            # Optionally test newly added instances
            if test_newly_added and instance_health["is_newly_added"]:
                # Pass the instance name directly to test_instance
                try:
                    test_result = await instance_service.test_instance(instance.name)
                    instance_health["connection_test"] = test_result
                except Exception as e:
                    instance_health["connection_test"] = {
                        "status": "error",
                        "message": f"Error testing instance: {str(e)}"
                    }
            
            health_info.append(instance_health)
        
        # Return health information
        return {
            "status": "success",
            "total_instances": len(instances),
            "filtered_instances": len(filtered_instances),
            "instances": health_info
        }
    except Exception as e:
        logger.error(f"Error getting all instances health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting all instances health: {str(e)}"
        )

@router.post("/instances/{instance_name}/verify")
async def verify_instance(
    instance_name: str,
    model: str = Body(None, description="Model to test with"),
    test_message: str = Body("This is a test message to verify the instance is working correctly.", description="Message to use for testing the instance")
) -> Dict[str, Any]:
    """
    Perform a comprehensive verification of an instance.
    
    This endpoint performs a thorough verification by checking configuration,
    model support, and sending a test message to the instance.
    This is particularly useful for newly deployed instances to ensure they are
    correctly configured and responding as expected.
    
    Args:
        instance_name: The name of the instance to verify
        model: The model to test with (optional, will use first supported model if not provided)
        test_message: The message to use for testing
        
    Returns:
        Verification results for the instance
        
    Raises:
        HTTPException: If the instance doesn't exist or verification fails
    """
    try:
        # Check if the instance exists
        if instance_name not in instance_manager.instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance '{instance_name}' not found"
            )
            
        # Get the instance
        instance = instance_manager.instances[instance_name]
        
        # Basic verification info
        verification_info = {
            "name": instance.name,
            "provider_type": instance.provider_type,
            "api_base": instance.api_base,
            "status": instance.status,
            "verification_timestamp": time.time(),
            "verification_steps": []
        }
        
        # Step 1: Verify instance configuration
        verification_info["verification_steps"].append({
            "step": "configuration_check",
            "status": "success",
            "details": {
                "provider_type": instance.provider_type,
                "api_base": instance.api_base,
                "api_version": instance.api_version,
                "max_tpm": instance.max_tpm,
                "max_input_tokens": instance.max_input_tokens,
                "supported_models": instance.supported_models,
                "model_deployments": instance.model_deployments
            }
        })
            
        # Step 2: Check if the model is supported and select a model if none provided
        if model is None:
            if instance.supported_models:
                model = instance.supported_models[0]
            else:
                verification_info["status"] = "failure"
                verification_info["message"] = "No supported models found for this instance."
                return verification_info
        
        model_supported = False
        deployment = model  # Default for non-Azure providers
        
        # For Azure, we need to check the model deployments mapping
        if instance.provider_type == "azure":
            if instance.model_deployments and model in instance.model_deployments:
                model_supported = True
                deployment = instance.model_deployments[model]
            elif model in instance.supported_models:
                model_supported = True
                # If it's in supported_models but not in model_deployments, 
                # assume the deployment name is the same as the model name
                deployment = model
        else:
            # For non-Azure providers, just check supported_models
            if model in instance.supported_models:
                model_supported = True
            
        verification_info["verification_steps"].append({
            "step": "model_support_check",
            "status": "success" if model_supported else "failure",
            "details": {
                "requested_model": model,
                "supported_models": instance.supported_models,
                "model_deployments": instance.model_deployments,
                "selected_deployment": deployment if model_supported else None
            }
        })
        
        if not model_supported:
            verification_info["status"] = "failure"
            verification_info["message"] = f"The requested model '{model}' is not supported by this instance."
            return verification_info
        
        # Step 3: Test the model with a simple message
        try:
            # Prepare a simple chat message
            # Base payload - will be adjusted based on provider type
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": test_message}
                ],
                "max_tokens": 100
            }
            
            # Set up the appropriate endpoint based on provider type
            if instance.provider_type == "azure":
                # Azure OpenAI format
                endpoint = f"/openai/deployments/{deployment}/chat/completions"
                # For Azure, we don't include the model field in the payload
                # The model is specified in the URL path
            else:
                # Regular OpenAI format
                endpoint = "/v1/chat/completions"
                # For non-Azure, include the model in the payload
                payload["model"] = model
            
            # Use the instance's forward_request method directly to ensure proper handling
            try:
                # Use the instance's build_url method to get the correct URL
                response = await instance_forwarder.forward_request(
                    instance=instance,
                    endpoint=endpoint,
                    deployment=deployment,
                    payload=payload,
                    method="POST"
                )
                
                # Extract useful information from the response
                response_content = ""
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        response_content = choice["message"]["content"]
                
                verification_info["verification_steps"].append({
                    "step": "model_test",
                    "status": "success",
                    "details": {
                        "model": model,
                        "deployment": deployment,
                        "response_received": True,
                        "response_content": response_content,
                        "tokens_used": response.get("usage", {})
                    }
                })
                
                # Update the instance's usage stats
                if "usage" in response and "total_tokens" in response["usage"]:
                    instance.update_tpm_usage(response["usage"]["total_tokens"])
                    instance.mark_healthy()
                
            except HTTPException as e:
                # Handle HTTP exceptions
                verification_info["verification_steps"].append({
                    "step": "model_test",
                    "status": "error",
                    "details": {
                        "error": f"HTTP {e.status_code}: {e.detail}"
                    }
                })
                verification_info["status"] = "failure"
                verification_info["message"] = f"Error during model test: {e.detail}"
                return verification_info
                
        except Exception as e:
            verification_info["verification_steps"].append({
                "step": "model_test",
                "status": "error",
                "details": {
                    "error": str(e)
                }
            })
            verification_info["status"] = "failure"
            verification_info["message"] = f"Error during model test: {str(e)}"
            return verification_info
        
        # Add a connectivity test step based on the model test results
        verification_info["verification_steps"].insert(1, {
            "step": "connectivity_test",
            "status": "success",
            "details": {
                "model": model,
                "deployment": deployment,
                "api_base": instance.api_base,
                "provider_type": instance.provider_type
            }
        })
        
        # Final verification status
        verification_info["status"] = "success"
        verification_info["message"] = "Instance verification completed successfully. All tests passed."
        
        return verification_info
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error verifying instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error verifying instance: {str(e)}"
        ) 