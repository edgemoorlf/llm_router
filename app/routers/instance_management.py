"""API for instance management (adding, removal, health)"""
from datetime import datetime
import logging
import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query, Body, HTTPException, BackgroundTasks, Request, Depends, status
from fastapi.responses import JSONResponse

from app.models.instance import InstanceConfig, InstanceStatus
from app.instance.instance_context import instance_manager
from app.services.instance_service import instance_service
from app.services.instance_verification_service import instance_verification_service
from app.services.instance_helper import filter_instances, sort_instances, paginate_instances, format_instance_list, filter_instances_optimized, filter_with_states
from app.errors.handlers import handle_errors
from app.errors.utils import check_instance_exists, handle_router_errors

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/instances",
    tags=["Instance Management"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
@handle_router_errors("getting instances configuration")
async def get_instances_config(
    request: Request,
    provider_type: Optional[str] = None,
    status: Optional[str] = None,
    model_support: Optional[str] = None,
    sort_dir: str = Query("asc", regex="^(asc|desc)$"),
    min_tpm: Optional[int] = None,
    max_tpm: Optional[int] = None,
    offset: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1),
    detailed: bool = Query(False)
) -> Dict[str, Any]:
    """
    Get all instances with optional filtering and sorting.
    
    Args:
        provider_type: Filter by provider type (azure, generic)
        status: Filter by instance status (healthy, error, rate_limited)
        model_support: Filter instances that support this model
        sort_dir: Sort direction - asc or desc
        min_tpm: Filter instances with current TPM >= this value
        max_tpm: Filter instances with max TPM <= this value
        offset: Start of pagination
        limit: Maximum instances to return
        detailed: Return detailed information about instances
    
    Returns:
        List of instance configurations that match the filters
    """
    # Optimization: First filter by config-only criteria (no Redis calls)
    all_configs = instance_manager.get_all_instance_configs()
    logger.debug(f"Initial config count: {len(all_configs)}")
    
    # Apply config-only filters
    filtered_configs = filter_instances_optimized(all_configs, provider_type, model_support)
    logger.debug(f"After config filtering: {len(filtered_configs)}")
    
    # If we need state-based filtering, get states only for the filtered configs
    if status is not None or min_tpm is not None:
        # Get the states only for configs that passed initial filtering
        instances_with_states = instance_manager.get_states_for_configs(filtered_configs)
        logger.debug(f"Got states for {len(instances_with_states)} instances")
        
        # Apply state-based filtering
        filtered = filter_with_states(instances_with_states, status, min_tpm, max_tpm)
    else:
        # If we don't need state filtering, get states for all filtered configs
        filtered = instance_manager.get_states_for_configs(filtered_configs)
    
    # Apply sorting
    sorted_instances = sort_instances(filtered, sort_dir == "desc")
    
    # Apply pagination
    paginated = paginate_instances(sorted_instances, offset, limit)
    
    # Format response
    formatted = format_instance_list(paginated, detailed)
    
    return {
        "status": "success",
        "message": f"Retrieved {len(filtered)} instances",
        "timestamp": int(time.time()),
        "data": {
            "instances": formatted,
            "total_available": len(all_configs),
            "total_filtered": len(filtered),
            "count": len(paginated),
            "pagination": {
                "offset": offset,
                "limit": limit
            },
            "filters": {
                "sort_dir": sort_dir
            }
        }
    }

@router.get("/debug-rate-limits")
@handle_router_errors("debugging rate limits")
async def debug_rate_limits(instance_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Debug rate limit data for instances.
    
    This endpoint provides detailed information about rate limiting for debugging purposes,
    including:
    1. Active rate limit keys and their TTLs
    2. Token usage windows with per-entry breakdown
    3. Total token usage per instance
    
    Args:
        instance_name: Optional name of a specific instance to debug
        
    Returns:
        Detailed rate limit diagnostics
    """
    # Check if we have access to the state store
    if not hasattr(instance_manager, "state_store") or not hasattr(instance_manager.state_store, "dump_rate_limit_data"):
        return {
            "status": "error",
            "message": "Rate limit debugging not supported by the current state store",
            "timestamp": int(time.time())
        }
    
    # Get the diagnostic data
    rate_limit_data = instance_manager.state_store.dump_rate_limit_data(instance_name)
    
    # Add instance config data if looking at a specific instance
    if instance_name:
        config = instance_manager.get_instance_config(instance_name)
        state = instance_manager.get_instance_state(instance_name)
        
        # Add to the result
        if config:
            rate_limit_data["instance_config"] = {
                "name": config.name,
                "provider_type": config.provider_type,
                "max_tpm": config.max_tpm
            }
        
        if state:
            rate_limit_data["instance_state"] = {
                "status": state.status.value,
                "current_tpm": state.current_tpm,
                "rate_limited_until": state.rate_limited_until
            }
    
    return {
        "status": "success",
        "message": "Rate limit diagnostic information",
        "timestamp": int(time.time()),
        "data": rate_limit_data
    }

@router.post("/reset-all-rate-limits")
@handle_router_errors("resetting all rate limits")
async def reset_all_rate_limits() -> Dict[str, Any]:
    """
    Reset rate limit status for all rate-limited instances.
    
    This endpoint will scan all instances and:
    1. Identify all instances that are currently rate-limited
    2. Reset their status to healthy
    3. Delete all associated rate limit keys
    
    This is useful for system-wide recovery after a burst of rate limit errors,
    or when restarting services after a period of inactivity.
    
    Returns:
        Standardized response with stats on how many instances were affected
    """
    # Get all instances
    states = instance_manager.get_all_states()
    
    # Track which instances were reset
    reset_instances = []
    
    # Check each instance
    for name, state in states.items():
        if state.status == InstanceStatus.RATE_LIMITED:
            # Reset this instance
            instance_manager.mark_healthy(name)
            
            # Explicitly delete any rate limit keys
            if hasattr(instance_manager.state_store, "redis"):
                rate_limit_key = f"{instance_manager.state_store.rate_limit_prefix}{name}"
                instance_manager.state_store.redis.delete(rate_limit_key)
                
            reset_instances.append(name)
            logger.info(f"Reset rate limit for instance {name}")
    
    return {
        "status": "success",
        "message": f"Reset rate limits for {len(reset_instances)} instances",
        "timestamp": int(time.time()),
        "data": {
            "affected_instances": reset_instances,
            "count": len(reset_instances)
        }
    }

@router.get("/{instance_name}")
@handle_router_errors("retrieving instance details")
async def get_instance_details(
    instance_name: str,
    test_connection: bool = Query(False, description="Test the connection to the instance API")
) -> Dict[str, Any]:
    """
    Get comprehensive information about a specific instance including:
    - Configuration
    - Current state
    - Health information
    - Usage statistics
    
    Args:
        instance_name: The name of the instance to retrieve
        test_connection: Whether to test the connection to the instance API
        
    Returns:
        Standardized response with detailed instance information
        
    Raises:
        InstanceNotFoundError: If the instance doesn't exist
    """
    # First check if the instance exists
    if not instance_manager.has_instance(instance_name):
        check_instance_exists(None, instance_name)  # This will raise an error
        
    # Get instance configuration and state
    config = instance_manager.get_instance_config(instance_name)
    state = instance_manager.get_instance_state(instance_name)
    
    # Get stats from the metrics function
    instance_stats = instance_manager.get_metrics(instance_name) or {}
    
    # Get instance config as dict
    instance_config = config.dict() if config else {}
    
    # Remove sensitive information
    if "api_key" in instance_config:
        instance_config["api_key"] = "**REDACTED**"
    
    # Get health information
    health_status = "unknown"
    health_details = {}
    
    # Generate basic health status from state
    if state:
        if state.status == InstanceStatus.ERROR.value:
            health_status = InstanceStatus.ERROR.value
            health_details["error_count"] = state.error_count
            health_details["last_error"] = state.last_error
            health_details["last_error_time"] = state.last_error_time
        elif state.status == InstanceStatus.RATE_LIMITED.value:
            health_status = InstanceStatus.RATE_LIMITED.value
            health_details["rate_limited_until"] = state.rate_limited_until
        else:
            health_status = InstanceStatus.HEALTHY.value
        
        # Get last used time
        if "last_used" in instance_stats:
            health_details["last_used"] = instance_stats["last_used"]
            
            # Calculate time since last used
            if instance_stats["last_used"]:
                last_used_time = datetime.fromtimestamp(instance_stats["last_used"])
                now = datetime.now()
                idle_seconds = (now - last_used_time).total_seconds()
                health_details["idle_time"] = {
                    "seconds": int(idle_seconds),
                    "minutes": round(idle_seconds / 60, 1),
                    "hours": round(idle_seconds / 3600, 2)
                }
    
    # Test connectivity if requested
    if test_connection:
        try:
            connection_test = await instance_service.test_connectivity(instance_name)
            health_details["connection_test"] = connection_test
            
            # Update health status based on connection test
            if connection_test["status"] != "passed":
                health_status = InstanceStatus.ERROR.value
        except Exception as e:
            health_details["connection_test"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
            health_status = InstanceStatus.ERROR.value
    
    # Build the response
    result = {
        "name": instance_name,
        "status": health_status,  # Overall status from health checks
        "config": instance_config,
        "stats": instance_stats,
        "health": {
            "status": health_status,
            "details": health_details
        },
        "timestamp": int(time.time())
    }
    
    return result

@router.post("/add")
@handle_router_errors("adding new instance")
async def add_instance(instance_config: InstanceConfig) -> Dict[str, Any]:
    """
    Add a new instance to the proxy service.
    
    This operation adds a new instance configuration to the system. The configuration is:
    1. Saved to the instance_configs.json file
    2. Persisted in Redis for state information
    3. Available immediately for routing requests
    
    The configuration will persist across service restarts.
    
    Args:
        instance_config: Configuration for the new instance
        
    Returns:
        Standardized response with status information about the newly added instance
    """
    result = await instance_service.add_instance(instance_config)
    return {
        "status": "success",
        "message": f"Instance '{instance_config.name}' added successfully",
        "timestamp": int(time.time()),
        "data": result
    }

@router.post("/add-many")
@handle_router_errors("adding multiple instances")
async def add_many_instances(instances_config: List[InstanceConfig]) -> Dict[str, Any]:
    """
    Add multiple instances to the proxy service.
    
    This operation adds multiple instance configurations to the system. Each configuration is:
    1. Saved to the instance_configs.json file
    2. Persisted in Redis for state information
    3. Available immediately for routing requests
    
    All configurations will persist across service restarts.
    
    Args:
        instances_config: List of configurations for the new instances
        
    Returns:
        Standardized response with information about successfully added, skipped, and failed instances
    """
    result = await instance_service.add_many_instances(instances_config)
    
    # Create a more structured response
    return {
        "status": "success",
        "message": f"Processed {len(instances_config)} instance configurations",
        "timestamp": int(time.time()),
        "data": {
            "added": result.get("added", []),
            "skipped": result.get("skipped", []),
            "failed": result.get("failed", {})
        }
    }

@router.delete("/{instance_name}")
@handle_router_errors("removing instance")
async def remove_instance(instance_name: str) -> Dict[str, Any]:
    """
    Remove an instance from the proxy service.
    
    This operation removes an instance from the system, which includes:
    1. Removing the configuration from instance_configs.json file
    2. Removing all state information from Redis
    3. Making the instance unavailable for routing requests
    
    The removal will persist across service restarts.
    
    Args:
        instance_name: Name of the instance to remove
        
    Returns:
        Standardized response with information about the removed instance
    """
    result = await instance_service.remove_instance(instance_name)
    return {
        "status": "success",
        "message": f"Instance '{instance_name}' removed successfully",
        "timestamp": int(time.time()),
        "data": {"name": instance_name, "removed": result}
    }

@router.patch("/{instance_name}")
@handle_router_errors("updating instance")
async def update_instance(
    instance_name: str, 
    update_data: Dict[str, Any] = Body(..., description="Attributes to update")
) -> Dict[str, Any]:
    """
    Update attributes of an existing instance.
    
    This endpoint allows updating instance configuration attributes, such as 
    priority, weight, max_tpm, etc. You can provide only the fields
    you want to update.
    
    The update process includes:
    1. Updating the configuration in the instance_configs.json file
    2. Updating relevant state information in Redis if needed
    3. Immediately applying changes to the running instance
    
    Sensitive attributes like api_key are only updated if explicitly provided.
    If api_base or provider_type are changed, the instance client will be re-initialized.
    
    All changes will persist across service restarts.
    
    Args:
        instance_name: Name of the instance to update
        update_data: Dictionary of attributes to update
        
    Returns:
        Standardized response with information about the updated instance
    """
    result = await instance_service.update_instance(instance_name, update_data)
    return {
        "status": result["status"],
        "message": result["message"],
        "timestamp": int(time.time()),
        "data": {
            "name": instance_name,
            "updated_fields": result.get("updated_fields", {}),
            "instance": result.get("instance", {})
        }
    }

@router.post("/verify/{instance_name}")
@handle_router_errors("verifying instance")
async def verify_instance(
    instance_name: str,
    request: Optional[Dict[str, Any]] = Body(None, description="Optional request payload with model and message to use for testing")
) -> Dict[str, Any]:
    """
    Verify an instance by running a comprehensive set of checks.
    
    This endpoint performs a full verification of an instance including:
    - Configuration check
    - Connectivity test
    - Model support check
    - Actual model testing
    
    Args:
        instance_name: Name of the instance to verify
        request: Optional request payload with model and message to use for testing
        
    Returns:
        Standardized response with detailed verification results
        
    Raises:
        InstanceNotFoundError: If the instance doesn't exist
    """
    # Determine which model to test
    model_to_test = None
    message_to_test = "This is a test message to verify instance functionality."
    
    # If request payload is provided, use the model and message from it
    if request:
        model_to_test = request.get("model")
        if "message" in request:
            message_to_test = request.get("message")
            
    # Call the verification service
    result = await instance_verification_service.verify_instance(
        instance_name=instance_name,
        model_to_test=model_to_test,
        message_to_test=message_to_test
    )
    
    return result 

@router.post("/{instance_name}/reset-health")
@handle_router_errors("resetting instance health")
async def reset_instance_health(instance_name: str) -> Dict[str, Any]:
    """
    Reset an instance's health status to healthy.
    
    This endpoint allows you to manually reset an instance that might have been 
    wrongfully marked as error or rate-limited. It will:
    1. Reset the error counter and clear error messages
    2. Change the status to healthy
    3. Remove any rate-limiting settings if present
    
    This is useful for cases where instances have been automatically marked as
    unhealthy due to temporary issues or false positives.
    
    Args:
        instance_name: Name of the instance to reset health status
        
    Returns:
        Standardized response with status information
        
    Raises:
        InstanceNotFoundError: If the instance doesn't exist
    """
    # Check if the instance exists first
    check_instance_exists(instance_manager.get_instance_config(instance_name), instance_name)
    
    # Get the current state to capture previous status
    previous_state = instance_manager.get_instance_state(instance_name)
    previous_status = previous_state.status.value if previous_state else "unknown"
    
    # Reset the instance health
    instance_manager.mark_healthy(instance_name)
    
    # Explicitly delete any rate limit keys
    if hasattr(instance_manager.state_store, "redis"):
        rate_limit_key = f"{instance_manager.state_store.rate_limit_prefix}{instance_name}"
        instance_manager.state_store.redis.delete(rate_limit_key)
        logger.info(f"Explicitly deleted rate limit key for instance {instance_name}")
    
    # Get the current state to confirm the change
    state = instance_manager.get_instance_state(instance_name)
    
    return {
        "status": "success",
        "message": f"Instance '{instance_name}' health status reset to healthy",
        "timestamp": int(time.time()),
        "data": {
            "name": instance_name,
            "previous_status": previous_status,
            "current_status": state.status.value if state else "unknown"
        }
    }

@router.post("/{instance_name}/reset-rate-limit")
@handle_router_errors("resetting rate limit")
async def reset_rate_limit(instance_name: str) -> Dict[str, Any]:
    """
    Reset an instance's rate limit status.
    
    This endpoint specifically targets instances that are in a rate-limited state.
    It will:
    1. Remove the rate-limited status
    2. Clear any rate limit expiration timestamps
    3. Delete the rate limit tracking keys in Redis
    
    This is useful for cases where instances have been automatically rate-limited
    and you need to force them back to service immediately.
    
    Args:
        instance_name: Name of the instance to reset rate limit
        
    Returns:
        Standardized response with status information
        
    Raises:
        InstanceNotFoundError: If the instance doesn't exist
    """
    # Check if the instance exists first
    check_instance_exists(instance_manager.get_instance_config(instance_name), instance_name)
    
    # Get the current state to capture previous status
    previous_state = instance_manager.get_instance_state(instance_name)
    previous_status = previous_state.status.value if previous_state else "unknown"
    
    # Only perform actions if the instance is actually rate-limited
    is_rate_limited = (previous_state and previous_state.status == InstanceStatus.RATE_LIMITED)
    
    if is_rate_limited:
        # Clear rate limit and mark healthy
        instance_manager.mark_healthy(instance_name)
        
        # Explicitly delete any rate limit keys
        if hasattr(instance_manager.state_store, "redis"):
            rate_limit_key = f"{instance_manager.state_store.rate_limit_prefix}{instance_name}"
            instance_manager.state_store.redis.delete(rate_limit_key)
            logger.info(f"Explicitly deleted rate limit key for instance {instance_name}")
        
        message = f"Rate limit for instance '{instance_name}' has been cleared"
    else:
        message = f"Instance '{instance_name}' was not rate-limited, no action taken"
    
    # Get the current state to confirm the change
    state = instance_manager.get_instance_state(instance_name)
    
    return {
        "status": "success",
        "message": message,
        "timestamp": int(time.time()),
        "data": {
            "name": instance_name,
            "previous_status": previous_status,
            "current_status": state.status.value if state else "unknown",
            "was_rate_limited": is_rate_limited
        }
    } 