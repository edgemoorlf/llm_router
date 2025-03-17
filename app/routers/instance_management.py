"""Instance management API router."""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Body, BackgroundTasks
import time
from datetime import datetime

from app.config import InstanceConfig
from app.instance.instance_context import instance_manager
from app.services.instance_service import instance_service
from app.services.instance_verification_service import instance_verification_service
from app.services.instance_helper import filter_instances, sort_instances, paginate_instances, format_instance_list
from app.errors.handlers import handle_errors
from app.errors.utils import check_instance_exists, handle_router_errors

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/instances",
    tags=["Instance Management"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def get_instances_config(
    provider_type: Optional[str] = Query(None, description="Filter by provider type (azure, generic)"),
    status: Optional[str] = Query(None, description="Filter by instance status (healthy, error, rate_limited)"),
    min_tpm: Optional[int] = Query(None, description="Filter instances with current TPM >= this value"),
    max_tpm: Optional[int] = Query(None, description="Filter instances with max TPM <= this value"),
    model_support: Optional[str] = Query(None, description="Filter instances that support this model"),
    sort_by: Optional[str] = Query(None, description="Field to sort by (name, status, current_tpm, priority)"),
    sort_dir: Optional[str] = Query("asc", description="Sort direction (asc, desc)"),
    limit: Optional[int] = Query(None, description="Limit the number of returned instances"),
    offset: Optional[int] = Query(0, description="Offset for pagination")
) -> Dict[str, Any]:
    """
    Get the current instance configuration with enhanced filtering and sorting.
    
    This endpoint returns information about all configured instances with extensive filtering options.
    
    Args:
        provider_type: Filter by provider type (azure, generic)
        status: Filter by instance status (healthy, error, rate_limited)
        min_tpm: Filter instances with current TPM >= this value
        max_tpm: Filter instances with max TPM <= this value
        model_support: Filter instances that support this model
        sort_by: Field to sort by (name, status, current_tpm, priority)
        sort_dir: Sort direction (asc, desc)
        limit: Limit the number of returned instances
        offset: Offset for pagination
        
    Returns:
        Standardized response with filtered instance configuration information
    """
    # Define inner function to apply the decorator
    @handle_errors
    async def _get_instances_config() -> Dict[str, Any]:

        # Get all instances as a list
        all_instances = list(instance_manager.get_all_instances().values())
        
        # Filter instances
        filtered_instances = filter_instances(
            all_instances,
            provider_type=provider_type,
            status=status,
            min_tpm=min_tpm,
            max_tpm=max_tpm,
            model_support=model_support
        )
        
        # Sort instances
        filtered_instances = sort_instances(
            filtered_instances,
            sort_by=sort_by,
            sort_dir=sort_dir
        )
        
        # Calculate total before pagination
        total_filtered = len(filtered_instances)
        
        # Apply pagination
        paginated_instances = paginate_instances(
            filtered_instances,
            offset=offset,
            limit=limit
        )
        
        # Format the instance data
        instances_list = format_instance_list(paginated_instances, detailed=True)
        
        # Create applied_filters object for response metadata
        applied_filters = {
            "provider_type": provider_type,
            "status": status,
            "min_tpm": min_tpm,
            "max_tpm": max_tpm,
            "model_support": model_support,
            "sort_by": sort_by,
            "sort_dir": sort_dir
        }
        # Remove None values
        applied_filters = {k: v for k, v in applied_filters.items() if v is not None}
        
        return {
            "status": "success",
            "message": f"Retrieved {len(instances_list)} instances",
            "timestamp": int(time.time()),
            "data": {
                "instances": instances_list,
                "total_available": len(all_instances),
                "total_filtered": total_filtered,
                "count": len(instances_list),
                "pagination": {
                    "offset": offset,
                    "limit": limit
                },
                "filters": applied_filters
            }
        }
    
    # Call the inner function to get the result
    return await _get_instances_config()

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
    instance = instance_manager.get_instance(instance_name)
    check_instance_exists(instance, instance_name)
    
    # Get instance stats
    instance_stats_response = instance_manager.get_instance_stats(instance_name)
    if instance_stats_response.get("status") == "success":
        instance_stats = instance_stats_response.get("instance", {})
    else:
        instance_stats = {}
    
    # Get instance config as dict
    instance_config = instance.model_dump()
    
    # Remove sensitive information
    if "api_key" in instance_config:
        instance_config["api_key"] = "**REDACTED**"
    
    # Get health information
    health_status = "unknown"
    health_details = {}
    
    # Generate basic health status from stats
    if instance_stats:
        if instance_stats.get("error_count", 0) > 0:
            health_status = "error"
            health_details["error_count"] = instance_stats.get("error_count", 0)
            health_details["recent_errors"] = instance_stats.get("recent_errors", [])
        elif instance_stats.get("rate_limited_count", 0) > 0:
            health_status = "rate_limited"
            health_details["rate_limited_count"] = instance_stats.get("rate_limited_count", 0)
        else:
            health_status = "healthy"
        
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
                health_status = "error"
        except Exception as e:
            health_details["connection_test"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
            health_status = "error"
    
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
    Add a new instance to the proxy service at runtime.
    
    Note: This change is not persisted to the configuration file and will be lost on service restart.
    
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
    Add multiple instances to the proxy service at runtime.
    
    Note: These changes are not persisted to the configuration file and will be lost on service restart.
    
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
    Remove an instance from the proxy service at runtime.
    
    Note: This change is not persisted to the configuration file and the instance will be 
    restored on service restart.
    
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
    
    This endpoint allows updating instance configuration attributes at runtime,
    such as priority, weight, max_tpm, etc. You can provide only the fields
    you want to update.
    
    Sensitive attributes like api_key are only updated if explicitly provided.
    If api_base or provider_type are changed, the instance client will be re-initialized.
    
    Note: These changes are not persisted to the configuration file and will be lost on service restart.
    
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