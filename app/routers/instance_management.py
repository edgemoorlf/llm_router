"""Instance management API router."""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Body
import time

from app.config import InstanceConfig
from app.instance.manager import instance_manager
from app.services.instance_service import instance_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/instances",
    tags=["Instance Management"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def get_instances_config(
    detailed: bool = Query(False, description="Whether to include detailed instance information"),
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
        detailed: If true, returns more detailed runtime information about instances
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
    from app.config import config_loader
    
    try:
        # Base data: either get from instance_manager or config
        if detailed:
            all_instances = list(instance_manager.get_all_instances().values())
            
            # Start with full list for filtering
            filtered_instances = all_instances
            
            # Apply filters
            if provider_type:
                filtered_instances = [i for i in filtered_instances if i.provider_type == provider_type]
                
            if status:
                filtered_instances = [i for i in filtered_instances if i.status == status]
                
            if min_tpm is not None:
                filtered_instances = [i for i in filtered_instances if i.instance_stats.current_tpm >= min_tpm]
                
            if max_tpm is not None:
                filtered_instances = [i for i in filtered_instances if i.max_tpm <= max_tpm]
                
            if model_support:
                filtered_instances = [
                    i for i in filtered_instances if 
                    (not i.supported_models or model_support in i.supported_models)
                ]
            
            # Sort the instances
            if sort_by:
                reverse = sort_dir.lower() == "desc"
                if sort_by == "name":
                    filtered_instances.sort(key=lambda i: i.name, reverse=reverse)
                elif sort_by == "status":
                    filtered_instances.sort(key=lambda i: i.status, reverse=reverse)
                elif sort_by == "current_tpm":
                    filtered_instances.sort(key=lambda i: i.instance_stats.current_tpm, reverse=reverse)
                elif sort_by == "priority":
                    filtered_instances.sort(key=lambda i: i.priority, reverse=reverse)
            
            # Calculate total before pagination
            total_filtered = len(filtered_instances)
            
            # Apply pagination
            if limit is not None:
                paginated_instances = filtered_instances[offset:offset+limit]
            else:
                paginated_instances = filtered_instances
                
            # Format the instance data
            instances_list = [
                {
                    "name": instance.name,
                    "provider_type": instance.provider_type,
                    "api_base": instance.api_base,
                    "priority": instance.priority,
                    "weight": instance.weight,
                    "supported_models": instance.supported_models,
                    "max_tpm": instance.max_tpm,
                    "status": instance.status,
                    "current_tpm": instance.instance_stats.current_tpm,
                    "tpm_usage_percent": round((instance.instance_stats.current_tpm / instance.max_tpm) * 100, 2) if instance.max_tpm > 0 else 0,
                    "max_input_tokens": instance.max_input_tokens,
                    "last_used": instance.last_used
                }
                for instance in paginated_instances
            ]
            
            # Create the applied_filters object for response metadata
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
        else:
            # Get from configuration
            configured_instances = config_loader.get_instances()
            
            # Filter by provider_type if specified
            if provider_type:
                configured_instances = [i for i in configured_instances if i.provider_type == provider_type]
                
            # Filter by model_support if specified
            if model_support:
                configured_instances = [
                    i for i in configured_instances if 
                    (not i.supported_models or model_support in i.supported_models)
                ]
                
            # Sort the instances
            if sort_by:
                reverse = sort_dir.lower() == "desc"
                if sort_by == "name":
                    configured_instances.sort(key=lambda i: i.name, reverse=reverse)
                elif sort_by == "priority":
                    configured_instances.sort(key=lambda i: i.priority, reverse=reverse)
                
            # Calculate total before pagination
            total_filtered = len(configured_instances)
            
            # Apply pagination
            if limit is not None:
                paginated_instances = configured_instances[offset:offset+limit]
            else:
                paginated_instances = configured_instances
                
            # Convert to dict format
            instances_list = []
            for instance in paginated_instances:
                instance_dict = instance.to_dict(exclude_sensitive=True)
                instances_list.append(instance_dict)
                
            # Create the applied_filters object for response metadata
            applied_filters = {
                "provider_type": provider_type,
                "model_support": model_support,
                "sort_by": sort_by,
                "sort_dir": sort_dir
            }
            # Remove None values
            applied_filters = {k: v for k, v in applied_filters.items() if v is not None}
            
            return {
                "status": "success",
                "message": f"Retrieved {len(instances_list)} configured instances",
                "timestamp": int(time.time()),
                "data": {
                    "instances": instances_list,
                    "total_available": len(config_loader.get_instances()),
                    "total_filtered": total_filtered,
                    "count": len(instances_list),
                    "pagination": {
                        "offset": offset,
                        "limit": limit
                    },
                    "filters": applied_filters,
                    "is_configured_view": True
                }
            }
    except Exception as e:
        logger.error(f"Error getting instance configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error getting instance configuration: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.get("/{instance_name}")
async def get_instance_config(instance_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific instance.
    
    Args:
        instance_name: Name of the instance
        
    Returns:
        Standardized response with instance configuration data
    """
    try:
        # Check if the instance exists
        instance = instance_manager.get_instance(instance_name)
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "error",
                    "message": f"Instance '{instance_name}' not found",
                    "timestamp": int(time.time()),
                    "data": None
                }
            )
        
        # Return the instance configuration
        config = {
            "name": instance.name,
            "provider_type": instance.provider_type,
            "api_base": instance.api_base,
            "api_version": instance.api_version,
            "priority": instance.priority,
            "weight": instance.weight,
            "max_tpm": instance.max_tpm,
            "max_input_tokens": instance.max_input_tokens,
            "supported_models": instance.supported_models,
            "model_deployments": instance.model_deployments
        }
        
        return {
            "status": "success",
            "message": f"Retrieved configuration for instance '{instance_name}'",
            "timestamp": int(time.time()),
            "data": config
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions with standardized format
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail if isinstance(e.detail, dict) else {
                "status": "error",
                "message": str(e.detail),
                "timestamp": int(time.time()),
                "data": None
            }
        )
    except Exception as e:
        logger.error(f"Error getting instance config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error getting instance config: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.post("/add")
async def add_instance(instance_config: InstanceConfig) -> Dict[str, Any]:
    """
    Add a new instance to the proxy service at runtime.
    
    Note: This change is not persisted to the configuration file and will be lost on service restart.
    
    Args:
        instance_config: Configuration for the new instance
        
    Returns:
        Standardized response with status information about the newly added instance
    """
    try:
        result = await instance_service.add_instance(instance_config)
        return {
            "status": "success",
            "message": f"Instance '{instance_config.name}' added successfully",
            "timestamp": int(time.time()),
            "data": result
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions directly but format them properly
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "status": "error",
                "message": str(e.detail),
                "timestamp": int(time.time()),
                "data": None
            }
        )
    except Exception as e:
        logger.error(f"Error adding instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error adding instance: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.post("/add-many")
async def add_many_instances(instances_config: List[InstanceConfig]) -> Dict[str, Any]:
    """
    Add multiple instances to the proxy service at runtime.
    
    Note: These changes are not persisted to the configuration file and will be lost on service restart.
    
    Args:
        instances_config: List of configurations for the new instances
        
    Returns:
        Standardized response with information about successfully added, skipped, and failed instances
    """
    try:
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
    except HTTPException as e:
        # Re-raise HTTP exceptions directly but format them properly
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "status": "error",
                "message": str(e.detail),
                "timestamp": int(time.time()),
                "data": None
            }
        )
    except Exception as e:
        logger.error(f"Error adding instances: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error adding instances: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.delete("/{instance_name}")
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
    try:
        result = await instance_service.remove_instance(instance_name)
        return {
            "status": "success",
            "message": f"Instance '{instance_name}' removed successfully",
            "timestamp": int(time.time()),
            "data": {"name": instance_name, "removed": result}
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions directly but format them properly
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "status": "error",
                "message": str(e.detail),
                "timestamp": int(time.time()),
                "data": None
            }
        )
    except Exception as e:
        logger.error(f"Error removing instance {instance_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error removing instance '{instance_name}': {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.patch("/{instance_name}")
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
    try:
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
    except HTTPException as e:
        # Re-raise HTTP exceptions directly but format them properly
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "status": "error",
                "message": str(e.detail),
                "timestamp": int(time.time()),
                "data": None
            }
        )
    except Exception as e:
        logger.error(f"Error updating instance {instance_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error updating instance '{instance_name}': {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        ) 