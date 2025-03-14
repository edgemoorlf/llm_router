"""Health monitoring API router for instances."""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Query
import time
from datetime import datetime, timedelta

from app.instance.manager import instance_manager
from app.services.instance_service import instance_service
from app.instance.monitor import InstanceMonitor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health Monitoring"],
    responses={404: {"description": "Not found"}},
)

@router.get("/instances/{instance_name}")
async def check_instance_health(
    instance_name: str, 
    test_connection: bool = Query(False, description="Test the connection to the instance API")
) -> Dict[str, Any]:
    """
    Check the health of a specific instance.
    
    This endpoint provides a quick health check for a specific instance, 
    with an option to test the connection to the instance API.
    
    For a comprehensive verification including model tests, use the 
    `/instances/verify/{instance_name}` endpoint instead.
    
    Args:
        instance_name: The name of the instance to check
        test_connection: Whether to test the connection to the instance API
        
    Returns:
        Standardized response with health information for the specified instance
        
    Raises:
        HTTPException: If the instance doesn't exist
    """
    try:
        # First check if the instance exists
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
            
        # Get basic health info
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
            "message": f"Health check completed for instance '{instance_name}'",
            "timestamp": int(time.time()),
            "data": health_info
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions directly but use standardized format
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
        logger.error(f"Error checking instance health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error checking instance health: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.get("/instances")
async def get_all_instances_health(
    status_filter: Optional[str] = Query(None, description="Filter instances by status (healthy, unhealthy, rate_limited)"),
    provider_type: Optional[str] = Query(None, description="Filter instances by provider type (azure, generic, etc.)"),
    test_new: bool = Query(False, description="Test newly added instances for connectivity"),
    last_used_minutes: Optional[int] = Query(None, description="Filter instances used within the specified minutes"),
    min_tpm_usage: Optional[float] = Query(None, description="Filter instances with TPM usage percentage above this value (0-100)"),
    max_tpm_usage: Optional[float] = Query(None, description="Filter instances with TPM usage percentage below this value (0-100)"),
    has_errors: Optional[bool] = Query(None, description="Filter instances that have errors (true) or not (false)"),
    sort_by: Optional[str] = Query(None, description="Sort instances by a field (name, status, tpm_usage_percent, last_used)"),
    sort_order: Optional[str] = Query("asc", description="Sort order (asc or desc)")
) -> Dict[str, Any]:
    """
    Get health information for all instances with enhanced filtering options.
    
    Args:
        status_filter: Filter instances by status ("healthy", "unhealthy", "rate_limited")
        provider_type: Filter instances by provider type ("azure", "generic", etc.)
        test_new: Test newly added instances for connectivity
        last_used_minutes: Filter instances used within the specified minutes
        min_tpm_usage: Filter instances with TPM usage percentage above this value (0-100)
        max_tpm_usage: Filter instances with TPM usage percentage below this value (0-100)
        has_errors: Filter instances that have errors (true) or not (false)
        sort_by: Sort instances by a field (name, status, tpm_usage_percent, last_used)
        sort_order: Sort order (asc or desc)
        
    Returns:
        Standardized response with filtered and sorted instance health information
    """
    try:
        all_instances = instance_manager.get_all_instances()
        
        # Get the monitor
        monitor = InstanceMonitor()
        
        # Get instance health information
        instances_health = monitor.get_instance_stats(all_instances)
        
        # Apply standard filters
        if status_filter:
            instances_health = [i for i in instances_health if i["status"] == status_filter]
            
        if provider_type:
            instances_health = [i for i in instances_health if i["provider_type"] == provider_type]
        
        # Apply additional filters if specified
        if last_used_minutes is not None:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=last_used_minutes)
            instances_health = [
                i for i in instances_health 
                if i.get("last_used") and datetime.fromisoformat(i["last_used"]) > cutoff_time
            ]
            
        if min_tpm_usage is not None:
            instances_health = [
                i for i in instances_health 
                if i.get("tpm_usage_percent", 0) >= min_tpm_usage
            ]
            
        if max_tpm_usage is not None:
            instances_health = [
                i for i in instances_health 
                if i.get("tpm_usage_percent", 0) <= max_tpm_usage
            ]
            
        if has_errors is not None:
            instances_health = [
                i for i in instances_health 
                if (i.get("error_count", 0) > 0) == has_errors
            ]
        
        # Sort the results if requested
        if sort_by:
            # Define default value for different fields to handle missing values
            default_values = {
                "name": "",
                "status": "",
                "tpm_usage_percent": 0,
                "last_used": "1970-01-01T00:00:00",
                "error_count": 0,
                "current_tpm": 0
            }
            
            # Get the default value for the sort field
            default = default_values.get(sort_by, None)
            
            # Sort the instances
            reverse = sort_order.lower() == "desc"
            instances_health = sorted(
                instances_health,
                key=lambda x: x.get(sort_by, default),
                reverse=reverse
            )
            
        # Test newly added instances if requested
        if test_new:
            # Get instances that are marked as newly added
            newly_added = [i for i in instances_health if i.get("is_newly_added", False)]
            
            # Test each newly added instance
            for instance_info in newly_added:
                instance_name = instance_info["name"]
                try:
                    # Test the instance and update its info
                    instance = instance_manager.get_instance(instance_name)
                    if instance:
                        test_result = await instance_service.test_instance(instance_name)
                        # Update the instance info with test results
                        instance_info["test_result"] = test_result
                except Exception as e:
                    # If testing fails, add error information
                    instance_info["test_result"] = {
                        "status": "error",
                        "message": f"Error testing instance: {str(e)}"
                    }
        
        # Prepare the filters used for metadata
        applied_filters = {
            "status": status_filter,
            "provider_type": provider_type,
            "last_used_minutes": last_used_minutes,
            "min_tpm_usage": min_tpm_usage,
            "max_tpm_usage": max_tpm_usage,
            "has_errors": has_errors,
            "sort_by": sort_by,
            "sort_order": sort_order
        }
        
        # Remove None values
        applied_filters = {k: v for k, v in applied_filters.items() if v is not None}
        
        return {
            "status": "success",
            "message": f"Retrieved health for {len(instances_health)} instances",
            "timestamp": int(time.time()),
            "data": {
                "total_instances": len(all_instances),
                "filtered_instances": len(instances_health),
                "applied_filters": applied_filters,
                "instances": instances_health
            }
        }
    except Exception as e:
        logger.error(f"Error getting instance health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error getting instance health: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.get("/instances/{instance_name}/stats")
async def get_instance_stats(instance_name: str) -> Dict[str, Any]:
    """
    Get detailed statistics for a specific instance.
    
    Args:
        instance_name: Name of the instance
        
    Returns:
        Standardized response with detailed statistics for the specified instance
    """
    try:
        # Check if instance exists
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
        
        # Get instance monitor
        monitor = InstanceMonitor()
        
        # Create a dictionary with just this instance to pass to get_instance_stats
        instances_dict = {instance_name: instance}
        
        # Get stats for this instance
        stats = monitor.get_instance_stats(instances_dict)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "error",
                    "message": f"No stats found for instance '{instance_name}'",
                    "timestamp": int(time.time()),
                    "data": None
                }
            )
            
        return {
            "status": "success",
            "message": f"Retrieved statistics for instance '{instance_name}'",
            "timestamp": int(time.time()),
            "data": stats[0]  # Should only be one instance
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions but use standardized format
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
        logger.error(f"Error getting instance stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error getting instance stats: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        ) 