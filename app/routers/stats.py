"""API 服务统计路由"""
from typing import Dict, Any, List, Optional
import time
import logging
import os
from fastapi import APIRouter, Query, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from app.instance.service_stats import service_stats
from app.instance.instance_context import instance_manager
from app.services.azure_openai import azure_openai_service
from app.services.generic_openai import generic_openai_service
from app.instance.monitor import InstanceMonitor
from app.services.instance_service import instance_service
from app.config import config_loader
from app.errors.utils import handle_router_errors

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/stats",
    tags=["Statistics"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
@handle_router_errors("retrieving service statistics")
async def get_stats(window: Optional[int] = Query(None, description="时间窗口（分钟），默认为 5 分钟")):
    """获取服务总体统计数据"""
    return service_stats.get_metrics(window)

@router.get("/windows")
@handle_router_errors("retrieving multi-window statistics")
async def get_multi_window_stats(
    windows: str = Query("5,15,30,60", description="逗号分隔的时间窗口列表（分钟），例如 '5,15,30,60'")
):
    """获取多个时间窗口的服务统计数据"""
    try:
        window_list = [int(w.strip()) for w in windows.split(",") if w.strip()]
        return service_stats.get_multiple_window_metrics(window_list)
    except ValueError:
        return {"error": "无效的时间窗口格式，请使用逗号分隔的整数，例如 '5,15,30,60'", "status": "error"}

@router.get("/instances")
@handle_router_errors("retrieving instance statistics")
async def get_instance_stats(
    background_tasks: BackgroundTasks,
    status_filter: Optional[str] = Query(None, description="Filter by instance status (healthy, error, rate_limited)"),
    provider_type: Optional[str] = Query(None, description="Filter by provider type (azure, generic)"),
    min_tpm: Optional[int] = Query(None, description="Filter instances with current TPM >= this value"),
    max_tpm: Optional[int] = Query(None, description="Filter instances with max TPM <= this value"),
    model_support: Optional[str] = Query(None, description="Filter instances that support this model"),
    sort_by: Optional[str] = Query(None, description="Field to sort by (name, status, current_tpm, priority)"),
    sort_dir: Optional[str] = Query("asc", description="Sort direction (asc, desc)"),
    test_connection: bool = Query(False, description="Test connection to all instances")
):
    """
    Get detailed statistics and health information for all instances.
    
    This endpoint combines:
    - Instance statistics (usage, performance)
    - Health status (healthy, errors, rate limited)
    - Connectivity information
    
    Args:
        background_tasks: BackgroundTasks for async operations
        status_filter: Filter by instance status (healthy, error, rate_limited)
        provider_type: Filter by provider type (azure, generic)
        min_tpm: Filter instances with current TPM >= this value
        max_tpm: Filter instances with max TPM <= this value
        model_support: Filter instances that support this model
        sort_by: Field to sort by (name, status, current_tpm, priority)
        sort_dir: Sort direction (asc, desc)
        test_connection: Test connection to all instances
    
    Returns:
        Comprehensive statistics and health information for all instances
    """
    # Get all instance stats
    try:
        stats_response = instance_manager.get_instance_stats()
        logger.debug(f"stats_response type: {type(stats_response)}")
        
        # Extract the actual instance stats from the response
        if not isinstance(stats_response, dict):
            logger.error(f"Unexpected response type from get_instance_stats: {type(stats_response)}")
            return {
                "status": "error",
                "message": "Failed to retrieve instance statistics: unexpected response type",
                "timestamp": int(time.time())
            }
        
        if "instances" not in stats_response:
            logger.error(f"Missing 'instances' key in response: {list(stats_response.keys())}")
            return {
                "status": "error",
                "message": "Failed to retrieve instance statistics: missing instances data",
                "timestamp": int(time.time())
            }
        
        instance_stats = stats_response["instances"]
        logger.debug(f"instance_stats type: {type(instance_stats)}, length: {len(instance_stats)}")
        
        # Get all instance configs
        instances = instance_manager.get_all_instances()
        logger.debug(f"Raw instances type: {type(instances)}")
        
        if instances is None:
            logger.error("get_all_instances() returned None")
            return {
                "status": "error", 
                "message": "Failed to retrieve instance configurations: null response",
                "timestamp": int(time.time())
            }
            
        # Print the actual content if it's not a dict or it's a string
        if not isinstance(instances, dict):
            if isinstance(instances, str):
                logger.error(f"instances is a string: '{instances}'")
            else:
                logger.error(f"instances is not a dict: {instances}")
            
            return {
                "status": "error",
                "message": f"Failed to process instances: expected dict, got {type(instances).__name__}",
                "timestamp": int(time.time())
            }
        
        # Check dict contents without any dict comprehension
        instance_configs = {}
        for key, value in instances.items():
            try:
                logger.debug(f"Processing instance key: {key}, value type: {type(value)}")
                instance_configs[key] = value
            except Exception as e:
                logger.exception(f"Error processing instance {key}: {e}")
                return {
                    "status": "error",
                    "message": f"Error processing instance {key}: {str(e)}",
                    "timestamp": int(time.time())
                }
    except Exception as e:
        logger.exception(f"Unexpected error in initial stats processing: {e}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "timestamp": int(time.time())
        }
    
    # Enhance stats with health information
    try:
        for i, stats in enumerate(instance_stats):
            instance_name = stats.get("name")
            
            # Skip if instance doesn't exist anymore
            if instance_name not in instance_configs:
                continue
            
            # Get the instance config
            instance_config = instance_configs[instance_name]
            
            # Add basic health status
            health_status = "unknown"
            if stats.get("error_count", 0) > 0:
                health_status = "error"
            elif stats.get("rate_limited_count", 0) > 0:
                health_status = "rate_limited"
            else:
                health_status = "healthy"
            
            # Add health status to stats
            stats["health_status"] = health_status
            
            # Add provider type from config
            stats["provider_type"] = instance_config.provider_type
            
            # Calculate utilization percentage
            if instance_config.max_tpm > 0 and "current_tpm" in stats:
                stats["utilization_percentage"] = round((stats["current_tpm"] / instance_config.max_tpm) * 100, 1)
            else:
                stats["utilization_percentage"] = 0
            
            # Enhanced error information
            if "errors" in stats and stats["errors"]:
                stats["error_details"] = stats["errors"]
            
    except Exception as e:
        logger.exception(f"Error processing instances: {e}")
        return {
            "status": "error",
            "message": f"Failed to process instance statistics: {str(e)}",
            "timestamp": int(time.time())
        }
    
    # Test connection if requested
    if test_connection:
        try:
            for stats in instance_stats:
                instance_name = stats.get("name")
                
                # Skip if instance doesn't exist
                if instance_name not in instance_configs:
                    continue
                
                try:
                    # Don't wait for the result, just add to background tasks
                    background_tasks.add_task(
                        instance_service.test_connectivity,
                        instance_name
                    )
                    stats["connection_testing"] = True
                except Exception as e:
                    logger.error(f"Error testing connection for instance {instance_name}: {str(e)}")
                    stats["connection_error"] = str(e)
        except Exception as e:
            logger.exception(f"Error in test_connection: {e}")
            # Don't return error here, just log it and continue
    
    # Apply filters
    try:
        filtered_stats = instance_stats
        
        # Filter by status
        if status_filter:
            filtered_stats = [
                s for s in filtered_stats 
                if s.get("health_status") == status_filter
            ]
        
        # Filter by provider type
        if provider_type:
            filtered_stats = [
                s for s in filtered_stats 
                if s.get("provider_type") == provider_type
            ]
        
        # Filter by min TPM
        if min_tpm is not None:
            filtered_stats = [
                s for s in filtered_stats 
                if s.get("current_tpm", 0) >= min_tpm
            ]
        
        # Filter by max TPM
        if max_tpm is not None:
            filtered_stats = [
                s for s in filtered_stats 
                if instance_configs.get(s.get("name")).max_tpm <= max_tpm
                if s.get("name") in instance_configs
            ]
        
        # Filter by model support
        if model_support:
            filtered_stats = [
                s for s in filtered_stats 
                if s.get("name") in instance_configs and
                model_support in instance_configs[s.get("name")].supported_models
            ]
        
        # Sort the results
        if sort_by:
            reverse = sort_dir.lower() == "desc"
            
            if sort_by == "name":
                filtered_stats = sorted(filtered_stats, key=lambda s: s.get("name", ""), reverse=reverse)
            elif sort_by == "status":
                filtered_stats = sorted(filtered_stats, key=lambda s: s.get("health_status", "unknown"), reverse=reverse)
            elif sort_by == "current_tpm":
                filtered_stats = sorted(filtered_stats, key=lambda s: s.get("current_tpm", 0), reverse=reverse)
            elif sort_by == "priority":
                # Sort by priority from instance config
                filtered_stats = sorted(
                    filtered_stats, 
                    key=lambda s: instance_configs.get(s.get("name")).priority if s.get("name") in instance_configs else 999,
                    reverse=reverse
                )
        
        # Calculate summary statistics
        total_instances = len(filtered_stats)
        healthy_instances = len([s for s in filtered_stats if s.get("health_status") == "healthy"])
        error_instances = len([s for s in filtered_stats if s.get("health_status") == "error"])
        rate_limited_instances = len([s for s in filtered_stats if s.get("health_status") == "rate_limited"])
        
        # Calculate total TPM across all instances
        total_current_tpm = sum(s.get("current_tpm", 0) for s in filtered_stats)
        total_max_tpm = sum(instance_configs.get(s.get("name")).max_tpm 
                          for s in filtered_stats 
                          if s.get("name") in instance_configs)
        
        # Overall utilization
        overall_utilization = round((total_current_tpm / total_max_tpm) * 100, 1) if total_max_tpm > 0 else 0
        
        # Build the response
        result = {
            "status": "success",
            "timestamp": int(time.time()),
            "summary": {
                "total_instances": total_instances,
                "healthy_instances": healthy_instances,
                "error_instances": error_instances,
                "rate_limited_instances": rate_limited_instances,
                "total_current_tpm": total_current_tpm,
                "total_max_tpm": total_max_tpm,
                "overall_utilization_percentage": overall_utilization
            },
            "instances": filtered_stats
        }
        
        return result
    except Exception as e:
        logger.exception(f"Error in filtering or building response: {e}")
        return {
            "status": "error",
            "message": f"Error processing instance statistics: {str(e)}",
            "timestamp": int(time.time())
        }

@router.get("/health")
@handle_router_errors("retrieving health status")
async def get_overall_health():
    """
    Get overall health status summary for all instances.
    
    This endpoint provides a simple overview of the health status 
    of all instances, including counts and percentages.
    
    Returns:
        Standardized response with health summary information
    """
    try:
        # Get all instance stats
        stats_response = instance_manager.get_instance_stats()
        logger.debug(f"health: stats_response type: {type(stats_response)}")
        
        # Extract the actual instance stats from the response
        if not isinstance(stats_response, dict):
            logger.error(f"health: Unexpected response type from get_instance_stats: {type(stats_response)}")
            return {
                "status": "error",
                "message": "Failed to retrieve instance statistics: unexpected response type",
                "timestamp": int(time.time())
            }
        
        if "instances" not in stats_response:
            available_keys = list(stats_response.keys()) if hasattr(stats_response, 'keys') else "no keys"
            logger.error(f"health: Missing 'instances' key in response. Available keys: {available_keys}")
            return {
                "status": "error",
                "message": "Failed to retrieve instance statistics: missing instances data",
                "timestamp": int(time.time())
            }
        
        instance_stats = stats_response["instances"]
        
        # Validate instance_stats is a list
        if not isinstance(instance_stats, list):
            logger.error(f"health: Unexpected type for instance_stats: {type(instance_stats)}")
            return {
                "status": "error",
                "message": "Failed to process instance statistics: expected a list",
                "timestamp": int(time.time())
            }
        
        # Calculate health counts
        total_instances = len(instance_stats)
        healthy_instances = 0
        error_instances = 0
        rate_limited_instances = 0
        unknown_instances = 0
        
        # Process each instance with robust error handling
        for i, stats in enumerate(instance_stats):
            try:
                if not isinstance(stats, dict):
                    logger.warning(f"health: Instance stats at index {i} is not a dict: {type(stats)}")
                    unknown_instances += 1
                    continue
                    
                if stats.get("error_count", 0) > 0:
                    error_instances += 1
                elif stats.get("rate_limited_count", 0) > 0:
                    rate_limited_instances += 1
                elif "last_used" in stats or "current_tpm" in stats:
                    healthy_instances += 1
                else:
                    unknown_instances += 1
            except Exception as e:
                logger.warning(f"health: Error processing instance at index {i}: {e}")
                unknown_instances += 1
        
        # Determine overall status
        overall_status = "healthy"
        if total_instances == 0:
            overall_status = "warning"
        elif healthy_instances == 0:
            overall_status = "critical"
        elif healthy_instances < total_instances * 0.5:
            overall_status = "degraded"
        
        # Calculate percentages safely
        try:
            healthy_percent = round((healthy_instances / total_instances) * 100) if total_instances > 0 else 0
            error_percent = round((error_instances / total_instances) * 100) if total_instances > 0 else 0
            rate_limited_percent = round((rate_limited_instances / total_instances) * 100) if total_instances > 0 else 0
            unknown_percent = round((unknown_instances / total_instances) * 100) if total_instances > 0 else 0
        except Exception as e:
            logger.warning(f"health: Error calculating percentages: {e}")
            healthy_percent = error_percent = rate_limited_percent = unknown_percent = 0
        
        # Build the response
        result = {
            "status": "success",
            "timestamp": int(time.time()),
            "health_summary": {
                "overall_status": overall_status,
                "total_instances": total_instances,
                "healthy": {
                    "count": healthy_instances,
                    "percentage": healthy_percent
                },
                "error": {
                    "count": error_instances,
                    "percentage": error_percent
                },
                "rate_limited": {
                    "count": rate_limited_instances,
                    "percentage": rate_limited_percent
                },
                "unknown": {
                    "count": unknown_instances,
                    "percentage": unknown_percent
                }
            }
        }
        
        return result
    except Exception as e:
        logger.exception(f"health: Unexpected error: {e}")
        return {
            "status": "error",
            "message": f"Error getting health status: {str(e)}",
            "timestamp": int(time.time())
        }

@router.post("/reset")
@handle_router_errors("resetting statistics")
async def reset_stats():
    """重置全局服务统计（仅用于测试）"""
    global service_stats
    result = service_stats.reset()
    if result["status"] == "success" and "stats" in result:
        # Update the global singleton
        service_stats = result["stats"]
        return {"status": "success", "message": "服务统计已重置"}
    else:
        return result
