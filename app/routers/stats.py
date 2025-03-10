"""API 服务统计路由"""
from typing import Dict, Any, List, Optional
import time
import logging
import os
from fastapi import APIRouter, Query, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from app.instance.service_stats import service_stats
from app.instance.manager import instance_manager
from app.instance.safe_stats import safe_service_stats
from app.instance.safe_manager import safe_instance_manager
from app.instance.instance_tester import perform_instance_test
from app.services.azure_openai import azure_openai_service
from app.services.generic_openai import generic_openai_service
from app.config import config_loader

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/stats",
    tags=["Statistics"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def get_stats(window: Optional[int] = Query(None, description="时间窗口（分钟），默认为 5 分钟")):
    """获取服务总体统计数据"""
    return safe_service_stats.get_metrics(window)

@router.get("/windows")
async def get_multi_window_stats(
    windows: str = Query("5,15,30,60", description="逗号分隔的时间窗口列表（分钟），例如 '5,15,30,60'")
):
    """获取多个时间窗口的服务统计数据"""
    try:
        window_list = [int(w.strip()) for w in windows.split(",") if w.strip()]
        return safe_service_stats.get_multiple_window_metrics(window_list)
    except ValueError:
        return {"error": "无效的时间窗口格式，请使用逗号分隔的整数，例如 '5,15,30,60'", "status": "error"}

@router.get("/instances")
async def get_instance_stats(
    background_tasks: BackgroundTasks,
    knockknock: bool = Query(False, description="主动测试那些标记为健康但长时间未接收负载的实例")
):
    """
    获取所有实例的详细统计数据和状态信息。
    
    此端点合并了以前的 /stats/instances 和 /stats/status 功能，提供实例的完整状态和统计信息。
    
    参数:
        knockknock: 如果设置为 true，将对标记为健康但在最近时间窗口内未接收负载的实例发送测试请求（后台运行）
    
    返回:
        关于所有 API 实例的详细信息，包括:
        - 基本状态（健康、速率受限、错误）
        - 请求和令牌使用统计
        - 错误率和类型
        - 配置设置
        - 上次使用时间
    """
    try:
        # Get stats from instance manager
        instance_stats_response = safe_instance_manager.get_instance_stats()
        
        # Ensure we have a proper list of dictionaries
        if isinstance(instance_stats_response, dict) and "instances" in instance_stats_response:
            # If it's wrapped in a response object
            instance_stats = instance_stats_response.get("instances", [])
        elif isinstance(instance_stats_response, list):
            # If it's already a list as expected
            instance_stats = instance_stats_response
        else:
            # Default to empty list if unexpected format
            logger.error(f"Unexpected format from get_instance_stats: {type(instance_stats_response)}")
            instance_stats = []
        
        # Get additional status data from services 
        try:
            azure_instances = await azure_openai_service.get_instances_status()
            generic_instances = await generic_openai_service.get_instances_status()
            
            # Create a mapping of instance name to service status
            service_status_map = {i.get("name", ""): i for i in azure_instances + generic_instances if isinstance(i, dict)}
        except Exception as e:
            logger.error(f"Error getting service status: {str(e)}")
            service_status_map = {}
        
        # Get the time window for activity threshold
        try:
            config = config_loader.get_config()
            window_minutes = config.monitoring.stats_window_minutes if config else 5
        except Exception:
            window_minutes = 5
            
        inactivity_threshold = time.time() - (window_minutes * 60)
        
        # Identify instances that need testing
        instances_to_test = []
        for i, instance_data in enumerate(instance_stats):
            if not isinstance(instance_data, dict):
                continue
                
            # Add a test_status field to track testing in the response
            instance_data["test_status"] = "not_tested"
            
            instance_name = instance_data.get("name", "")
            if not instance_name:
                continue
                
            if (
                instance_data.get("status") == "healthy" and
                instance_data.get("last_used", 0) < inactivity_threshold
            ):
                instances_to_test.append((i, instance_name, instance_data))
        
        # Test instances based on parameters
        if knockknock and instances_to_test:
            # Schedule background tests - these won't be reflected in the response
            logger.info(f"Scheduling background tests for {len(instances_to_test)} instances")
            for idx, instance_name, instance_data in instances_to_test:
                # Mark that we're scheduling a test but results will come later
                instance_stats[idx]["test_status"] = "scheduled"
                background_tasks.add_task(perform_instance_test, instance_name)
        
        # Get routing strategy
        try:
            config = config_loader.get_config()
            routing_strategy = config.routing.strategy if config else os.getenv("API_ROUTING_STRATEGY", "failover")
        except Exception:
            routing_strategy = os.getenv("API_ROUTING_STRATEGY", "failover")
        
        # Count healthy instances safely
        healthy_count = 0
        for inst in instance_stats:
            if isinstance(inst, dict) and inst.get("status") == "healthy":
                healthy_count += 1
        
        # Sort instances by status: healthy first, then rate_limited, then error
        def status_priority(instance):
            if not isinstance(instance, dict):
                return 3  # Unknown status gets lowest priority
            
            status = instance.get("status", "")
            if status == "healthy":
                return 0  # Highest priority
            elif status == "rate_limited":
                return 1  # Medium priority
            elif status == "error":
                return 2  # Low priority
            else:
                return 3  # Unknown status gets lowest priority
        
        # Get fresh stats after background tests (if any)
        instance_stats = instance_manager.get_instance_stats()
        
        # Sort the instances
        try:
            sorted_instances = sorted(instance_stats, key=status_priority)
        except Exception as e:
            logger.error(f"Error sorting instances: {str(e)}")
            sorted_instances = instance_stats  # Fallback to unsorted if sorting fails
        
        # Combine all data with defensive checking
        result = {
            "status": "success",
            "timestamp": int(time.time()),
            "instances": sorted_instances,
            "total_instances": len(instance_stats),
            "healthy_instances": healthy_count,
            "routing_strategy": routing_strategy,
            "test_info": {
                "instances_tested": len(instances_to_test),
                "test_mode": "background" if knockknock else "none"
            }
        }
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Error getting instances status: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": f"Error getting instances status: {str(e)}"}
        )

@router.get("/health")
async def get_health_status():
    """获取实例健康状态摘要"""
    return safe_instance_manager.get_health_status()

@router.post("/reset")
async def reset_stats():
    """重置全局服务统计（仅用于测试）"""
    result = safe_service_stats.reset()
    if result["status"] == "success" and "stats" in result:
        # Update the global singleton
        global service_stats
        service_stats = result["stats"]
        return {"status": "success", "message": "服务统计已重置"}
    else:
        return result
