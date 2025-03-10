"""API 服务统计路由"""
from typing import Dict, Any, List, Optional
import time
import logging
import os
import traceback
import httpx
from fastapi import APIRouter, Query, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from app.instance.service_stats import service_stats
from app.instance.manager import instance_manager
from app.instance.safe_stats import safe_service_stats
from app.instance.safe_manager import safe_instance_manager
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

# Separate the test logic into its own function that returns the result
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
        # Get the instance from the manager
        instance = instance_manager.instances.get(instance_name)
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

# Create a redirect for the old endpoint
@router.get("/status")
async def get_instances_status_redirect(
    background_tasks: BackgroundTasks,
    knockknock: bool = Query(False, description="主动测试那些标记为健康但长时间未接收负载的实例")
):
    """
    获取所有 API 实例的状态信息（已迁移到 /stats/instances）。
    此端点保留向后兼容性，但已被 /stats/instances 端点取代。
    """
    return await get_instance_stats(background_tasks, knockknock)

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
