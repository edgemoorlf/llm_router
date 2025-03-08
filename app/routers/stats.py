"""API 服务统计路由"""
from typing import Dict, Any, List, Optional
import time
import logging
import os
import traceback
from fastapi import APIRouter, Query, HTTPException, status, Request
from fastapi.responses import JSONResponse

from app.instance.service_stats import service_stats
from app.instance.manager import instance_manager
from app.instance.safe_stats import safe_service_stats
from app.instance.safe_manager import safe_instance_manager
from app.services.azure_openai import azure_openai_service
from app.services.generic_openai import generic_openai_service

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
async def get_instance_stats():
    """获取所有实例的详细统计数据"""
    return safe_instance_manager.get_instance_stats()

@router.get("/health")
async def get_health_status():
    """获取实例健康状态摘要"""
    return safe_instance_manager.get_health_status()

@router.get("/status")
async def get_instances_status() -> JSONResponse:
    """
    获取所有 API 实例的状态信息。
    
    返回每个实例的详细信息，包括：
    - 健康状态（健康，速率受限，错误）
    - 当前 TPM 使用情况
    - TPM 限制
    - 优先级和权重设置
    - 错误信息（如适用）
    """
    try:
        # 获取两种服务的状态并合并
        azure_instances = await azure_openai_service.get_instances_status()
        generic_instances = await generic_openai_service.get_instances_status()
        instances = azure_instances + generic_instances
        return JSONResponse(
            content={
                "status": "success",
                "timestamp": int(time.time()),
                "instances": instances,
                "total_instances": len(instances),
                "healthy_instances": len([i for i in instances if i["status"] == "healthy"]),
                "routing_strategy": os.getenv("API_ROUTING_STRATEGY", "failover")
            }
        )
    except Exception as e:
        logger.exception(f"Error getting instances status: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": f"Error getting instances status: {str(e)}"}
        )

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