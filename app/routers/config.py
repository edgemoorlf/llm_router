"""Configuration API router."""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.instance.instance_context import instance_manager
import time

from app.config import config_loader
from app.errors.utils import handle_router_errors

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
@handle_router_errors("retrieving configuration")
async def get_config() -> Dict[str, Any]:
    """
    Get the current configuration (excluding secrets).
    
    Returns:
        Standardized response with the current configuration
    """
    config_data = config_loader.to_dict()
    return {
        "status": "success",
        "message": "Configuration retrieved successfully",
        "timestamp": int(time.time()),
        "data": config_data
    }

@router.post("/reload")
@handle_router_errors("reloading configuration")
async def reload_config() -> Dict[str, Any]:
    """
    Reload configuration from disk.
    
    This endpoint reloads the configuration files and reinitializes all instances.
    """
    # Reload configuration
    config = config_loader.reload()
    
    # Reload instance manager        
    instance_manager.reload_config()
    
    # Get the number of instances after reload
    instance_count = len(instance_manager.get_all_instances())
    
    return {
        "status": "success",
        "message": "Configuration reloaded successfully",
        "timestamp": int(time.time()),
        "data": {
            "instances_count": instance_count,
            "config_version": config.version
        }
    }

@router.post("/update")
@handle_router_errors("updating configuration")
async def update_config(update_request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Update configuration values.
    
    This endpoint allows updating certain configuration parameters at runtime.
    """
    updated_fields = {}
    
    # Update routing strategy if provided
    if update_request.routing_strategy is not None:
        valid_strategies = ["priority", "round_robin", "weighted_random"]
        if update_request.routing_strategy not in valid_strategies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": f"Invalid routing strategy. Must be one of: {', '.join(valid_strategies)}",
                    "timestamp": int(time.time())
                }
            )
        
        config_loader.config.routing.strategy = update_request.routing_strategy
        updated_fields["routing_strategy"] = update_request.routing_strategy
    
    # Update stats window if provided
    if update_request.stats_window_minutes is not None:
        if update_request.stats_window_minutes < 1 or update_request.stats_window_minutes > 1440:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error", 
                    "message": "Stats window must be between 1 and 1440 minutes",
                    "timestamp": int(time.time())
                }
            )
        
        config_loader.config.monitoring.stats_window_minutes = update_request.stats_window_minutes
        updated_fields["stats_window_minutes"] = update_request.stats_window_minutes
    
    # If no fields were updated, return an error
    if not updated_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "No valid fields provided for update",
                "timestamp": int(time.time())
            }
        )
    
    # Save the updated configuration
    config_loader.save_config(config_loader.config)
    
    return {
        "status": "success",
        "message": "Configuration updated successfully",
        "timestamp": int(time.time()),
        "data": {
            "updated_fields": updated_fields
        }
    } 