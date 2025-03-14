"""Configuration API router."""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import time

from app.config import config_loader

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
    """
    Get the current configuration (excluding secrets).
    
    Returns:
        Standardized response with the current configuration
    """
    try:
        config_data = config_loader.to_dict()
        return {
            "status": "success",
            "message": "Configuration retrieved successfully",
            "timestamp": int(time.time()),
            "data": config_data
        }
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error retrieving configuration: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.post("/reload")
async def reload_config() -> Dict[str, Any]:
    """
    Reload configuration from disk.
    
    This endpoint reloads the configuration files and reinitializes all instances.
    """
    try:
        # Reload configuration
        config = config_loader.reload()
        
        # Reload instance manager
        from app.instance.manager import instance_manager
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
    except Exception as e:
        logger.error(f"Error reloading configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error reloading configuration: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.post("/update")
async def update_config(update_request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Update configuration values.
    
    This endpoint allows updating certain configuration parameters at runtime.
    """
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
            "timestamp": int(time.time()),
            "data": {
                "updated_fields": updated_fields
            }
        }
    except ValueError as e:
        logger.error(f"Invalid configuration update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": str(e),
                "timestamp": int(time.time()),
                "data": None
            }
        )
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error updating configuration: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        ) 