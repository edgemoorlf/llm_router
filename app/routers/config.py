"""Configuration API router."""
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel

from app.config import config_loader
from app.instance.manager import instance_manager

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
    """Get the current configuration (excluding secrets)."""
    try:
        return config_loader.to_dict()
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving configuration: {str(e)}"
        )

@router.post("/reload")
async def reload_config() -> Dict[str, Any]:
    """Reload configuration from disk."""
    try:
        # Reload configuration
        config = config_loader.reload()
        if not config:
            raise ValueError("Failed to reload configuration")
        
        # Reload instance manager
        instance_manager.reload_config()
        
        return {
            "status": "success",
            "message": "Configuration reloaded successfully",
            "instances": len(instance_manager.instances)
        }
    except Exception as e:
        logger.error(f"Error reloading configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading configuration: {str(e)}"
        )

@router.get("/instances")
async def get_instances_config() -> Dict[str, Any]:
    """Get the current instance configuration (excluding secrets)."""
    try:
        config_dict = config_loader.to_dict()
        return {
            "instances": config_dict.get("instances", []),
            "total": len(config_dict.get("instances", []))
        }
    except Exception as e:
        logger.error(f"Error getting instance configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving instance configuration: {str(e)}"
        )

@router.post("/update")
async def update_config(update: ConfigUpdateRequest = Body(...)) -> Dict[str, Any]:
    """Update configuration settings."""
    try:
        config = config_loader.get_config()
        updated = False
        
        # Update routing strategy
        if update.routing_strategy is not None:
            config.routing.strategy = update.routing_strategy
            updated = True
            
        # Update stats window
        if update.stats_window_minutes is not None:
            config.monitoring.stats_window_minutes = update.stats_window_minutes
            updated = True
            
        if updated:
            # Save configuration
            if config_loader.save_config(config):
                # Reload configuration
                instance_manager.reload_config()
                
                return {
                    "status": "success",
                    "message": "Configuration updated successfully"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to save configuration"
                )
        else:
            return {
                "status": "success",
                "message": "No changes to configuration"
            }
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating configuration: {str(e)}"
        ) 