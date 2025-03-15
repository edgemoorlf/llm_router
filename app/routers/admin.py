"""Administrative API endpoints for the Azure OpenAI Proxy."""
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from fastapi.responses import JSONResponse

from app.config.config_hierarchy import config_hierarchy
from app.instance.manager import instance_manager
from app.auth.admin_auth import verify_admin_token
from app.config import config_loader
from app.errors.utils import handle_router_errors

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_admin_token)],
)

logger = logging.getLogger(__name__)

class ConfigSourceInfo(BaseModel):
    """Information about a configuration source."""
    name: str
    priority: int
    description: str
    timestamp: Optional[float] = None

class ConfigHierarchyResponse(BaseModel):
    """Configuration hierarchy information response."""
    sources: List[ConfigSourceInfo]
    active_source_count: int

class InstanceSourceInfo(BaseModel):
    """Information about where instance settings come from."""
    sources: Dict[str, str]
    instance: Dict[str, Any]

@router.get("/config/sources", response_model=ConfigHierarchyResponse)
@handle_router_errors("retrieving configuration sources")
async def get_config_sources():
    """Get information about configuration sources."""
    sources_data = config_hierarchy.get_configuration_sources()
    sources = [ConfigSourceInfo(**source) for source in sources_data]
    
    # Count active sources (those that provided some configuration)
    active_count = sum(1 for source in sources if source.timestamp is not None)
    
    return ConfigHierarchyResponse(
        sources=sources,
        active_source_count=active_count
    )

@router.get("/config/instances/{instance_name}/sources", response_model=InstanceSourceInfo)
@handle_router_errors("retrieving instance sources")
async def get_instance_sources(instance_name: str = Path(..., description="Name of the instance")):
    """
    Get information about which source each instance setting comes from.
    
    Args:
        instance_name: Name of the instance
    """
    # First check if instance exists
    instance = instance_manager.get_instance(instance_name)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance '{instance_name}' not found"
        )
        
    # Get source information
    source_info = config_hierarchy.get_instance_source(instance_name)
    
    return InstanceSourceInfo(
        sources=source_info.get('sources', {}),
        instance=source_info.get('instance', {})
    )

@router.get("/config/effective", response_model=Dict[str, Any])
@handle_router_errors("retrieving effective configuration")
async def get_effective_config(
    section: Optional[str] = None, 
    include_secrets: bool = False
):
    """Get the effective configuration after applying the hierarchy."""
    config = config_hierarchy.get_configuration()
    
    # Remove sensitive information
    if 'instances' in config:
        for instance_name, instance in config['instances'].items():
            if 'api_key' in instance:
                # Replace API key with asterisks, keeping first/last 3 chars
                api_key = instance['api_key']
                if api_key and len(api_key) > 6:
                    instance['api_key'] = f"{api_key[:3]}{'*' * (len(api_key) - 6)}{api_key[-3:]}"
                elif api_key:
                    instance['api_key'] = "***"
    
    return config

@router.post("/config/reload")
@handle_router_errors("reloading configuration")
async def reload_config():
    """Reload configuration from all sources."""
    # Reload configuration and initialize instances
    instance_manager.reload_config()
    
    # Get number of instances
    instances = instance_manager.get_all_instances()
    
    return {
        "status": "success",
        "message": f"Configuration reloaded successfully with {len(instances)} instances"
    } 