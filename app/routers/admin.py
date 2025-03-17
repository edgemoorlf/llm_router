"""Administrative API endpoints for the Azure OpenAI Proxy."""
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from fastapi.responses import JSONResponse

from app.config.config_hierarchy import config_hierarchy
from app.instance.new_manager import NewInstanceManager
from app.instance.instance_context import instance_manager
from app.auth.admin_auth import verify_admin_token
from app.config import config_loader
from app.errors.utils import handle_router_errors
from app.models.instance import InstanceConfig, InstanceState

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
    if not instance_manager.get_instance(instance_name):
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
    # Reload configuration
    instance_manager.reload_config()
    
    # Get number of instances
    instances = instance_manager.get_all_configs()
    
    return {
        "status": "success",
        "message": f"Configuration reloaded successfully with {len(instances)} instances"
    }

@router.get("/instances/{instance_name}/config")
@handle_router_errors("getting instance configuration")
async def get_instance_config(instance_name: str = Path(..., description="Name of the instance")):
    """
    Get the configuration for an instance.
    
    Args:
        instance_name: Name of the instance
    
    Returns:
        The instance configuration
    """
    # Get the configuration
    config = instance_manager.get_instance_config(instance_name)
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance '{instance_name}' not found"
        )
    
    # Convert to dict to ensure serialization
    return config.dict()

@router.get("/instances/{instance_name}/state")
@handle_router_errors("getting instance state")
async def get_instance_state(instance_name: str = Path(..., description="Name of the instance")):
    """
    Get the state for an instance.
    
    Args:
        instance_name: Name of the instance
    
    Returns:
        The instance state
    """
    # Get the state
    state = instance_manager.get_instance_state(instance_name)
    
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance '{instance_name}' not found"
        )
    
    # Convert to dict to ensure serialization
    return state.dict()

@router.post("/instances")
@handle_router_errors("adding instance")
async def add_instance(config: InstanceConfig):
    """
    Add a new instance.
    
    Args:
        config: Instance configuration
    
    Returns:
        Status of the operation
    """
    # Check if instance already exists
    if instance_manager.get_instance_config(config.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Instance '{config.name}' already exists"
        )
    
    # Add the instance
    success = instance_manager.add_instance(config)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add instance '{config.name}'"
        )
    
    return {
        "status": "success",
        "message": f"Instance '{config.name}' added successfully"
    }

@router.put("/instances/{instance_name}/config")
@handle_router_errors("updating instance configuration")
async def update_instance_config(
    instance_name: str = Path(..., description="Name of the instance"),
    config_update: Dict[str, Any] = Body(..., description="Configuration updates")
):
    """
    Update configuration for an instance.
    
    Args:
        instance_name: Name of the instance
        config_update: Configuration properties to update
    
    Returns:
        Status of the operation
    """
    # Check if instance exists
    if not instance_manager.get_instance_config(instance_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance '{instance_name}' not found"
        )
    
    # Update the configuration
    success = instance_manager.update_instance_config(instance_name, config_update)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration for instance '{instance_name}'"
        )
    
    return {
        "status": "success",
        "message": f"Configuration for instance '{instance_name}' updated successfully"
    }

@router.put("/instances/{instance_name}/state")
@handle_router_errors("updating instance state")
async def update_instance_state(
    instance_name: str = Path(..., description="Name of the instance"),
    state_update: Dict[str, Any] = Body(..., description="State updates")
):
    """
    Update state for an instance.
    
    Args:
        instance_name: Name of the instance
        state_update: State properties to update
    
    Returns:
        Status of the operation
    """
    # Check if instance exists
    if not instance_manager.get_instance_state(instance_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance '{instance_name}' not found"
        )
    
    # Update the state
    success = instance_manager.update_instance_state(instance_name, state_update)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update state for instance '{instance_name}'"
        )
    
    return {
        "status": "success",
        "message": f"State for instance '{instance_name}' updated successfully"
    }

@router.delete("/instances/{instance_name}")
@handle_router_errors("deleting instance")
async def delete_instance(instance_name: str = Path(..., description="Name of the instance")):
    """
    Delete an instance.
    
    Args:
        instance_name: Name of the instance
    
    Returns:
        Status of the operation
    """
    # Check if instance exists
    if not instance_manager.get_instance_config(instance_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance '{instance_name}' not found"
        )
    
    # Delete the instance
    success = instance_manager.delete_instance(instance_name)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete instance '{instance_name}'"
        )
    
    return {
        "status": "success",
        "message": f"Instance '{instance_name}' deleted successfully"
    }

@router.get("/instances")
@handle_router_errors("getting all instances")
async def get_all_instances():
    """
    Get all instances.
    
    Returns:
        Dictionary with all instance configurations and states
    """
    # Get all configurations and states
    configs = instance_manager.get_all_configs()
    states = instance_manager.get_all_states()
    
    # Convert to dictionaries for serialization
    configs_dict = {name: config.dict() for name, config in configs.items()}
    states_dict = {name: state.dict() for name, state in states.items()}
    
    # Create response with both configurations and states
    instances = {}
    
    for name in set(list(configs.keys()) + list(states.keys())):
        instances[name] = {
            "config": configs_dict.get(name),
            "state": states_dict.get(name)
        }
    
    return {
        "status": "success",
        "instances": instances,
        "count": len(instances)
    }

@router.post("/instances/{instance_name}/clear-error")
async def clear_instance_error(
    instance_name: str,
    admin_token: str = Depends(verify_admin_token)
):
    """Clear error state for an instance."""
    if instance_manager.clear_instance_error(instance_name):
        return {
            "status": "success",
            "message": f"Error state cleared for instance {instance_name}"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Instance {instance_name} not found or not in error state"
        ) 