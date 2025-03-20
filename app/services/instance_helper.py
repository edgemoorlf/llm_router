"""Helper module for instance filtering, sorting, and related operations."""
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple

from app.models.instance import InstanceConfig, InstanceState

logger = logging.getLogger(__name__)

def filter_instances(
    instances: List[Tuple[InstanceConfig, InstanceState]],
    provider_type: Optional[str] = None,
    status: Optional[str] = None,
    min_tpm: Optional[int] = None,
    max_tpm: Optional[int] = None,
    model_support: Optional[str] = None
) -> List[Tuple[InstanceConfig, InstanceState]]:
    """
    Filter instances based on various criteria.
    
    Args:
        instances: List of (InstanceConfig, InstanceState) tuples to filter
        provider_type: Filter by provider type (azure, generic)
        status: Filter by instance status (healthy, error, rate_limited)
        min_tpm: Filter instances with current TPM >= this value
        max_tpm: Filter instances with max TPM <= this value
        model_support: Filter instances that support this model
        
    Returns:
        Filtered list of (InstanceConfig, InstanceState) tuples
    """
    filtered = instances
    
    # Apply filters if provided
    if provider_type:
        filtered = [(config, state) for config, state in filtered if config.provider_type == provider_type]
        
    if status:
        filtered = [(config, state) for config, state in filtered if state.status == status]
        
    if min_tpm is not None:
        filtered = [(config, state) for config, state in filtered if state.current_tpm >= min_tpm]
        
    if max_tpm is not None:
        filtered = [(config, state) for config, state in filtered if config.max_tpm <= max_tpm]
        
    if model_support:
        filtered = [
            (config, state) for config, state in filtered if 
            (not config.supported_models or model_support in config.supported_models)
        ]
    
    return filtered

def sort_instances(
    instances: List[Tuple[InstanceConfig, InstanceState]],
    sort_by: Optional[str] = None,
    sort_dir: str = "asc"
) -> List[Tuple[InstanceConfig, InstanceState]]:
    """
    Sort instances based on a field.
    
    Args:
        instances: List of (InstanceConfig, InstanceState) tuples to sort
        sort_by: Field to sort by (name, status, current_tpm, priority)
        sort_dir: Sort direction (asc, desc)
        
    Returns:
        Sorted list of (InstanceConfig, InstanceState) tuples
    """
    if not sort_by:
        return instances
        
    reverse = sort_dir.lower() == "desc"
    
    # Define sorting keys
    sort_keys = {
        "name": lambda pair: pair[0].name,
        "status": lambda pair: pair[1].status,
        "current_tpm": lambda pair: pair[1].current_tpm,
        "priority": lambda pair: pair[0].priority
    }
    
    # Use default key if not found
    key_func = sort_keys.get(sort_by, lambda pair: pair[0].name)
    
    return sorted(instances, key=key_func, reverse=reverse)

def paginate_instances(
    instances: List[Tuple[InstanceConfig, InstanceState]],
    offset: int = 0,
    limit: Optional[int] = None
) -> List[Tuple[InstanceConfig, InstanceState]]:
    """
    Apply pagination to a list of instances.
    
    Args:
        instances: List of (InstanceConfig, InstanceState) tuples to paginate
        offset: Starting index
        limit: Maximum number of items to return, or None for all
        
    Returns:
        Paginated list of (InstanceConfig, InstanceState) tuples
    """
    if limit is None:
        return instances[offset:]
    else:
        return instances[offset:offset+limit]

def format_instance_list(instances: List[Tuple[InstanceConfig, InstanceState]], detailed: bool = True) -> List[Dict[str, Any]]:
    """
    Format a list of instances into a consistent dictionary format.
    
    Args:
        instances: List of (InstanceConfig, InstanceState) tuples to format
        detailed: Whether to include detailed runtime information
        
    Returns:
        List of formatted instance dictionaries
    """
    if detailed:
        return [
            {
                "name": config.name,
                "provider_type": config.provider_type,
                "api_base": config.api_base,
                "priority": config.priority,
                "weight": config.weight,
                "supported_models": config.supported_models,
                "max_tpm": config.max_tpm,
                "status": state.status,
                "current_tpm": state.current_tpm,
                "tpm_usage_percent": round((state.current_tpm / config.max_tpm) * 100, 2) if config.max_tpm > 0 else 0,
                "max_input_tokens": config.max_input_tokens,
                "last_used": state.last_used
            }
            for config, state in instances
        ]
    else:
        return [config.model_dump(exclude={"api_key"}) for config, _ in instances]

def filter_instances_optimized(
    configs: List[InstanceConfig],
    provider_type: Optional[str] = None,
    model_support: Optional[str] = None
) -> List[InstanceConfig]:
    """
    Pre-filter instance configs without needing states (avoids Redis calls).
    
    This is an optimization to reduce Redis calls by first filtering instances
    based only on their configurations (which don't require Redis).
    
    Args:
        configs: List of InstanceConfig objects to filter
        provider_type: Filter by provider type (azure, generic)
        model_support: Filter instances that support this model
        
    Returns:
        Filtered list of InstanceConfig objects
    """
    filtered = configs
    
    # Apply filters if provided
    if provider_type:
        filtered = [config for config in filtered if config.provider_type == provider_type]
        
    if model_support:
        filtered = [
            config for config in filtered if 
            (not config.supported_models or model_support in config.supported_models)
        ]
    
    return filtered

def filter_with_states(
    instances: List[Tuple[InstanceConfig, InstanceState]],
    status: Optional[str] = None,
    min_tpm: Optional[int] = None,
    max_tpm: Optional[int] = None
) -> List[Tuple[InstanceConfig, InstanceState]]:
    """
    Filter instances with states based on state-specific criteria.
    
    This should be called after filter_instances_optimized to apply
    state-specific filters.
    
    Args:
        instances: List of (InstanceConfig, InstanceState) tuples to filter
        status: Filter by instance status (healthy, error, rate_limited)
        min_tpm: Filter instances with current TPM >= this value
        max_tpm: Filter instances with max TPM <= this value
        
    Returns:
        Filtered list of (InstanceConfig, InstanceState) tuples
    """
    filtered = instances
    
    # Apply state filters if provided
    if status:
        filtered = [(config, state) for config, state in filtered if state.status == status]
        
    if min_tpm is not None:
        filtered = [(config, state) for config, state in filtered if state.current_tpm >= min_tpm]
        
    if max_tpm is not None:
        filtered = [(config, state) for config, state in filtered if config.max_tpm <= max_tpm]
    
    return filtered 