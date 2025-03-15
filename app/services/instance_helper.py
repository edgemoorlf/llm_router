"""Helper module for instance filtering, sorting, and related operations."""
import logging
from typing import Dict, List, Any, Optional, Callable

from app.instance.api_instance import APIInstance

logger = logging.getLogger(__name__)

def filter_instances(
    instances: List[APIInstance],
    provider_type: Optional[str] = None,
    status: Optional[str] = None,
    min_tpm: Optional[int] = None,
    max_tpm: Optional[int] = None,
    model_support: Optional[str] = None
) -> List[APIInstance]:
    """
    Filter instances based on various criteria.
    
    Args:
        instances: List of APIInstance objects to filter
        provider_type: Filter by provider type (azure, generic)
        status: Filter by instance status (healthy, error, rate_limited)
        min_tpm: Filter instances with current TPM >= this value
        max_tpm: Filter instances with max TPM <= this value
        model_support: Filter instances that support this model
        
    Returns:
        Filtered list of instances
    """
    filtered = instances
    
    # Apply filters if provided
    if provider_type:
        filtered = [i for i in filtered if i.provider_type == provider_type]
        
    if status:
        filtered = [i for i in filtered if i.status == status]
        
    if min_tpm is not None:
        filtered = [i for i in filtered if i.instance_stats.current_tpm >= min_tpm]
        
    if max_tpm is not None:
        filtered = [i for i in filtered if i.max_tpm <= max_tpm]
        
    if model_support:
        filtered = [
            i for i in filtered if 
            (not i.supported_models or model_support in i.supported_models)
        ]
    
    return filtered

def sort_instances(
    instances: List[APIInstance],
    sort_by: Optional[str] = None,
    sort_dir: str = "asc"
) -> List[APIInstance]:
    """
    Sort instances based on a field.
    
    Args:
        instances: List of APIInstance objects to sort
        sort_by: Field to sort by (name, status, current_tpm, priority)
        sort_dir: Sort direction (asc, desc)
        
    Returns:
        Sorted list of instances
    """
    if not sort_by:
        return instances
        
    reverse = sort_dir.lower() == "desc"
    
    # Define sorting keys
    sort_keys = {
        "name": lambda i: i.name,
        "status": lambda i: i.status,
        "current_tpm": lambda i: i.instance_stats.current_tpm,
        "priority": lambda i: i.priority
    }
    
    # Use default key if not found
    key_func = sort_keys.get(sort_by, lambda i: i.name)
    
    return sorted(instances, key=key_func, reverse=reverse)

def paginate_instances(
    instances: List[APIInstance],
    offset: int = 0,
    limit: Optional[int] = None
) -> List[APIInstance]:
    """
    Apply pagination to a list of instances.
    
    Args:
        instances: List of APIInstance objects to paginate
        offset: Starting index
        limit: Maximum number of items to return, or None for all
        
    Returns:
        Paginated list of instances
    """
    if limit is None:
        return instances[offset:]
    else:
        return instances[offset:offset+limit]

def format_instance_list(instances: List[APIInstance], detailed: bool = True) -> List[Dict[str, Any]]:
    """
    Format a list of instances into a consistent dictionary format.
    
    Args:
        instances: List of APIInstance objects to format
        detailed: Whether to include detailed runtime information
        
    Returns:
        List of formatted instance dictionaries
    """
    if detailed:
        return [
            {
                "name": instance.name,
                "provider_type": instance.provider_type,
                "api_base": instance.api_base,
                "priority": instance.priority,
                "weight": instance.weight,
                "supported_models": instance.supported_models,
                "max_tpm": instance.max_tpm,
                "status": instance.status,
                "current_tpm": instance.instance_stats.current_tpm,
                "tpm_usage_percent": round((instance.instance_stats.current_tpm / instance.max_tpm) * 100, 2) if instance.max_tpm > 0 else 0,
                "max_input_tokens": instance.max_input_tokens,
                "last_used": instance.last_used
            }
            for instance in instances
        ]
    else:
        return [instance.model_dump(exclude={"api_key"}) for instance in instances] 