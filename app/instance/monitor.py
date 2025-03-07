import logging
from typing import Dict, List, Any

from .api_instance import APIInstance

logger = logging.getLogger(__name__)

class InstanceMonitor:
    """Monitors and provides statistics for API instances."""
    
    @staticmethod
    def get_instance_stats(instances: Dict[str, APIInstance]) -> List[Dict[str, Any]]:
        """
        Get statistics for all instances.
        
        Args:
            instances: Dictionary of API instances
            
        Returns:
            List of instance statistics dictionaries
        """
        return [
            {
                "name": instance.name,
                "provider_type": instance.provider_type,
                "status": instance.status,
                "current_tpm": instance.current_tpm,
                "max_tpm": instance.max_tpm,
                "tpm_usage_percent": round((instance.current_tpm / instance.max_tpm) * 100, 2) if instance.max_tpm > 0 else 0,
                "error_count": instance.error_count,
                "last_error": instance.last_error,
                "rate_limited_until": instance.rate_limited_until,
                "priority": instance.priority,
                "weight": instance.weight,
                "last_used": instance.last_used,
                "supported_models": instance.supported_models,
                "model_deployments": instance.model_deployments,
            }
            for instance in instances.values()
        ]
    
    @staticmethod
    def get_health_status(instances: Dict[str, APIInstance]) -> Dict[str, Any]:
        """
        Get health status summary for all instances.
        
        Args:
            instances: Dictionary of API instances
            
        Returns:
            Summary of instance health status
        """
        total = len(instances)
        healthy = sum(1 for i in instances.values() if i.status == "healthy")
        rate_limited = sum(1 for i in instances.values() if i.status == "rate_limited")
        error = sum(1 for i in instances.values() if i.status == "error")
        
        return {
            "total_instances": total,
            "healthy_instances": healthy,
            "rate_limited_instances": rate_limited,
            "error_instances": error,
            "health_percentage": round((healthy / total) * 100, 2) if total > 0 else 0,
            "has_available_capacity": any(
                i.status == "healthy" and i.current_tpm < i.max_tpm * 0.9
                for i in instances.values()
            ),
        } 