"""Safe wrapper for instance manager to handle API calls safely."""
import logging
import traceback
from typing import Dict, List, Any, Optional

from .manager import instance_manager

logger = logging.getLogger(__name__)

class SafeInstanceManager:
    """
    Safe wrapper for instance manager that ensures all API calls are handled
    properly, even if the instance manager is not fully initialized.
    """
    
    def __init__(self):
        self.manager = instance_manager
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """
        Safely get instance statistics.
        
        Returns:
            Dictionary with status and instance stats or error message
        """
        try:
            if not self.manager or not hasattr(self.manager, 'get_instance_stats'):
                logger.error("Instance manager not properly initialized")
                return {"status": "error", "message": "Instance manager not properly initialized"}
            
            stats = self.manager.get_instance_stats()
            return {
                "status": "success",
                "instances": stats,
                "count": len(stats) if stats else 0
            }
        except Exception as e:
            error_msg = f"Error getting instance stats: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Safely get instance health status.
        
        Returns:
            Dictionary with health status or error message
        """
        try:
            if not self.manager or not hasattr(self.manager, 'get_health_status'):
                logger.error("Instance manager not properly initialized")
                return {"status": "error", "message": "Instance manager not properly initialized"}
            
            health_status = self.manager.get_health_status()
            return {
                "status": "success",
                "health_status": health_status
            }
        except Exception as e:
            error_msg = f"Error getting health status: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}

    def get_service_metrics(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Safely get service metrics.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary with service metrics or error message
        """
        try:
            if not self.manager or not hasattr(self.manager, 'get_service_metrics'):
                logger.error("Instance manager not properly initialized")
                return {"status": "error", "message": "Instance manager not properly initialized"}
            
            metrics = self.manager.get_service_metrics(window_minutes)
            return {
                "status": "success",
                "metrics": metrics
            }
        except Exception as e:
            error_msg = f"Error getting service metrics: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}

# Create a singleton instance
safe_instance_manager = SafeInstanceManager() 