"""Safe wrapper for service stats to handle API calls safely."""
import logging
import traceback
from typing import Dict, List, Any, Optional

from .service_stats import service_stats

logger = logging.getLogger(__name__)

class SafeServiceStats:
    """
    Safe wrapper for service stats that ensures all API calls are handled
    properly, even if the service_stats object is not fully initialized.
    """
    
    def __init__(self):
        self.stats = service_stats
    
    def get_metrics(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Safely get service metrics.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary with metrics or error message
        """
        try:
            if not self.stats or not hasattr(self.stats, 'get_metrics'):
                logger.error("Service stats not properly initialized")
                return {"status": "error", "message": "Service stats not properly initialized"}
            
            metrics = self.stats.get_metrics(window_minutes)
            return {
                "status": "success",
                "metrics": metrics
            }
        except Exception as e:
            error_msg = f"Error getting service metrics: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}
    
    def get_multiple_window_metrics(self, windows: List[int]) -> Dict[str, Any]:
        """
        Safely get metrics for multiple time windows.
        
        Args:
            windows: List of time windows in minutes
            
        Returns:
            Dictionary with metrics or error message
        """
        try:
            if not self.stats or not hasattr(self.stats, 'get_multiple_window_metrics'):
                logger.error("Service stats not properly initialized")
                return {"status": "error", "message": "Service stats not properly initialized"}
            
            metrics = self.stats.get_multiple_window_metrics(windows)
            return {
                "status": "success",
                "windows": metrics
            }
        except Exception as e:
            error_msg = f"Error getting multi-window metrics: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}
    
    def reset(self) -> Dict[str, Any]:
        """
        Safely reset service stats.
        
        Returns:
            Dictionary with success or error message
        """
        try:
            if not self.stats:
                logger.error("Service stats not properly initialized")
                return {"status": "error", "message": "Service stats not properly initialized"}
            
            # Reset by creating a new instance with the same default window
            default_window = getattr(self.stats, 'default_window_minutes', 5)
            new_stats = type(self.stats)(default_window)
            
            # This doesn't update the global singleton, so caller must update it
            return {
                "status": "success",
                "message": "Service stats object created for reset",
                "stats": new_stats
            }
        except Exception as e:
            error_msg = f"Error resetting service stats: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}

# Create a singleton instance
safe_service_stats = SafeServiceStats() 