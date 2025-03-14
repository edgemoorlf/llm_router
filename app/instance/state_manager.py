"""State management classes for storing and retrieving instance configuration state."""
import os
import json
import time
import logging
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class StateManager(ABC):
    """Abstract base class for state management."""
    
    @abstractmethod
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load instance state from storage.
        
        Returns:
            Dictionary containing instance state, or None if not available/error
        """
        pass
    
    @abstractmethod
    def save_state(self, instances_data: List[Dict[str, Any]], worker_id: str) -> bool:
        """
        Save instance state to storage.
        
        Args:
            instances_data: List of instance data dictionaries to save
            worker_id: Unique identifier for the worker saving the state
            
        Returns:
            True if save was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def check_for_updates(self, last_check_time: float) -> Tuple[bool, float]:
        """
        Check if there are updates since last check time.
        
        Args:
            last_check_time: Timestamp of last check
            
        Returns:
            Tuple of (has_updates, current_time)
        """
        pass

class FileStateManager(StateManager):
    """File-based implementation of state management."""
    
    def __init__(self, file_path: Optional[str] = None, poll_interval: int = 5):
        """
        Initialize the file state manager.
        
        Args:
            file_path: Path to state file, defaults to temp directory
            poll_interval: Interval in seconds for polling updates
        """
        self.file_path = file_path or os.environ.get(
            'INSTANCE_MANAGER_STATE_FILE', 
            os.path.join(tempfile.gettempdir(), 'azure_openai_proxy_instances.json')
        )
        self.poll_interval = poll_interval
        logger.info(f"Initialized file state manager with state file: {self.file_path}")
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load instance state from file.
        
        Returns:
            Dictionary containing instance state, or None if not available/error
        """
        if not os.path.exists(self.file_path):
            logger.debug(f"State file does not exist: {self.file_path}")
            return None
            
        try:
            with open(self.file_path, 'r') as f:
                state = json.load(f)
            
            logger.debug(f"Loaded state from file with {len(state.get('instances', []))} instances")
            return state
        except Exception as e:
            logger.error(f"Error loading state from file: {str(e)}")
            return None
    
    def save_state(self, instances_data: List[Dict[str, Any]], worker_id: str) -> bool:
        """
        Save instance state to file.
        
        Args:
            instances_data: List of instance data dictionaries to save
            worker_id: Unique identifier for the worker saving the state
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            state = {
                'timestamp': time.time(),
                'worker_id': worker_id,
                'instances': instances_data
            }
            
            # Write to temporary file first, then rename for atomicity
            temp_file = f"{self.file_path}.{worker_id}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(state, f)
                
            # Atomic rename
            os.replace(temp_file, self.file_path)
            
            logger.debug(f"Saved state to file with {len(instances_data)} instances")
            return True
        except Exception as e:
            logger.error(f"Error saving state to file: {str(e)}")
            return False
    
    def check_for_updates(self, last_check_time: float) -> Tuple[bool, float]:
        """
        Check if there are updates to the state file since last check.
        
        Args:
            last_check_time: Timestamp of last check
            
        Returns:
            Tuple of (has_updates, current_time)
        """
        current_time = time.time()
        
        # Only check if enough time has passed since last check
        if current_time - last_check_time < self.poll_interval:
            return (False, current_time)
        
        # Check file modification time
        if not os.path.exists(self.file_path):
            return (False, current_time)
            
        try:
            mtime = os.path.getmtime(self.file_path)
            return (mtime > last_check_time, current_time)
        except Exception as e:
            logger.error(f"Error checking file modification time: {str(e)}")
            return (False, current_time)

# Factory function to create appropriate state manager
def create_state_manager(manager_type: str = "file", **kwargs) -> StateManager:
    """
    Factory function to create a state manager.
    
    Args:
        manager_type: Type of state manager ('file', 'redis', etc.)
        **kwargs: Additional arguments for the state manager
        
    Returns:
        Initialized state manager
        
    Raises:
        ValueError: If manager_type is not supported
    """
    if manager_type.lower() == "file":
        return FileStateManager(**kwargs)
    
    # Add support for other state manager types here
    # elif manager_type.lower() == "redis":
    #     return RedisStateManager(**kwargs)
    
    raise ValueError(f"Unsupported state manager type: {manager_type}") 