"""State management for instance data."""
import abc
import os
import json
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class StateManager(abc.ABC):
    """Abstract base class for state management."""
    
    @abc.abstractmethod
    def load_state(self) -> Dict[str, Any]:
        """
        Load the current state.
        
        Returns:
            The current state
        """
        pass
        
    @abc.abstractmethod
    def save_state(self, instances_data: List[Dict[str, Any]], worker_id: str) -> bool:
        """
        Save the current state.
        
        Args:
            instances_data: List of instance data to save
            worker_id: ID of the worker saving the state
            
        Returns:
            True if state was saved successfully, False otherwise
        """
        pass
        
    @abc.abstractmethod
    def check_for_updates(self, last_check_time: float) -> Tuple[bool, float]:
        """
        Check if there are updates since the last check.
        
        Args:
            last_check_time: Timestamp of the last check
            
        Returns:
            Tuple of (has_updates, current_time)
        """
        pass

    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get the version history.
        
        Returns:
            List of version metadata, including whether each version is the current one
        """
        return []

    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a specific version.
        
        Args:
            version_id: ID of the version to rollback to
            
        Returns:
            True if rollback was successful, False otherwise
        """
        return False
        
    def get_instance(self, instance_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific instance by name.
        
        Args:
            instance_name: Name of the instance to get
            
        Returns:
            Instance data if it exists, None otherwise
        """
        state = self.load_state()
        if not state or 'instances' not in state:
            return None
            
        for instance in state['instances']:
            if instance.get('name') == instance_name:
                return instance
                
        return None

class FileStateManager(StateManager):
    """State manager that uses a file for persistence."""
    
    def __init__(self, state_file_path: Optional[str] = None, polling_interval: float = 30.0):
        """
        Initialize the file state manager.
        
        Args:
            state_file_path: Path to the state file (default: .temp/instance_state.json in the working directory)
            polling_interval: Interval in seconds to check for updates (default: 30.0)
        """
        if state_file_path:
            self.state_file = state_file_path
        else:
            # Default state file in .temp directory
            temp_dir = os.path.join(os.getcwd(), '.temp')
            os.makedirs(temp_dir, exist_ok=True)
            self.state_file = os.path.join(temp_dir, 'instance_state.json')
            
        self.polling_interval = polling_interval
        self.file_lock = threading.Lock()
        logger.info(f"Initialized FileStateManager with state file: {self.state_file}")
        
    def load_state(self) -> Dict[str, Any]:
        """
        Load the current state from the state file.
        
        Returns:
            The current state
        """
        try:
            with self.file_lock:
                if not os.path.exists(self.state_file):
                    return {}
                    
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                return state
        except Exception as e:
            logger.error(f"Error loading state from file: {str(e)}")
            return {}
            
    def save_state(self, instances_data: List[Dict[str, Any]], worker_id: str) -> bool:
        """
        Save the current state to the state file.
        
        Args:
            instances_data: List of instance data to save
            worker_id: ID of the worker saving the state
            
        Returns:
            True if state was saved successfully, False otherwise
        """
        try:
            with self.file_lock:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
                
                # Create state data
                state = {
                    'instances': instances_data,
                    'last_updated': time.time(),
                    'worker_id': worker_id
                }
                
                # Write to file
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                    
                return True
        except Exception as e:
            logger.error(f"Error saving state to file: {str(e)}")
            return False
            
    def check_for_updates(self, last_check_time: float) -> Tuple[bool, float]:
        """
        Check if there are updates since the last check.
        
        Args:
            last_check_time: Timestamp of the last check
            
        Returns:
            Tuple of (has_updates, current_time)
        """
        # Calculate current time
        current_time = time.time()
        
        # Skip check if polling interval hasn't elapsed
        if last_check_time > 0 and current_time - last_check_time < self.polling_interval:
            return False, current_time
            
        try:
            with self.file_lock:
                if not os.path.exists(self.state_file):
                    return False, current_time
                    
                # Get file modification time
                mtime = os.path.getmtime(self.state_file)
                
                # Check if the file has been modified since the last check
                if mtime > last_check_time:
                    logger.debug(f"State file modified since last check ({mtime} > {last_check_time})")
                    return True, current_time
                    
                return False, current_time
        except Exception as e:
            logger.error(f"Error checking for updates: {str(e)}")
            return False, current_time

def create_state_manager(manager_type: str = "file", **kwargs) -> StateManager:
    """
    Create a state manager instance.
    
    Args:
        manager_type: Type of state manager to create:
            - "file": File-based state manager (legacy)
        **kwargs: Additional arguments to pass to the state manager constructor

    Returns:
        StateManager instance
        
    Note:
        The file-based state manager is maintained for legacy support.
        New implementations should use the Redis-based state management
        from the new_manager module.
    """
    if manager_type == "file":
        return FileStateManager(**kwargs)
    else:
        raise ValueError(f"Unknown state manager type: {manager_type}. Only 'file' type is supported for legacy compatibility.") 