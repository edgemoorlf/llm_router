"""
State storage implementation.

This module provides the StateStore class for managing instance runtime states
that change frequently based on operational metrics.
"""

import json
import os
import time
import logging
import threading
from typing import Dict, Optional, List, Any
from app.models.instance import InstanceState, create_instance_state

logger = logging.getLogger(__name__)

class StateStore:
    """
    Stores and manages instance runtime states.
    
    This class handles the persistence and retrieval of instance states,
    which represent dynamic information derived from operation.
    """
    
    def __init__(self, state_file: Optional[str] = "instance_states.json", 
                auto_save_interval: int = 300):
        """
        Initialize the state store.
        
        Args:
            state_file: Path to the state file
            auto_save_interval: Interval in seconds for automatic state saving
        """
        self.state_file = state_file
        self.states: Dict[str, InstanceState] = {}
        self.file_lock = threading.RLock()  # Lock for thread-safe access
        self.last_save_time = 0
        self.auto_save_interval = auto_save_interval
        
        # Load existing states
        self._load_states()
        
        # Start auto-save thread if interval is positive
        if auto_save_interval > 0:
            self._start_auto_save()
    
    def reload(self):
        """
        Reload states from the file.
        """
        with self.file_lock:
            self._load_states()
    
    def _load_states(self):
        """Load states from storage."""
        if not self.state_file or not os.path.exists(self.state_file):
            return
            
        with self.file_lock:
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    for name, state_data in data.items():
                        try:
                            self.states[name] = InstanceState(**state_data)
                        except Exception as e:
                            logger.warning(f"Error loading state for instance {name}: {e}")
                logger.info(f"Loaded {len(self.states)} instance states from {self.state_file}")
            except Exception as e:
                logger.error(f"Error loading states from {self.state_file}: {e}")
    
    def _save_states(self):
        """Persist states to storage."""
        if not self.state_file:
            return
            
        with self.file_lock:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.state_file) or '.', exist_ok=True)
                
                # Convert to dict of dicts for JSON serialization
                data = {name: state.dict() for name, state in self.states.items()}
                
                # Write to a temporary file first
                temp_file = f"{self.state_file}.tmp"
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=2)
                
                # Rename to the actual file
                os.replace(temp_file, self.state_file)
                
                self.last_save_time = time.time()
                logger.debug(f"Saved {len(self.states)} instance states to {self.state_file}")
            except Exception as e:
                logger.error(f"Error saving states to {self.state_file}: {e}")
    
    def _start_auto_save(self):
        """Start a thread to periodically save states."""
        def auto_save_worker():
            while True:
                time.sleep(self.auto_save_interval)
                try:
                    # Only save if changes have been made
                    if time.time() - self.last_save_time >= self.auto_save_interval:
                        self._save_states()
                except Exception as e:
                    logger.error(f"Error in auto-save thread: {e}")
        
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
        logger.info(f"Started auto-save thread with interval {self.auto_save_interval}s")
    
    def get_state(self, name: str) -> Optional[InstanceState]:
        """
        Get the current state for a specific instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            The instance state or None if not found
        """
        with self.file_lock:
            return self.states.get(name)
    
    def get_all_states(self) -> Dict[str, InstanceState]:
        """
        Get all instance states.
        
        Returns:
            Dictionary of instance name to state
        """
        with self.file_lock:
            # Return a copy to avoid external modifications
            return dict(self.states)
    
    def update_state(self, name: str, **kwargs) -> InstanceState:
        """
        Update specific state attributes for an instance.
        
        Args:
            name: Name of the instance
            **kwargs: Attributes to update
            
        Returns:
            The updated instance state
        """
        with self.file_lock:
            # Create state if it doesn't exist
            if name not in self.states:
                self.states[name] = create_instance_state(name)
            
            # Update specified fields
            for key, value in kwargs.items():
                if hasattr(self.states[name], key):
                    setattr(self.states[name], key, value)
            
            # Auto-save if it's been a while since the last save
            if self.state_file and time.time() - self.last_save_time >= self.auto_save_interval:
                self._save_states()
                
            return self.states[name]
    
    def delete_state(self, name: str) -> bool:
        """
        Delete an instance state.
        
        Args:
            name: Name of the instance
            
        Returns:
            True if the state was deleted, False if it wasn't found
        """
        with self.file_lock:
            if name in self.states:
                del self.states[name]
                if self.state_file:
                    self._save_states()
                return True
            return False
    
    def record_request(self, name: str, success: bool, tokens: int = 0, 
                      latency_ms: Optional[float] = None, 
                      error: Optional[str] = None) -> InstanceState:
        """
        Record a request to update instance metrics.
        
        Args:
            name: Name of the instance
            success: Whether the request was successful
            tokens: Number of tokens processed
            latency_ms: Request latency in milliseconds
            error: Error message if request failed
        
        Returns:
            The updated instance state
        """
        with self.file_lock:
            # Create state if it doesn't exist
            if name not in self.states:
                self.states[name] = create_instance_state(name)
                
            state = self.states[name]
            curr_time = time.time()
            
            # Update basic metrics
            state.last_used = curr_time
            state.total_requests += 1
            
            if success:
                state.successful_requests += 1
                # Reset error count on success
                if state.error_count > 0:
                    state.error_count = 0
                    
                # This is simplified - in a real app you'd want a sliding window
                state.current_tpm += tokens
            else:
                state.error_count += 1
                state.last_error = error
                state.last_error_time = curr_time
                
                # Update status based on errors
                if state.error_count >= 3:
                    state.status = "error"
                    state.health_status = "error"
            
            # Update latency metrics if provided
            if latency_ms is not None:
                if state.avg_latency_ms is None:
                    state.avg_latency_ms = latency_ms
                else:
                    # Simple exponential moving average
                    state.avg_latency_ms = (state.avg_latency_ms * 0.9) + (latency_ms * 0.1)
            
            # Auto-save if it's been a while
            if self.state_file and curr_time - self.last_save_time >= self.auto_save_interval:
                self._save_states()
                
            return state 