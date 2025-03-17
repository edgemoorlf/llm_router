"""
New instance manager implementation with separated config and state.

This module provides a new implementation of the InstanceManager class
that uses the separated configuration and state architecture.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union

from app.models.instance import InstanceConfig, InstanceState, create_instance_state
from app.storage.config_store import ConfigStore
from app.storage.redis_state_store import RedisStateStore

logger = logging.getLogger(__name__)

class InstanceManager:
    """
    Manages OpenAI-compatible API instances with separated config and state.
    
    This class handles the loading, tracking, and selection of API instances
    for forwarding requests to OpenAI-compatible APIs. It maintains a clear
    separation between configuration (defined by operators) and state (derived
    from runtime operation).
    """
    
    def __init__(self, 
                config_file: str = "instance_configs.json",
                redis_url: str = "redis://localhost:6379/0",
                rpm_window_minutes: int = 5):
        """
        Initialize the instance manager.
        
        Args:
            config_file: Path to the configuration file
            redis_url: Redis connection URL for state storage
            rpm_window_minutes: Time window in minutes for statistics calculation
        """
        # Initialize stores
        self.config_store = ConfigStore(config_file)
        self.state_store = RedisStateStore(redis_url=redis_url)
        self.rpm_window_minutes = rpm_window_minutes
        
        # Initialize states for all configs
        self._initialize_states()
        
        self.router = None  # Will be initialized later
        
        # Reset transient states on startup
        self.state_store.reset_states_on_startup()
        
    def _initialize_states(self):
        """Initialize states for all configured instances."""
        configs = self.config_store.get_all_configs()
        if configs:
            logger.info(f"Initializing states for {len(configs)} instances")
            # Get list of instance names
            instance_names = list(configs.keys())
            # Reset states and initialize new ones
            self.state_store.reset_states_on_startup(instance_names)
            # Initialize states for each instance
            for name in instance_names:
                if not self.state_store.get_state(name):
                    state = create_instance_state(name)
                    self.state_store.update_state(name, **state.dict())
        
    def reload_config(self):
        """Reload configurations from storage and YAML."""
        # First try to reload from JSON storage
        self.config_store = ConfigStore(self.config_store.config_file)
        
        # If no configs loaded, try YAML
        if not self.config_store.get_all_configs():
            self.config_store._load_from_yaml()
        
        # Make sure all configured instances have a state
        self._initialize_states()
        
    def get_instance(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get an instance by name.
        
        Args:
            name: Name of the instance
            
        Returns:
            Combined instance data or None if not found
        """
        config = self.config_store.get_config(name)
        if not config:
            return None
            
        state = self.state_store.get_state(name)
        if not state:
            return config.dict()
            
        # Combine config and state, preferring config values
        result = config.dict()
        for k, v in state.dict().items():
            if k != 'name':  # Skip name to avoid duplication
                result[k] = v
        return result
        
    def get_all_instances(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all instances.
        
        Returns:
            Dictionary of instance name to combined properties
        """
        result = {}
        configs = self.config_store.get_all_configs()
        states = self.state_store.get_all_states()
        
        for name, config in configs.items():
            result[name] = config.dict()
            if name in states:
                state = states[name]
                for k, v in state.dict().items():
                    if k != 'name':  # Skip name to avoid duplication
                        result[name][k] = v
                        
        return result
        
    def get_instance_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all instances.
        
        Returns:
            Dictionary with instance statistics
        """
        states = self.state_store.get_all_states()
        
        # Filter to only include instances that have configuration
        configs = self.config_store.get_all_configs()
        instance_stats = [
            state.dict() for name, state in states.items()
            if name in configs
        ]
        
        return {
            "status": "success",
            "timestamp": int(time.time()),
            "instances": instance_stats,
            "count": len(instance_stats)
        }
        
    def update_instance_state(self, name: str, **kwargs) -> bool:
        """
        Update state for a specific instance.
        
        Args:
            name: Name of the instance
            **kwargs: State properties to update
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the instance exists in config
        if not self.config_store.get_config(name):
            logger.warning(f"Cannot update state for non-existent instance: {name}")
            return False
            
        try:
            self.state_store.update_state(name, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Error updating state for {name}: {e}")
            return False
            
    def clear_instance_error(self, name: str) -> bool:
        """
        Clear error state for an instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            True if successful, False otherwise
        """
        return self.state_store.clear_error_state(name)
        
    def record_request(self, name: str, success: bool, tokens: int = 0,
                      latency_ms: Optional[float] = None,
                      error: Optional[str] = None,
                      status_code: Optional[int] = None) -> bool:
        """
        Record a request to update instance metrics.
        
        Args:
            name: Name of the instance
            success: Whether the request was successful
            tokens: Number of tokens processed
            latency_ms: Request latency in milliseconds
            error: Error message if request failed
            status_code: HTTP status code if request failed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state_store.record_request(
                name=name,
                success=success,
                tokens=tokens,
                latency_ms=latency_ms,
                error=error,
                status_code=status_code
            )
            return True
        except Exception as e:
            logger.error(f"Error recording request for {name}: {e}")
            return False

    def get_all_configs(self) -> Dict[str, InstanceConfig]:
        """
        Get all instance configurations.
        
        Returns:
            Dictionary of instance name to configuration
        """
        return self.config_store.get_all_configs()

    def get_instance_config(self, name: str) -> Optional[InstanceConfig]:
        """
        Get configuration for a specific instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            The instance configuration or None if not found
        """
        return self.config_store.get_config(name)

    def get_instance_state(self, name: str) -> Optional[InstanceState]:
        """
        Get current state for a specific instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            The instance state or None if not found
        """
        return self.state_store.get_state(name)

    def get_all_states(self) -> Dict[str, InstanceState]:
        """
        Get all instance states.
        
        Returns:
            Dictionary of instance name to state
        """
        return self.state_store.get_all_states()

    def update_tpm_usage(self, name: str, tokens: int) -> None:
        """Update the tokens per minute usage for an instance."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        current_time = int(time.time())
        window_start = current_time - 60
        state.usage_window = {ts: usage for ts, usage in state.usage_window.items() if ts >= window_start}
        state.usage_window[current_time] = state.usage_window.get(current_time, 0) + tokens
        state.current_tpm = sum(state.usage_window.values())
        state.total_tokens_served += tokens
        
        self.state_store.update_state(name, **state.dict())
    
    def update_rpm_usage(self, name: str) -> None:
        """Update the requests per minute counter for an instance."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        current_time = int(time.time())
        window_seconds = self.rpm_window_minutes * 60
        window_start = current_time - window_seconds
        
        # Update sliding window
        state.request_window = {ts: count for ts, count in state.request_window.items() if ts >= window_start}
        state.request_window[current_time] = state.request_window.get(current_time, 0) + 1
        
        # Calculate RPM
        total_requests_in_window = sum(state.request_window.values())
        if self.rpm_window_minutes > 0:
            state.current_rpm = int(total_requests_in_window / self.rpm_window_minutes)
        else:
            state.current_rpm = total_requests_in_window
            
        self.state_store.update_state(name, **state.dict())
        self.update_error_rates(name)
    
    def update_error_rates(self, name: str) -> None:
        """Update error rates based on the time window."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        current_time = int(time.time())
        window_seconds = self.rpm_window_minutes * 60
        window_start = current_time - window_seconds
        
        # Calculate request count in the window
        total_requests_in_window = sum(state.request_window.values())
        
        # Clean up old entries in error windows
        state.error_500_window = {ts: count for ts, count in state.error_500_window.items() if ts >= window_start}
        state.error_503_window = {ts: count for ts, count in state.error_503_window.items() if ts >= window_start}
        state.error_other_window = {ts: count for ts, count in state.error_other_window.items() if ts >= window_start}
        
        # Calculate error counts in the window
        errors_500_in_window = sum(state.error_500_window.values())
        errors_503_in_window = sum(state.error_503_window.values())
        errors_other_in_window = sum(state.error_other_window.values())
        total_errors_in_window = errors_500_in_window + errors_503_in_window + errors_other_in_window
        
        # Update error rates
        if total_requests_in_window > 0:
            state.current_error_rate = round(total_errors_in_window / total_requests_in_window, 4)
            state.current_500_rate = round(errors_500_in_window / total_requests_in_window, 4)
            state.current_503_rate = round(errors_503_in_window / total_requests_in_window, 4)
        else:
            state.current_error_rate = 0.0
            state.current_500_rate = 0.0
            state.current_503_rate = 0.0
            
        self.state_store.update_state(name, **state.dict())
    
    def record_error(self, name: str, status_code: int) -> None:
        """Record an instance-level error."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        current_time = int(time.time())
        
        if status_code == 500:
            state.total_errors_500 += 1
            state.error_500_window[current_time] = state.error_500_window.get(current_time, 0) + 1
        elif status_code == 503:
            state.total_errors_503 += 1
            state.error_503_window[current_time] = state.error_503_window.get(current_time, 0) + 1
        else:
            state.total_other_errors += 1
            state.error_other_window[current_time] = state.error_other_window.get(current_time, 0) + 1
            
        self.state_store.update_state(name, **state.dict())
        self.update_error_rates(name)
        
        logger.debug(f"Recorded instance-level error {status_code} for instance {name}, "
                    f"current error rate: {state.current_error_rate:.2%}")
    
    def record_client_error(self, name: str, status_code: int, timestamp: Optional[int] = None) -> None:
        """Record a client-level error."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        if timestamp is None:
            timestamp = int(time.time())
            
        if status_code == 500:
            state.total_client_errors_500 += 1
            state.client_error_500_window[timestamp] = state.client_error_500_window.get(timestamp, 0) + 1
        elif status_code == 503:
            state.total_client_errors_503 += 1
            state.client_error_503_window[timestamp] = state.client_error_503_window.get(timestamp, 0) + 1
        else:
            state.total_client_errors_other += 1
            state.client_error_other_window[timestamp] = state.client_error_other_window.get(timestamp, 0) + 1
            
        self.state_store.update_state(name, **state.dict())
        self.update_error_rates(name)
        
        logger.debug(f"Recorded client-level error {status_code} for instance {name}, "
                    f"current client error rate: {state.current_client_error_rate:.2%}")
    
    def record_upstream_error(self, name: str, status_code: int) -> None:
        """Record an upstream error."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        current_time = int(time.time())
        
        if status_code == 429:
            state.total_upstream_429_errors += 1
            state.upstream_429_window[current_time] = state.upstream_429_window.get(current_time, 0) + 1
        elif status_code == 400:
            state.total_upstream_400_errors += 1
            state.upstream_400_window[current_time] = state.upstream_400_window.get(current_time, 0) + 1
        elif status_code == 500:
            state.total_upstream_500_errors += 1
            state.upstream_500_window[current_time] = state.upstream_500_window.get(current_time, 0) + 1
        else:
            state.total_upstream_other_errors += 1
            state.upstream_other_window[current_time] = state.upstream_other_window.get(current_time, 0) + 1
            
        self.state_store.update_state(name, **state.dict())
        self.update_error_rates(name)
        
        logger.debug(f"Recorded upstream error {status_code} for instance {name}, "
                    f"current upstream error rate: {state.current_upstream_error_rate:.2%}")
    
    def is_rate_limited(self, name: str) -> bool:
        """Check if an instance is currently rate limited."""
        state = self.state_store.get_state(name)
        if not state:
            return True
            
        # Check explicit rate limit status
        if state.status == "rate_limited":
            current_time = time.time()
            if state.rate_limited_until and current_time >= state.rate_limited_until:
                logger.info(f"Instance {name} rate limit has expired, marking as healthy")
                state.status = "healthy"
                state.rate_limited_until = None
                self.state_store.update_state(name, **state.dict())
                return False
            return True
            
        # Check TPM-based rate limiting
        config = self.config_store.get_config(name)
        if config and state.current_tpm >= config.max_tpm * 0.95:
            logger.warning(f"Instance {name} is approaching TPM limit: {state.current_tpm}/{config.max_tpm}")
            return True
            
        return False
    
    def mark_rate_limited(self, name: str, retry_after: Optional[int] = None) -> None:
        """Mark an instance as rate limited."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        state.status = "rate_limited"
        if retry_after is None:
            retry_after = 60
        state.rate_limited_until = time.time() + retry_after
        self.state_store.update_state(name, **state.dict())
        logger.warning(f"Instance {name} marked as rate limited for {retry_after} seconds")
    
    def mark_healthy(self, name: str) -> None:
        """Mark an instance as healthy."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        state.status = "healthy"
        state.error_count = 0
        state.last_error = None
        state.rate_limited_until = None
        self.state_store.update_state(name, **state.dict())
    
    def get_metrics(self, name: str) -> Optional[Dict[str, Any]]:
        """Get all metrics for an instance."""
        state = self.state_store.get_state(name)
        if not state:
            return None
            
        return {
            "current_tpm": state.current_tpm,
            "current_rpm": state.current_rpm,
            "total_tokens_served": state.total_tokens_served,
            "error_rates": {
                "instance": {
                    "total": state.current_error_rate,
                    "500": state.current_500_rate,
                    "503": state.current_503_rate
                },
                "client": {
                    "total": state.current_client_error_rate,
                    "500": state.current_client_500_rate,
                    "503": state.current_client_503_rate
                },
                "upstream": {
                    "total": state.current_upstream_error_rate,
                    "429": state.current_upstream_429_rate,
                    "400": state.current_upstream_400_rate
                }
            },
            "error_counts": {
                "instance": {
                    "500": state.total_errors_500,
                    "503": state.total_errors_503,
                    "other": state.total_other_errors
                },
                "client": {
                    "500": state.total_client_errors_500,
                    "503": state.total_client_errors_503,
                    "other": state.total_client_errors_other
                },
                "upstream": {
                    "429": state.total_upstream_429_errors,
                    "400": state.total_upstream_400_errors,
                    "500": state.total_upstream_500_errors,
                    "other": state.total_upstream_other_errors
                }
            }
        } 