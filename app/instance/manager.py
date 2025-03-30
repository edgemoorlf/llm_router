"""
New instance manager implementation with separated config and state.

This module provides a new implementation of the InstanceManager class
that uses the separated configuration and state architecture.
"""

import logging
import time
import os
import random
from typing import Dict, List, Optional, Any, Union, Tuple
from fastapi import HTTPException

from app.models.instance import InstanceConfig, InstanceState, InstanceStatus, create_instance_state
from app.storage.config_store import ConfigStore
from app.storage.redis_state_store import RedisStateStore
from app.utils.rate_limiter import get_rate_limiter, RateLimiter

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
                redis_url: Optional[str] = None,
                redis_password: Optional[str] = None,
                rpm_window_minutes: int = 5):
        """
        Initialize the instance manager.
        
        Args:
            config_file: Path to the configuration file
            redis_url: Redis connection URL for state storage (defaults to env var REDIS_URL)
            redis_password: Redis password for authentication (defaults to env var REDIS_PASSWORD)
            rpm_window_minutes: Time window in minutes for statistics calculation
        """
        # Initialize stores
        self.config_store = ConfigStore(config_file)
        
        # Get Redis configuration from environment or parameters
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.redis_password = redis_password or os.environ.get("REDIS_PASSWORD", "")
        
        # Initialize state store with Redis configuration
        self.state_store = RedisStateStore(redis_url=self.redis_url, redis_password=self.redis_password)
        self.rpm_window_minutes = rpm_window_minutes
        
        # Initialize rate limiters dictionary
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # Initialize states and rate limiters for all configs
        self._initialize_states()
        
        self.router = None  # Will be initialized later
        
    def _initialize_states(self):
        """Initialize states and rate limiters for all configured instances."""
        configs = self.config_store.get_all_configs()
        if configs:
            logger.info(f"Initializing states and rate limiters for {len(configs)} instances")
            # Get list of instance names
            instance_names = list(configs.keys())
            # Reset states and initialize new ones
            self.state_store.reset_states_on_startup(instance_names)
            
            # Initialize states and rate limiters for each instance
            for name, config in configs.items():
                # Initialize state if not exists
                if not self.state_store.get_state(name):
                    state = create_instance_state(name)
                    state_dict = state.dict()
                    state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
                    self.state_store.update_state(name, **state_dict)
                
                # Only create rate limiter if it doesn't exist
                if name not in self.rate_limiters:
                    self.rate_limiters[name] = get_rate_limiter(
                        instance_id=name,
                        tokens_per_minute=config.max_tpm,
                        redis_url=self.redis_url,
                        redis_password=self.redis_password,
                        max_input_tokens=config.max_input_tokens
                    )
                else:
                    # Update existing rate limiter if config changed
                    existing_limiter = self.rate_limiters[name]
                    if (existing_limiter.tokens_per_minute != config.max_tpm or 
                        existing_limiter.max_input_tokens != config.max_input_tokens):
                        logger.info(f"Updating rate limiter configuration for instance {name}")
                        self.rate_limiters[name] = get_rate_limiter(
                            instance_id=name,
                            tokens_per_minute=config.max_tpm,
                            redis_url=self.redis_url,
                            redis_password=self.redis_password,
                            max_input_tokens=config.max_input_tokens
                        )

    def _get_rate_limiter(self, name: str) -> Optional[RateLimiter]:
        """
        Get the rate limiter for an instance.
        
        If the rate limiter doesn't exist in memory but the instance config exists,
        create it on-demand to ensure rate limiting continues to work properly.
        
        Args:
            name: Name of the instance
            
        Returns:
            Rate limiter for the instance or None if not found
        """
        rate_limiter = self.rate_limiters.get(name)
        if not rate_limiter:
            # Instead of just logging a warning, try to create the rate limiter
            # if the instance config exists
            config = self.get_instance_config(name)
            if config:
                logger.warning(f"Rate limiter for {name} not found in memory, creating on-demand")
                self.rate_limiters[name] = get_rate_limiter(
                    instance_id=name,
                    tokens_per_minute=config.max_tpm,
                    redis_url=self.redis_url,
                    redis_password=self.redis_password,
                    max_input_tokens=config.max_input_tokens
                )
                return self.rate_limiters[name]
            else:
                logger.error(f"Cannot create rate limiter for {name}: config not found")
        return rate_limiter

    def reload_config(self):
        """Reload configurations from storage and YAML."""

        old_configs = self.config_store.configs
        
        new_configs = self.config_store.get_all_configs()
        
        # Remove rate limiters for instances that no longer exist
        removed_instances = set(old_configs.keys()) - set(new_configs.keys())
        for name in removed_instances:
            if name in self.rate_limiters:
                logger.info(f"Removing rate limiter for deleted instance {name}")
                del self.rate_limiters[name]
        
        # Initialize states and update rate limiters for current configs
        self._initialize_states()

    def get_instance(self, name: str) -> Optional[Tuple[InstanceConfig, InstanceState]]:
        """
        Get an instance by name, returning both config and state.
        
        Args:
            name: Name of the instance
            
        Returns:
            Tuple of (config, state) if found, None otherwise
        """
        
        # Get fresh config
        config = self.config_store.get_config(name)
        if not config:
            return None
            
        # Get fresh state directly from Redis
        state = self.state_store.get_state(name)
        if not state:
            # If state doesn't exist, create a new one
            state = create_instance_state(name)
            state_dict = state.dict()
            state_dict.pop('name', None)  # Remove name to avoid duplicate
            self.state_store.update_state(name, **state_dict)
            
        return (config, state)
        
    def get_all_instances(self) -> List[Tuple[InstanceConfig, InstanceState]]:
        """
        Get all instances with their configs and states directly from storage.
        
        This method always gets fresh data from both the config file and Redis,
        to ensure consistency across multiple worker processes.
        
        Returns:
            List of (config, state) tuples
        """
        results = []
        
        # Always reload configs to get fresh data
        self.config_store.reload()
        
        # Get fresh configs
        configs = self.config_store.get_all_configs()
        
        # Get fresh states directly from Redis
        states = self.state_store.get_all_states()
        
        logger.debug(f"get_all_instances: retrieved {len(configs)} configs and {len(states)} states")
        
        for name, config in configs.items():
            # Get or create state if needed
            state = states.get(name)
            if not state:
                state = create_instance_state(name)
                state_dict = state.dict()
                state_dict.pop('name', None)  # Remove name to avoid duplicate
                self.state_store.update_state(name, **state_dict)
                
            results.append((config, state))
            
        return results
        
    def get_instance_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all instances.
        
        Returns:
            Dictionary with instance statistics
        """
        logger.debug("Calling get_instance_stats")
        
        # First sync all rate limiter data to instance states
        configs = self.config_store.get_all_configs()
        for name in configs.keys():
            self.sync_rate_limiter_to_state(name)
        
        # Now get the updated states
        states = self.state_store.get_all_states()
        logger.debug(f"Retrieved {len(states)} states: {list(states.keys())}")
        
        # Filter to only include instances that have configuration
        instance_stats = [
            state.dict() for name, state in states.items()
            if name in configs
        ]
        
        logger.debug(f"Filtered to {len(instance_stats)} instances with both state and config")
        
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
            latency_ms: Request latency in milliseconds (ignored)
            error: Error message if request failed
            status_code: HTTP status code if request failed (ignored)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state_store.record_request(
                name=name,
                success=success,
                tokens=tokens,
                error=error
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
        Get current state for a specific instance directly from Redis.
        
        Args:
            name: Name of the instance
            
        Returns:
            The instance state or None if not found
        """
        # Always get a fresh state from Redis, no caching
        logger.debug(f"Getting fresh state for instance {name} directly from Redis")
        return self.state_store.get_state(name)
    
    def get_all_states(self) -> Dict[str, InstanceState]:
        """
        Get all instance states directly from Redis.
        
        Returns:
            Dictionary of instance name to state
        """
        # Always get fresh states from Redis, no caching
        logger.debug("Getting all fresh states directly from Redis")
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
        
        state_dict = state.dict()
        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
        self.state_store.update_state(name, **state_dict)
    
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
            
        state_dict = state.dict()
        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
        self.state_store.update_state(name, **state_dict)
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
            
        state_dict = state.dict()
        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
        self.state_store.update_state(name, **state_dict)
    
    def record_error(self, name: str, error_type: str, error_message: str) -> None:
        """
        Record an error for an instance.
        
        Args:
            name: Name of the instance
            error_type: Type of error (client_error, upstream_error)
            error_message: Error message
        """
        state = self.state_store.get_state(name)
        if not state:
            return
            
        current_time = int(time.time())
        
        # Update state with error information
        state.error_count += 1
        state.last_error = error_message
        state.last_error_time = current_time
        
        # Update state based on error type
        if error_type == "client_error":
            # Use a generic status code for client errors
            self.record_client_error(name, 400)
        elif error_type == "upstream_error":
            # Use a generic status code for upstream errors
            self.record_upstream_error(name, 500)
        
        # If too many errors, mark instance as unhealthy
        if state.error_count >= 3:
            state.status = InstanceStatus.ERROR
            
        # Save the updated state
        state_dict = state.dict()
        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
        self.state_store.update_state(name, **state_dict)
        
        logger.debug(f"Recorded {error_type} for instance {name}: {error_message}")
    
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
            
        state_dict = state.dict()
        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
        self.state_store.update_state(name, **state_dict)
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
            
        state_dict = state.dict()
        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
        self.state_store.update_state(name, **state_dict)
        self.update_error_rates(name)
        
        logger.debug(f"Recorded upstream error {status_code} for instance {name}, "
                    f"current upstream error rate: {state.current_upstream_error_rate:.2%}")
    
    def is_rate_limited(self, name: str) -> bool:
        """Check if an instance is currently rate limited."""
        state = self.state_store.get_state(name)
        if not state:
            return True
            
        # Check explicit rate limit status
        if state.status == InstanceStatus.RATE_LIMITED:
            current_time = time.time()
            if state.rate_limited_until and current_time >= state.rate_limited_until:
                logger.info(f"Instance {name} rate limit has expired, marking as healthy")
                state.status = InstanceStatus.HEALTHY
                state.rate_limited_until = None
                state_dict = state.dict()
                state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
                self.state_store.update_state(name, **state_dict)
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
            
        state.status = InstanceStatus.RATE_LIMITED
        if retry_after is None:
            retry_after = 60
        state.rate_limited_until = time.time() + retry_after
        state_dict = state.dict()
        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
        self.state_store.update_state(name, **state_dict)
        logger.warning(f"Instance {name} marked as rate limited for {retry_after} seconds")
    
    def mark_healthy(self, name: str) -> None:
        """Mark an instance as healthy."""
        state = self.state_store.get_state(name)
        if not state:
            return
            
        state.status = InstanceStatus.HEALTHY
        state.error_count = 0
        state.last_error = None
        state.rate_limited_until = None
        state_dict = state.dict()
        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
        self.state_store.update_state(name, **state_dict)
    
    def update_token_usage(self, name: str, tokens: int) -> None:
        """Update token usage for an instance after a successful request.
        
        This should be called after a request has been successfully processed
        to update the token consumption for rate limiting purposes.
        
        Args:
            name: Name of the instance
            tokens: Number of tokens consumed
        """
        if not self.state_store:
            return
            
        # Get instance config to check if rate limiting is enabled
        instance_data = self.get_instance(name)
        if not instance_data:
            logger.warning(f"Cannot update token usage for {name}: instance not found")
            return
            
        # Extract config from the tuple
        config, _ = instance_data
        
        # Get the current state
        state = self.state_store.get_state(name)
        if not state:
            logger.warning(f"Cannot update token usage for {name}: state not found")
            return
            
        # If rate limiting is enabled, update the token usage
        if config.rate_limit_enabled and tokens > 0:
            # Get or create rate limiter if needed
            rate_limiter = self._get_rate_limiter(name)
            if rate_limiter:
                try:
                    rate_limiter.update_usage(tokens)
                    
                    # Update current_tpm in the instance state to match rate limiter
                    try:
                        current_usage = rate_limiter.get_current_usage()
                        state.current_tpm = current_usage
                        
                        # Update state in Redis
                        state_dict = state.dict()
                        state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
                        self.state_store.update_state(name, **state_dict)
                    except Exception as e:
                        logger.error(f"Error updating state after token usage for {name}: {e}")
                except Exception as e:
                    logger.error(f"Error updating token usage for {name}: {e}")
            else:
                logger.error(f"Cannot update token usage for {name}: rate limiter not available")
                
        # Update historical total_tokens_served regardless of rate limiting
        try:
            self.state_store.update_total_tokens(name, tokens)
        except Exception as e:
            logger.error(f"Error updating total tokens for {name}: {e}")
    
    def get_metrics(self, name: str) -> Optional[Dict[str, Any]]:
        """Get all metrics for an instance."""
        # First sync the rate limiter data to the instance state
        self.sync_rate_limiter_to_state(name)
        
        # Now get the updated state
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

    def check_rate_limit(self, name: str, tokens: int) -> bool:
        """Check if an instance has capacity for tokens under rate limiting.
        
        This method ONLY checks capacity and doesn't update any token usage.
        Token usage should be updated after a successful request using update_token_usage.
        
        Args:
            name: Name of the instance
            tokens: Number of tokens to check
            
        Returns:
            True if instance has capacity, False otherwise
        """
        instance_config = self.get_instance_config(name)
        if not instance_config:
            logger.warning(f"Instance config not found for {name}")
            return False
            
        # If rate limiting is not enabled, always return True
        if not getattr(instance_config, "rate_limit_enabled", True):
            return True
            
        # Get or create rate limiter if needed
        rate_limiter = self._get_rate_limiter(name)
        if not rate_limiter:
            logger.error(f"Rate limiter not available for {name} and cannot be created")
            # Fail open to allow request to proceed
            return True
            
        try:
            allowed, _ = rate_limiter.check_capacity(tokens)
            return allowed
        except Exception as e:
            logger.error(f"Error checking rate limit for {name}: {e}")
            # Fail open on errors to allow requests to proceed
            return True
            
    def get_current_usage(self, name: str) -> int:
        """
        Get current token usage for an instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            Current token usage in the rate limit window
        """
        # Get or create rate limiter if needed
        rate_limiter = self._get_rate_limiter(name)
        if not rate_limiter:
            logger.error(f"Rate limiter not available for {name} and cannot be created")
            return 0
            
        try:
            return rate_limiter.get_current_usage()
        except Exception as e:
            logger.error(f"Error getting current usage for {name}: {e}")
            return 0

    def select_instance_for_request(self, model: str, tokens: int, provider_type: str = None) -> Optional[str]:
        """
        Select the best instance for a request based on model compatibility and capacity.
        
        This method delegates to InstanceSelector for the actual selection logic.
        
        Args:
            model: The model name to use
            tokens: Required tokens for the request
            provider_type: Optional provider type filter
            
        Returns:
            Name of the selected instance or None if no suitable instance is found
        """
        # Import here to avoid circular import
        from app.services.instance_selector import instance_selector
        
        return instance_selector.select_instance_for_request(model, tokens, provider_type)
    
    def has_instance(self, name: str) -> bool:
        """
        Check if an instance with the given name exists.
        
        Args:
            name: Name of the instance to check
            
        Returns:
            True if the instance exists, False otherwise
        """
        return name in self.config_store.get_all_configs()
    
    def add_instance(self, config: InstanceConfig) -> bool:
        """
        Add a new instance configuration and initialize its state.
        
        Args:
            config: The instance configuration to add
            
        Returns:
            True if the instance was added successfully, False otherwise
        """
        # Add the configuration
        added = self.config_store.add_config(config)
        
        if added:
            # Initialize state if doesn't exist
            if not self.state_store.get_state(config.name):
                state = create_instance_state(config.name)
                state_dict = state.dict()
                state_dict.pop('name', None)  # Remove name to avoid duplicate
                self.state_store.update_state(config.name, **state_dict)
            
            # Initialize rate limiter
            if config.name not in self.rate_limiters:
                self.rate_limiters[config.name] = get_rate_limiter(
                    instance_id=config.name,
                    tokens_per_minute=config.max_tpm,
                    redis_url=self.redis_url,
                    redis_password=self.redis_password,
                    max_input_tokens=config.max_input_tokens
                )
        
        return added
        
    def set_rpm_window(self, name: str, minutes: int) -> None:
        """
        Set the RPM window size for an instance.
        
        Args:
            name: Name of the instance
            minutes: Window size in minutes
        """
        # Update the RPM window size
        self.rpm_window_minutes = minutes
        
        # Create a new window in state if needed
        state = self.state_store.get_state(name)
        if state:
            # No need to update the state here as the RPM window
            # is managed by the instance manager
            pass
    
    def update_instance(self, name: str, config_updates: Dict[str, Any]) -> bool:
        """
        Update an instance configuration.
        
        Args:
            name: Name of the instance to update
            config_updates: Dictionary with fields to update in the configuration
            
        Returns:
            True if successful, False otherwise
        """
        # Get the current config
        current_config = self.config_store.get_config(name)
        if not current_config:
            logger.warning(f"Cannot update non-existent instance: {name}")
            return False
            
        # Create a dictionary from the current config
        updated_config_dict = current_config.dict()
        
        # Update with the new values
        updated_config_dict.update(config_updates)
        
        # Ensure name is preserved
        updated_config_dict["name"] = name
        
        # Convert back to InstanceConfig
        updated_config = InstanceConfig(**updated_config_dict)
        
        # Update the configuration
        result = self.config_store.add_config(updated_config)
        
        # If tpm or input tokens changed, update the rate limiter
        if result and (updated_config.max_tpm != current_config.max_tpm or 
                    updated_config.max_input_tokens != current_config.max_input_tokens):
            # Update rate limiter if it exists
            if name in self.rate_limiters:
                self.rate_limiters[name] = get_rate_limiter(
                    instance_id=name,
                    tokens_per_minute=updated_config.max_tpm,
                    redis_url=self.redis_url,
                    redis_password=self.redis_password,
                    max_input_tokens=updated_config.max_input_tokens
                )
                
        return result
        
    def remove_instance(self, name: str) -> bool:
        """
        Remove an instance configuration and state.
        
        Args:
            name: Name of the instance to remove
            
        Returns:
            True if the instance was removed, False otherwise
        """
        # Remove rate limiter if it exists
        if name in self.rate_limiters:
            del self.rate_limiters[name]
        
        # Remove from storage
        config_removed = self.config_store.remove_config(name)
        state_removed = self.state_store.remove_state(name)
        
        return config_removed or state_removed 

    def get_all_instance_configs(self) -> List[InstanceConfig]:
        """
        Get all instance configurations without fetching states from Redis.
        
        This is an optimization to avoid Redis calls during initial filtering stages
        when only configuration data (models, names, etc.) is needed. This method
        always reloads configs to ensure fresh data.
        
        Returns:
            List of InstanceConfig objects
        """
        # Always reload configs to get fresh data
        self.config_store.reload()
        
        # Return all configs as a list
        logger.debug("get_all_instance_configs: retrieving configs without Redis calls")
        return list(self.config_store.get_all_configs().values()) 

    def get_states_for_configs(self, configs: List[InstanceConfig]) -> List[Tuple[InstanceConfig, InstanceState]]:
        """
        Get states only for the provided list of instance configurations.
        
        This is an optimization to avoid fetching all states from Redis when only
        a subset of instances is needed. This method fetches states only for
        the configurations provided, limiting Redis calls to only what's necessary.
        
        Args:
            configs: List of instance configurations to get states for
            
        Returns:
            List of (config, state) tuples for the provided configurations
        """
        result = []
        
        for config in configs:
            # Get state only for this specific instance
            state = self.get_instance_state(config.name)
            if not state:
                # If state doesn't exist, create a new one
                state = create_instance_state(config.name)
                state_dict = state.dict()
                state_dict.pop('name', None)  # Remove name to avoid duplicate
                self.state_store.update_state(config.name, **state_dict)
            
            result.append((config, state))
        
        logger.debug(f"get_states_for_configs: fetched {len(result)} states for specified configs")
        return result

    def sync_rate_limiter_to_state(self, name: str) -> None:
        """
        Sync token usage data from the rate limiter to the instance state.
        This ensures current_tpm in the instance state matches the rate limiter's data.
        
        Args:
            name: Name of the instance
        """
        # Skip if no state store
        if not self.state_store:
            return
            
        # Get or create rate limiter if needed
        rate_limiter = self._get_rate_limiter(name)
        if not rate_limiter:
            logger.warning(f"Cannot sync rate limiter data for {name}: rate limiter not available")
            return
            
        # Get current state
        state = self.state_store.get_state(name)
        if not state:
            logger.warning(f"Cannot sync rate limiter data for {name}: state not found")
            return
            
        # Get current usage from rate limiter
        current_usage = rate_limiter.get_current_usage()
        
        # Update state only if different
        if state.current_tpm != current_usage:
            state.current_tpm = current_usage
            state_dict = state.dict()
            state_dict.pop('name', None)  # Remove name from dict to avoid duplicate
            self.state_store.update_state(name, **state_dict)
            logger.debug(f"Synced current_tpm for instance {name}: {current_usage}") 