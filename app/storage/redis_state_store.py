"""
Redis-based state storage implementation.

This module provides a Redis-backed StateStore for managing instance runtime states
with persistent error tracking and auto-expiring rate limits.
"""

import json
import time
import logging
import os
from typing import Dict, Optional, List, Any
from redis import Redis
from urllib.parse import urlparse
from app.models.instance import InstanceState, create_instance_state

logger = logging.getLogger(__name__)

class RedisStateStore:
    """
    Redis-backed store for instance runtime states.
    
    Handles instance states with different persistence rules:
    - Error states persist until manually cleared or config changed
    - Rate limits auto-expire after their duration
    - Runtime metrics persist with the server
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize the Redis state store.
        
        Args:
            redis_url: Redis connection URL
        """
        # Parse Redis connection parameters
        parsed = urlparse(redis_url)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 6379
        db = int(parsed.path.lstrip('/') or '0')
        password = os.environ.get('REDIS_PASSWORD')  # Get password directly from env

        # Initialize Redis connection with explicit parameters
        self.redis = Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        
        self.state_prefix = "instance:state:"
        self.rate_limit_prefix = "instance:rate_limit:"
        
        # Test connection
        try:
            self.redis.ping()
            logger.info(f"Successfully connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
    def get_state(self, name: str) -> Optional[InstanceState]:
        """Get instance state with rate limit expiration handling."""
        key = f"{self.state_prefix}{name}"
        data = self.redis.get(key)
        if not data:
            # Create fresh state and store it directly without calling update_state
            state = create_instance_state(name)
            self.redis.set(key, json.dumps(state.dict()))
            return state
            
        try:
            state_data = json.loads(data)
            # Check if rate limited and if it should be cleared
            if state_data.get("status") == "rate_limited":
                rate_limit_key = f"{self.rate_limit_prefix}{name}"
                if not self.redis.exists(rate_limit_key):
                    # Rate limit expired, but keep other state information
                    state_data["status"] = "healthy" if state_data.get("error_count", 0) == 0 else "error"
                    state_data["rate_limited_until"] = None
                    # Store updated state directly
                    self.redis.set(key, json.dumps(state_data))
            return InstanceState(**state_data)
        except Exception as e:
            logger.error(f"Error deserializing state for {name}: {e}")
            state = create_instance_state(name)
            self.redis.set(key, json.dumps(state.dict()))
            return state
            
    def update_state(self, name: str, **kwargs) -> InstanceState:
        """Update instance state with error type awareness."""
        key = f"{self.state_prefix}{name}"
        
        # Get current state directly from Redis
        data = self.redis.get(key)
        if data:
            try:
                current_data = json.loads(data)
                current = InstanceState(**current_data)
            except Exception:
                current = create_instance_state(name)
        else:
            current = create_instance_state(name)
            
        # Update with new values
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, v)
                
        # Special handling for rate limiting
        if kwargs.get("status") == "rate_limited":
            rate_limit_key = f"{self.rate_limit_prefix}{name}"
            rate_limit_until = kwargs.get("rate_limited_until")
            if rate_limit_until:
                ttl = int(rate_limit_until - time.time())
                if ttl > 0:
                    # Only the rate limit key expires, not the main state
                    self.redis.setex(rate_limit_key, ttl, "1")
                    
        # Store updated state (without TTL)
        self.redis.set(key, json.dumps(current.dict()))
        return current
        
    def record_request(self, name: str, success: bool, tokens: int = 0,
                      latency_ms: Optional[float] = None,
                      error: Optional[str] = None,
                      status_code: Optional[int] = None) -> InstanceState:
        """Record request metrics with error type awareness."""
        curr_time = time.time()
        
        # Update metrics atomically
        pipe = self.redis.pipeline()
        key = f"{self.state_prefix}{name}"
        
        try:
            state = self.get_state(name)
            if not state:
                state = create_instance_state(name)
                
            state.last_used = curr_time
            state.total_requests += 1
            
            if success:
                state.successful_requests += 1
                if state.status != "rate_limited":  # Don't clear rate limiting
                    state.status = "healthy"
                state.error_count = 0
                state.current_tpm += tokens
            else:
                state.error_count += 1
                state.last_error = error
                state.last_error_time = curr_time
                
                # Handle different types of errors
                if status_code:
                    if status_code in [401, 403, 404]:
                        # Permanent errors - mark instance as error
                        state.status = "error"
                        logger.error(f"Instance {name} marked as error due to {status_code}: {error}")
                    elif status_code == 429:
                        # Rate limiting - will be handled by update_state
                        pass
                    elif status_code >= 500:
                        # Server errors - mark as error after multiple failures
                        if state.error_count >= 3:
                            state.status = "error"
                            logger.warning(f"Instance {name} marked as error after {state.error_count} server errors")
                
            if latency_ms is not None:
                if state.avg_latency_ms is None:
                    state.avg_latency_ms = latency_ms
                else:
                    state.avg_latency_ms = (state.avg_latency_ms * 0.9) + (latency_ms * 0.1)
                    
            # Store updated state (without TTL)
            pipe.set(key, json.dumps(state.dict()))
            pipe.execute()
            
            return state
            
        except Exception as e:
            logger.error(f"Error recording request for {name}: {e}")
            pipe.reset()
            return create_instance_state(name)
            
    def get_all_states(self) -> Dict[str, InstanceState]:
        """Get all current instance states."""
        states = {}
        for key in self.redis.keys(f"{self.state_prefix}*"):
            name = key.replace(self.state_prefix, "")
            state = self.get_state(name)
            if state:
                states[name] = state
        return states
            
    def clear_error_state(self, name: str) -> bool:
        """
        Manually clear error state for an instance.
        This can be used when an admin wants to re-enable an errored instance.
        """
        state = self.get_state(name)
        if state and state.status == "error":
            state.status = "healthy"
            state.error_count = 0
            state.last_error = None
            state.last_error_time = None
            self.update_state(name, **state.dict())
            return True
        return False
        
    def reset_states_on_startup(self, instance_names: List[str] = None):
        """
        Reset all instance states on server startup.
        Clears all existing states and starts fresh.
        
        Args:
            instance_names: Optional list of instance names to initialize.
                          If None, only clears existing states.
        """
        try:
            # Clear all keys with our prefixes
            for key in self.redis.keys(f"{self.state_prefix}*"):
                self.redis.delete(key)
            for key in self.redis.keys(f"{self.rate_limit_prefix}*"):
                self.redis.delete(key)
            
            logger.info("Cleared all existing states from Redis")
            
            # Initialize fresh states for provided instances
            if instance_names:
                for name in instance_names:
                    state = create_instance_state(name)
                    key = f"{self.state_prefix}{name}"
                    self.redis.set(key, json.dumps(state.dict()))
                logger.info(f"Initialized fresh states for {len(instance_names)} instances")
            
        except Exception as e:
            logger.error(f"Error resetting states on startup: {e}")
        
        # Reset usage metrics and clear rate limits for all states
        for key in self.redis.keys(f"{self.state_prefix}*"):
            try:
                data = self.redis.get(key)
                if data:
                    state_data = json.loads(data)
                    name = key.replace(self.state_prefix, "")
                    
                    # Clear rate limits
                    rate_limit_key = f"{self.rate_limit_prefix}{name}"
                    self.redis.delete(rate_limit_key)
                    
                    # Reset usage metrics
                    state_data["current_tpm"] = 0
                    state_data["current_rpm"] = 0
                    
                    # Only clear rate limited status
                    if state_data.get("status") == "rate_limited":
                        state_data["status"] = "healthy" if state_data.get("error_count", 0) == 0 else "error"
                        state_data["rate_limited_until"] = None
                        
                    # Store updated state directly instead of using update_state
                    self.redis.set(key, json.dumps(state_data))
            except Exception as e:
                logger.error(f"Error resetting state for {key}: {e}") 