"""Rate limiters for controlling API token usage."""
import os
import time
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import redis
import tiktoken
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default token rate limit (tokens per minute)
DEFAULT_TOKEN_RATE_LIMIT = 1000000
DEFAULT_MAX_INPUT_TOKENS_LIMIT = 16384

class TokenUsage(BaseModel):
    """Model for tracking token usage."""
    
    prompt_tokens: int = Field(default=0, description="Number of prompt tokens used")
    completion_tokens: int = Field(default=0, description="Number of completion tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")

class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    def __init__(self, instance_id: str = "global"):
        """
        Initialize the base rate limiter.
        
        Args:
            instance_id: Unique identifier for the instance this limiter belongs to
                        (defaults to "global" for backward compatibility)
        """
        self.instance_id = instance_id
        self.encoding_cache: Dict[str, tiktoken.Encoding] = {}
    
    @abstractmethod
    def check_and_update(self, tokens: int, instance_id: Optional[str] = None) -> Tuple[bool, int]:
        """
        Check if the request is allowed and update token count.
        
        Args:
            tokens: Number of tokens to check/update
            instance_id: Optional instance ID for per-instance rate limiting
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        pass
    
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate the number of tokens in the given text for the specified model.
        
        Args:
            text: Text to estimate tokens for
            model: Model to use for token counting
            
        Returns:
            Estimated token count
        """
        try:
            # Use tiktoken for accurate token counting
            if model not in self.encoding_cache:
                try:
                    self.encoding_cache[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fall back to cl100k_base encoding for new models
                    self.encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
            
            encoding = self.encoding_cache[model]
            token_count = len(encoding.encode(text))
            return token_count
        except Exception as e:
            logger.warning(f"Error estimating tokens: {str(e)}. Using character-based estimation.")
            # Fallback to character-based estimation
            return len(text) // 4  # Rough estimate: 4 chars per token
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the rate limiter."""
        pass


class InMemoryRateLimiter(RateLimiter):
    """In-memory implementation of rate limiter based on a sliding window."""
    
    def __init__(self, 
                 instance_id: str = "global",
                 tokens_per_minute: int = DEFAULT_TOKEN_RATE_LIMIT, 
                 max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS_LIMIT):
        """
        Initialize the in-memory rate limiter.
        
        Args:
            instance_id: Unique identifier for the instance
            tokens_per_minute: Token rate limit per minute
            max_input_tokens: Maximum allowed tokens per request
        """
        super().__init__(instance_id)
        self.tokens_per_minute = tokens_per_minute
        self.max_input_tokens = max_input_tokens
        self.window_seconds = 60  # 1 minute window
        self.usage_windows: Dict[str, Dict[int, int]] = {}
        self.lock = threading.Lock()
        
        logger.info(f"Initialized in-memory rate limiter for instance {instance_id} "
                   f"with {tokens_per_minute} tokens per minute")
    
    def check_and_update(self, tokens: int, instance_id: Optional[str] = None) -> Tuple[bool, int]:
        """
        Check if the request is allowed and update token count.
        
        Args:
            tokens: Number of tokens to check/update
            instance_id: Optional instance ID for per-instance rate limiting
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if self.max_input_tokens > 0 and tokens > self.max_input_tokens:
            logger.warning(f"Request exceeds maximum allowed tokens for instance {self.instance_id}: "
                         f"{tokens} > {self.max_input_tokens}")
            return False, 60
            
        current_time = int(time.time())
        window_key = instance_id or self.instance_id
        
        # Initialize window for this instance if not exists
        if window_key not in self.usage_windows:
            self.usage_windows[window_key] = {}
            
        window = self.usage_windows[window_key]
        
        # Remove entries older than 1 minute
        cutoff = current_time - 60
        window = {ts: count for ts, count in window.items() if ts > cutoff}
        self.usage_windows[window_key] = window
        
        # Calculate current token usage
        current_usage = sum(window.values())
        
        # Check if adding these tokens would exceed the limit
        if current_usage + tokens > self.tokens_per_minute:
            # Find the oldest timestamp that's still counted
            oldest = min(window.keys()) if window else current_time
            retry_after = oldest - cutoff
            logger.warning(f"Rate limit exceeded for instance {self.instance_id}: "
                         f"{current_usage}/{self.tokens_per_minute} tokens used. "
                         f"Retry after {retry_after} seconds")
            return False, max(1, retry_after)
            
        # Update the window with the new tokens
        window[current_time] = window.get(current_time, 0) + tokens
        return True, 0
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        with self.lock:
            self.usage_windows.clear()


class RedisRateLimiter(RateLimiter):
    """Redis-based implementation of rate limiter for distributed environments."""
    
    def __init__(
        self, 
        instance_id: str = "global",
        tokens_per_minute: int = DEFAULT_TOKEN_RATE_LIMIT,
        redis_url: str = "redis://localhost:6379",
        redis_password: str = "",
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS_LIMIT
    ):
        """
        Initialize the Redis rate limiter.
        
        Args:
            instance_id: Unique identifier for the instance
            tokens_per_minute: Token rate limit per minute
            redis_url: URL for Redis connection
            redis_password: Redis password
            max_input_tokens: Maximum allowed tokens per request
        """
        super().__init__(instance_id)
        self.tokens_per_minute = tokens_per_minute
        self.max_input_tokens = max_input_tokens
        self.window_seconds = 60  # 1 minute window
        # Each instance gets its own Redis key space
        self.redis_key = f"instance:rate_limit:window:{instance_id}"
        self.redis = redis.from_url(url=redis_url, password=redis_password)
        
        logger.info(f"Initialized Redis rate limiter for instance {instance_id} "
                   f"with {tokens_per_minute} tokens per minute")
    
    def get_current_usage(self) -> int:
        """
        Get the current token usage within the rate limit window.
        
        Returns:
            Current token usage in tokens per minute
        """
        current_time = int(time.time())
        cutoff = current_time - self.window_seconds
        
        try:
            # Remove old entries first
            self.redis.zremrangebyscore(self.redis_key, "-inf", cutoff)
            
            # Get all entries within the window
            entries = self.redis.zrange(self.redis_key, 0, -1, withscores=True)
            
            # Calculate total token usage from the entries
            current_usage = 0
            for value, _ in entries:
                try:
                    token_value = value.decode().split(":")[0]
                    current_usage += int(token_value)
                except (ValueError, IndexError, AttributeError) as e:
                    logger.error(f"Error parsing token value '{value}': {e}")
            
            if entries:
                logger.debug(f"Current usage for instance {self.instance_id}: {current_usage}/{self.tokens_per_minute} tokens")
            
            return current_usage
            
        except redis.RedisError as e:
            logger.error(f"Redis error getting usage for {self.instance_id}: {e}")
            return 0
    
    def check_capacity(self, tokens: int) -> Tuple[bool, int]:
        """
        Check if the instance has capacity without updating usage.
        
        Args:
            tokens: Number of tokens to check
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if self.max_input_tokens > 0 and tokens > self.max_input_tokens:
            logger.warning(f"Request exceeds maximum allowed tokens for instance {self.instance_id}: "
                         f"{tokens} > {self.max_input_tokens}")
            return False, 60
        
        current_time = int(time.time())
        
        try:
            with self.redis.pipeline() as pipe:
                # Remove entries older than window period
                cutoff = current_time - self.window_seconds
                pipe.zremrangebyscore(self.redis_key, "-inf", cutoff)
                
                # Get current usage
                pipe.zrange(self.redis_key, 0, -1, withscores=True)
                
                # Execute commands
                _, entries = pipe.execute()
                
                # Calculate current usage
                current_usage = 0
                for entry, _ in entries:
                    if isinstance(entry, bytes):
                        entry = entry.decode('utf-8')
                    tokens_str = entry.split(':')[0]
                    current_usage += int(tokens_str)
                
                # Check if adding these tokens would exceed the limit
                if current_usage + tokens > self.tokens_per_minute:
                    oldest = min((ts for _, ts in entries), default=current_time)
                    retry_after = oldest - cutoff
                    logger.warning(f"Rate limit exceeded for instance {self.instance_id}: "
                                 f"{current_usage}/{self.tokens_per_minute} tokens used. "
                                 f"Retry after {retry_after} seconds")
                    return False, max(1, retry_after)
                
                if current_usage > 0:
                    logger.debug(f"Instance {self.instance_id} has capacity: {current_usage}/{self.tokens_per_minute} tokens used")
                return True, 0
                
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter for {self.instance_id}: {e}")
            # Fail open if Redis is unavailable
            return True, 0

    def update_usage(self, tokens: int) -> None:
        """
        Update token usage for the instance after a successful request.
        
        Args:
            tokens: Number of tokens to add
        """
        current_time = int(time.time())
        
        try:
            # Add the new tokens
            unique_token_key = f"{tokens}:{time.time_ns()}"
            self.redis.zadd(self.redis_key, {unique_token_key: current_time})
            
            # Remove explicit TTL - we'll manage expiration through zremrangebyscore
            # Instead, just clean up old entries beyond our window
            cutoff = current_time - self.window_seconds
            self.redis.zremrangebyscore(self.redis_key, "-inf", cutoff)
            
            # Check current usage after updating (for debugging)
            current_usage = self.get_current_usage()
            
            logger.debug(f"Updated usage for instance {self.instance_id}: +{tokens} tokens, "
                        f"current usage: {current_usage}/{self.tokens_per_minute}")
        except redis.RedisError as e:
            logger.error(f"Redis error updating usage for {self.instance_id}: {e}")

    def check_and_update(self, tokens: int, instance_id: Optional[str] = None) -> Tuple[bool, int]:
        """
        Check if the request is allowed and update token count.
        
        Args:
            tokens: Number of tokens to check/update
            instance_id: Optional instance ID for per-instance rate limiting (ignored, using self.instance_id)
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        # Use the new methods
        allowed, retry_after = self.check_capacity(tokens)
        if allowed:
            self.update_usage(tokens)
        return allowed, retry_after
    
    def reset(self) -> None:
        """Reset the rate limiter by clearing all usage data."""
        try:
            result = self.redis.delete(self.redis_key)
            logger.info(f"Explicitly reset rate limiter for instance {self.instance_id} (deleted {result} keys)")
        except redis.RedisError as e:
            logger.error(f"Redis error resetting rate limiter for {self.instance_id}: {e}")


def get_rate_limiter(
    instance_id: str = "global",
    tokens_per_minute: int = DEFAULT_TOKEN_RATE_LIMIT,
    use_redis: bool = False,
    redis_url: str = "redis://localhost:6379",
    redis_password: str = "",
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS_LIMIT
) -> RateLimiter:
    """
    Factory function to create a rate limiter for a specific instance.
    
    Args:
        instance_id: Unique identifier for the instance (defaults to "global" for backward compatibility)
        tokens_per_minute: Token rate limit per minute
        use_redis: Whether to use Redis-based rate limiter
        redis_url: URL for Redis connection
        redis_password: Redis password
        max_input_tokens: Maximum allowed tokens per request
        
    Returns:
        Configured rate limiter instance
    """
    if use_redis:
        return RedisRateLimiter(
            instance_id=instance_id,
            tokens_per_minute=tokens_per_minute,
            redis_url=redis_url,
            redis_password=redis_password,
            max_input_tokens=max_input_tokens
        )
    return InMemoryRateLimiter(
        instance_id=instance_id,
        tokens_per_minute=tokens_per_minute,
        max_input_tokens=max_input_tokens
    )

# Create a default global rate limiter instance for backward compatibility
rate_limiter = get_rate_limiter(
    instance_id="global",
    tokens_per_minute=int(os.environ.get("TOKEN_RATE_LIMIT", DEFAULT_TOKEN_RATE_LIMIT)),
    use_redis=os.environ.get("USE_REDIS", "").lower() in ("true", "1", "yes"),
    redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
    redis_password=os.environ.get("REDIS_PASSWORD", ""),
    max_input_tokens=int(os.environ.get("MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS_LIMIT))
)
