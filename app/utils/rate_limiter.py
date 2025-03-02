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
DEFAULT_TOKEN_RATE_LIMIT = 30000

class TokenUsage(BaseModel):
    """Model for tracking token usage."""
    
    prompt_tokens: int = Field(default=0, description="Number of prompt tokens used")
    completion_tokens: int = Field(default=0, description="Number of completion tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")

class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    def check_and_update(self, tokens: int) -> Tuple[bool, int]:
        """
        Check if the request can be processed and update token count.
        
        Args:
            tokens: Number of tokens to be used
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate the number of tokens in the given text for the specified model.
        
        Args:
            text: Text to estimate tokens for
            model: Model to use for token counting
            
        Returns:
            Estimated token count
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the rate limiter."""
        pass


class InMemoryRateLimiter(RateLimiter):
    """In-memory implementation of rate limiter based on a sliding window."""
    
    def __init__(self, tokens_per_minute: int = DEFAULT_TOKEN_RATE_LIMIT):
        """
        Initialize the in-memory rate limiter.
        
        Args:
            tokens_per_minute: Token rate limit per minute
        """
        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = 60  # 1 minute window
        self.usage_window: Dict[int, int] = {}  # timestamp (s) -> tokens used
        self.lock = threading.Lock()
        self.encoding_cache: Dict[str, tiktoken.Encoding] = {}
        
        logger.info(f"Initialized in-memory rate limiter with {tokens_per_minute} tokens per minute")
    
    def check_and_update(self, tokens: int) -> Tuple[bool, int]:
        """
        Check if the request can be processed under the rate limit.
        
        Args:
            tokens: Number of tokens to be used
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        with self.lock:
            current_time = int(time.time())
            cutoff_time = current_time - self.window_seconds
            
            # Clean up old entries
            self.usage_window = {ts: tokens for ts, tokens in self.usage_window.items() if ts >= cutoff_time}
            
            # Calculate current token usage in the window
            current_usage = sum(self.usage_window.values())
            
            # Check if adding these tokens would exceed the limit
            if current_usage + tokens > self.tokens_per_minute:
                # Calculate when tokens will be available
                oldest_timestamp = min(self.usage_window.keys()) if self.usage_window else current_time
                seconds_until_reset = oldest_timestamp + self.window_seconds - current_time
                logger.warning(f"Rate limit exceeded: {current_usage}/{self.tokens_per_minute} tokens used. "
                              f"Retry after {seconds_until_reset} seconds")
                return False, seconds_until_reset
            
            # Update usage window
            self.usage_window[current_time] = self.usage_window.get(current_time, 0) + tokens
            return True, 0
    
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate the number of tokens in the given text.
        
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
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        with self.lock:
            self.usage_window.clear()


class RedisRateLimiter(RateLimiter):
    """Redis-based implementation of rate limiter for distributed environments."""
    
    def __init__(
        self, 
        tokens_per_minute: int = DEFAULT_TOKEN_RATE_LIMIT,
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize the Redis rate limiter.
        
        Args:
            tokens_per_minute: Token rate limit per minute
            redis_url: URL for Redis connection
        """
        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = 60  # 1 minute window
        self.redis_key_prefix = "azure_openai_proxy:rate_limit"
        self.redis = redis.from_url(redis_url)
        self.encoding_cache: Dict[str, tiktoken.Encoding] = {}
        
        logger.info(f"Initialized Redis rate limiter with {tokens_per_minute} tokens per minute")
    
    def check_and_update(self, tokens: int) -> Tuple[bool, int]:
        """
        Check if the request can be processed under the rate limit.
        
        Args:
            tokens: Number of tokens to be used
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        current_time = int(time.time())
        cutoff_time = current_time - self.window_seconds
        window_key = f"{self.redis_key_prefix}:window"
        
        # Execute in a pipeline for atomicity
        with self.redis.pipeline() as pipe:
            # Clean up old entries and calculate current usage
            pipe.zremrangebyscore(window_key, 0, cutoff_time)
            pipe.zrange(window_key, 0, -1, withscores=True)
            _, window_data = pipe.execute()
            
            # Sum up current token usage
            current_usage = sum(int(item[0]) for item in window_data)
            
            # Check if adding these tokens would exceed the limit
            if current_usage + tokens > self.tokens_per_minute:
                if window_data:
                    oldest_timestamp = min(int(score) for _, score in window_data)
                    seconds_until_reset = oldest_timestamp + self.window_seconds - current_time
                else:
                    seconds_until_reset = 0
                    
                logger.warning(f"Rate limit exceeded: {current_usage}/{self.tokens_per_minute} tokens used. "
                              f"Retry after {seconds_until_reset} seconds")
                return False, max(1, seconds_until_reset)
            
            # Update usage window
            self.redis.zadd(window_key, {str(tokens): current_time})
            # Set expiry on the sorted set to auto-cleanup
            self.redis.expire(window_key, self.window_seconds * 2)
            
            return True, 0
    
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate the number of tokens in the given text.
        
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
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        window_key = f"{self.redis_key_prefix}:window"
        self.redis.delete(window_key)


def get_rate_limiter() -> RateLimiter:
    """Factory function to get the configured rate limiter."""
    tokens_per_minute = int(os.getenv("TOKEN_RATE_LIMIT", DEFAULT_TOKEN_RATE_LIMIT))
    
    use_redis = os.getenv("USE_REDIS_RATE_LIMITER", "false").lower() == "true"
    if use_redis:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        return RedisRateLimiter(tokens_per_minute=tokens_per_minute, redis_url=redis_url)
    else:
        return InMemoryRateLimiter(tokens_per_minute=tokens_per_minute)

# Create a singleton instance
rate_limiter = get_rate_limiter()
