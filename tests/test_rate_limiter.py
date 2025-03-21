import pytest
import time
import redis
from unittest.mock import MagicMock, patch

from app.utils.rate_limiter import RedisRateLimiter


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    # Create a mock Redis client
    mock_client = MagicMock(spec=redis.Redis)
    
    # Create a mock pipeline that will be returned by pipeline()
    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = [1, []]  # Default return value for pipeline.execute()
    
    # Make pipeline() return our mock pipeline
    mock_client.pipeline.return_value.__enter__.return_value = mock_pipeline
    mock_client.pipeline.return_value.__exit__.return_value = None
    
    return mock_client, mock_pipeline


@pytest.fixture
def rate_limiter(mock_redis_client):
    """Create a RedisRateLimiter instance with mock Redis."""
    mock_client, _ = mock_redis_client
    
    with patch('redis.Redis', return_value=mock_client):
        limiter = RedisRateLimiter(
            instance_id="test-instance",
            tokens_per_minute=1000,
            redis_url="redis://localhost:6379",
            redis_password="",
            max_input_tokens=8000
        )
        # Replace the Redis client with our mock
        limiter.redis = mock_client
        # Set the key prefix for easier testing
        limiter.key_prefix = "test:rate_limit:"
        limiter.redis_key = f"{limiter.key_prefix}{limiter.instance_id}"
        return limiter


class TestRedisRateLimiter:
    """Tests for the RedisRateLimiter class."""

    def test_check_capacity_and_update_usage_separation(self, rate_limiter, mock_redis_client):
        """Test that check_capacity and update_usage work correctly as separate operations."""
        mock_client, mock_pipeline = mock_redis_client
        
        # Mock responses for check_capacity
        mock_pipeline.execute.return_value = [1, []]  # Empty list for zrange result
        
        # Test check_capacity with empty window
        result, retry_after = rate_limiter.check_capacity(50)
        assert result is True
        
        # Verify pipeline commands were called
        mock_pipeline.zremrangebyscore.assert_called_once()
        mock_pipeline.zrange.assert_called_once()
        mock_client.zadd.assert_not_called()  # Should not call zadd
        
        # Reset mocks
        mock_pipeline.reset_mock()
        mock_client.reset_mock()
        
        # Test update_usage
        rate_limiter.update_usage(50)
        
        # Verify that zadd was called during update_usage
        mock_client.zadd.assert_called_once()
        key_arg = mock_client.zadd.call_args[0][0]
        assert key_arg.startswith("test:rate_limit:")
        
        # Reset mocks
        mock_pipeline.reset_mock()
        mock_client.reset_mock()
        
        # Mock a window that exceeds rate limit
        mock_pipeline.execute.return_value = [
            1,  # Result of zremrangebyscore
            [(b"100", 1000.0)]  # Result of zrange - simulating 100 tokens already used
        ]
        
        # Set tokens_per_minute to 120 to trigger rate limiting (100 + 50 > 120)
        rate_limiter.tokens_per_minute = 120
        
        # Test check_capacity with window that exceeds limit
        result, retry_after = rate_limiter.check_capacity(50)
        assert result is False
        assert retry_after > 0
        
        # Verify pipeline commands were called
        mock_pipeline.zremrangebyscore.assert_called_once()
        mock_pipeline.zrange.assert_called_once()
        mock_client.zadd.assert_not_called()  # Should not call zadd
    
    def test_check_and_update(self, rate_limiter, mock_redis_client):
        """Test the combined check_and_update method."""
        mock_client, mock_pipeline = mock_redis_client
        
        # Mock responses for successful check
        mock_pipeline.execute.return_value = [1, []]  # Empty list for zrange result
        
        # Test successful check_and_update
        result, retry_after = rate_limiter.check_and_update(50)
        assert result is True
        
        # Verify that both pipeline commands and zadd were called
        mock_pipeline.zremrangebyscore.assert_called_once()
        mock_pipeline.zrange.assert_called_once()
        mock_client.zadd.assert_called_once()
        
        # Reset mocks
        mock_pipeline.reset_mock()
        mock_client.reset_mock()
        
        # Mock a window that exceeds rate limit
        mock_pipeline.execute.return_value = [
            1,  # Result of zremrangebyscore
            [(b"100", 1000.0)]  # Result of zrange - simulating 100 tokens already used
        ]
        
        # Set tokens_per_minute to 120 to trigger rate limiting (100 + 50 > 120)
        rate_limiter.tokens_per_minute = 120
        
        # Test failed check_and_update
        result, retry_after = rate_limiter.check_and_update(50)
        assert result is False
        assert retry_after > 0
        
        # Verify that pipeline commands were called but zadd was not
        mock_pipeline.zremrangebyscore.assert_called_once()
        mock_pipeline.zrange.assert_called_once()
        mock_client.zadd.assert_not_called()
    
    def test_unique_token_entries(self, rate_limiter, mock_redis_client):
        """Test that each token entry is unique even when called in rapid succession."""
        mock_client, _ = mock_redis_client
        
        # Capture the keys used in zadd calls
        keys = []
        
        def mock_zadd(key, mapping, **kwargs):
            # Extract the token key from the mapping dict
            for token_key in mapping.keys():
                keys.append(token_key)
            return 1
            
        # Configure the mock to capture the keys
        mock_client.zadd.side_effect = mock_zadd
        
        # Mock time to ensure we can control the timestamps
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            
            # Call update_usage multiple times in rapid succession
            rate_limiter.update_usage(10)
            rate_limiter.update_usage(10)
            rate_limiter.update_usage(10)
            
            # Check that zadd was called with unique keys each time
            assert mock_client.zadd.call_count == 3
            
            # Verify all values contain the token amount "10"
            assert all(k.startswith("10:") for k in keys)
            
            # Extract timestamps from the keys
            timestamps = [k.split(":")[1] for k in keys]
            
            # Verify timestamps are unique
            assert len(set(timestamps)) == 3, "Timestamps should be unique" 