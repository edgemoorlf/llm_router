"""
Models for instance configuration and state.

This module defines the core data models for API instances:
- InstanceConfig: Static configuration defined by operators
- InstanceState: Dynamic runtime state and metrics
- InstanceStatus: Enum for instance status values
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

import time

class InstanceStatus(str, Enum):
    """Status of an API instance."""
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited" 
    ERROR = "error"

class InstanceConfig(BaseModel):
    """
    Configuration settings for an OpenAI-compatible service instance.
    
    These settings are defined by operators and should change
    infrequently. They represent how the instance should be configured.
    """
    name: str = Field(..., description="Unique identifier for this instance")
    provider_type: str = Field(default="azure", description="Provider type (azure or generic)")
    api_key: str = Field(..., description="API key for the service")
    api_base: str = Field(..., description="API base URL")
    api_version: str = Field(default="2023-05-15", description="API version")
    proxy_url: Optional[str] = Field(
        default=None,
        description="Proxy URL for HTTP requests (e.g. http://user:pass@host:port)",
        example="http://user:pass@proxyhost:1000"
    )
    priority: int = Field(default=100, description="Priority (lower is higher priority)")
    weight: int = Field(default=100, description="Weight for weighted distribution (higher gets more traffic)")
    max_tpm: int = Field(default=240000, description="Maximum TPM (tokens per minute) for this instance")
    max_input_tokens: int = Field(default=0, description="Maximum input tokens allowed (0=unlimited)")
    supported_models: List[str] = Field(default_factory=list, description="List of models supported by this instance")
    model_deployments: Dict[str, str] = Field(default_factory=dict, description="Mapping of model names to deployment names (for Azure)")
    enabled: bool = Field(default=True, description="Whether this instance is enabled")
    timeout_seconds: float = Field(default=60.0, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries for failed requests")
    rate_limit_enabled: bool = Field(default=True, description="Whether rate limiting is enabled for this instance")

    class Config:
        schema_extra = {
            "example": {
                "name": "azure-east-us",
                "provider_type": "azure",
                "api_base": "https://example.openai.azure.com",
                "api_key": "sk-...",
                "api_version": "2023-05-15",
                "max_tpm": 240000,
                "supported_models": ["gpt-4", "gpt-3.5-turbo"],
                "priority": 100
            }
        }

class InstanceState(BaseModel):
    """
    Dynamic state for an OpenAI-compatible service instance.
    
    These values represent the runtime state of the instance and
    change frequently based on operational metrics and monitoring.
    """
    name: str = Field(..., description="Instance identifier (links to config)")
    
    # Status information
    status: InstanceStatus = Field(default=InstanceStatus.HEALTHY, description="Current instance status")
    health_status: str = Field(default="unknown", description="Health status (healthy, error, rate_limited, unknown)")
    connection_status: str = Field(default="unknown", description="Connection status (connected, disconnected, unknown)")
    
    # Error tracking
    error_count: int = Field(default=0, description="Count of consecutive errors")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    last_error_time: Optional[float] = Field(default=None, description="When the last error occurred")
    
    # Instance-level error statistics
    total_errors_500: int = Field(default=0, description="Total number of 500 errors encountered")
    total_errors_503: int = Field(default=0, description="Total number of 503 errors encountered")
    total_other_errors: int = Field(default=0, description="Total number of other errors encountered")
    error_500_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 500 errors")
    error_503_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 503 errors")
    error_other_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of other errors")
    current_error_rate: float = Field(default=0.0, description="Current error rate (errors/requests)")
    current_500_rate: float = Field(default=0.0, description="Current 500 error rate")
    current_503_rate: float = Field(default=0.0, description="Current 503 error rate")
    
    # Client-level error statistics
    total_client_errors_500: int = Field(default=0, description="Total number of 500 errors returned to clients")
    total_client_errors_503: int = Field(default=0, description="Total number of 503 errors returned to clients")
    total_client_errors_other: int = Field(default=0, description="Total number of other errors returned to clients")
    client_error_500_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of client 500 errors")
    client_error_503_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of client 503 errors")
    client_error_other_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of client other errors")
    current_client_error_rate: float = Field(default=0.0, description="Current client error rate")
    current_client_500_rate: float = Field(default=0.0, description="Current client 500 error rate")
    current_client_503_rate: float = Field(default=0.0, description="Current client 503 error rate")
    
    # Upstream error statistics
    total_upstream_429_errors: int = Field(default=0, description="Total 429 rate limit errors from upstream APIs")
    total_upstream_400_errors: int = Field(default=0, description="Total 400 bad request errors from upstream APIs")
    total_upstream_500_errors: int = Field(default=0, description="Total 500 internal errors from upstream APIs")
    total_upstream_other_errors: int = Field(default=0, description="Total other errors from upstream APIs")
    upstream_429_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 429 errors")
    upstream_400_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 400 errors")
    upstream_500_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 500 errors")
    upstream_other_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of other errors")
    current_upstream_error_rate: float = Field(default=0.0, description="Current upstream error rate")
    current_upstream_429_rate: float = Field(default=0.0, description="Current upstream 429 error rate")
    current_upstream_400_rate: float = Field(default=0.0, description="Current upstream 400 error rate")
    
    # Rate limiting
    rate_limited_until: Optional[float] = Field(default=None, description="Timestamp when rate limit expires")
    
    # Usage metrics
    current_tpm: int = Field(default=0, description="Current tokens per minute usage")
    current_rpm: int = Field(default=0, description="Current requests per minute")
    total_requests: int = Field(default=0, description="Total number of requests processed")
    successful_requests: int = Field(default=0, description="Number of successful requests")
    total_tokens_served: int = Field(default=0, description="Total tokens served by this instance")
    
    # Usage windows
    usage_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of token usage")
    request_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of request counts")
    
    # Performance metrics
    avg_latency_ms: Optional[float] = Field(default=None, description="Average request latency in milliseconds")
    utilization_percentage: float = Field(default=0.0, description="Current utilization as percentage of max_tpm")
    
    # Timestamps
    last_used: float = Field(default=0.0, description="Timestamp when this instance was last used")

    @property
    def is_healthy(self) -> bool:
        """Check if the instance is healthy based on its status."""
        return self.status == InstanceStatus.HEALTHY

    class Config:
        schema_extra = {
            "example": {
                "name": "azure-east-us",
                "status": "healthy",
                "current_tpm": 12500,
                "health_status": "healthy",
                "error_count": 0,
                "utilization_percentage": 20.8,
                "last_used": 1642097263.45
            }
        }

def create_instance_state(name: str) -> InstanceState:
    """
    Create a new instance state with default values.
    
    Args:
        name: The instance name
        
    Returns:
        A new InstanceState object with default values
    """
    return InstanceState(
        name=name,
        status=InstanceStatus.HEALTHY,
        health_status="unknown",
        last_used=time.time()
    )
