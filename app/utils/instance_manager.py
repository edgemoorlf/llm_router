"""Instance manager for multiple Azure OpenAI service instances."""
import os
import time
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
import httpx
from fastapi import HTTPException, status
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RoutingStrategy(str, Enum):
    """Strategy used to select an Azure OpenAI instance."""
    
    PRIORITY = "priority"  # Use instance with highest priority (lowest number)
    ROUND_ROBIN = "round_robin"  # Simple round-robin distribution
    WEIGHTED = "weighted"  # Weight-based distribution
    LEAST_LOADED = "least_loaded"  # Use instance with lowest TPM consumption
    FAILOVER = "failover"  # Try instances in priority order until one succeeds


class InstanceStatus(str, Enum):
    """Status of an Azure OpenAI instance."""
    
    HEALTHY = "healthy"  # Instance is operational
    RATE_LIMITED = "rate_limited"  # Instance is rate limited
    ERROR = "error"  # Instance has encountered an error


class AzureOpenAIInstance(BaseModel):
    """Configuration and state for an Azure OpenAI service instance."""
    
    name: str = Field(..., description="Unique identifier for this instance")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_base: str = Field(..., description="Azure OpenAI API base URL")
    api_version: str = Field(..., description="Azure OpenAI API version")
    priority: int = Field(default=100, description="Priority (lower is higher priority)")
    weight: int = Field(default=100, description="Weight for weighted distribution (higher gets more traffic)")
    max_tpm: int = Field(default=240000, description="Maximum TPM (tokens per minute) for this instance")
    
    # Runtime state
    status: InstanceStatus = Field(default=InstanceStatus.HEALTHY, description="Current instance status")
    current_tpm: int = Field(default=0, description="Current TPM usage (resets each minute)")
    error_count: int = Field(default=0, description="Consecutive error count")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    rate_limited_until: Optional[float] = Field(default=None, description="Timestamp when rate limit expires")
    usage_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of token usage (timestamp -> tokens)")
    client: Optional[Any] = Field(default=None, exclude=True, description="HTTP client for this instance")
    last_used: float = Field(default=0.0, description="Timestamp when this instance was last used")

    def initialize_client(self) -> None:
        """Initialize the HTTP client for this instance."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),  # 5 minutes total timeout
                headers={"api-key": self.api_key},
            )
    
    def update_tpm_usage(self, tokens: int) -> None:
        """
        Update the TPM usage for this instance.
        
        Args:
            tokens: Number of tokens used
        """
        current_time = int(time.time())
        window_start = current_time - 60  # 1 minute window
        
        # Clean up old entries
        self.usage_window = {ts: usage for ts, usage in self.usage_window.items() if ts >= window_start}
        
        # Add new usage
        self.usage_window[current_time] = self.usage_window.get(current_time, 0) + tokens
        
        # Update current TPM
        self.current_tpm = sum(self.usage_window.values())
    
    def is_rate_limited(self) -> bool:
        """Check if this instance is currently rate limited."""
        # Check if we're in a rate-limited state
        if self.status == InstanceStatus.RATE_LIMITED:
            current_time = time.time()
            
            # Check if rate limit has expired
            if self.rate_limited_until and current_time >= self.rate_limited_until:
                logger.info(f"Instance {self.name} rate limit has expired, marking as healthy")
                self.status = InstanceStatus.HEALTHY
                self.rate_limited_until = None
                return False
            
            return True
        
        # Check if we're approaching the TPM limit
        if self.current_tpm >= self.max_tpm * 0.95:  # 95% of max TPM
            logger.warning(f"Instance {self.name} is approaching TPM limit: {self.current_tpm}/{self.max_tpm}")
            return True
        
        return False
    
    def mark_rate_limited(self, retry_after: Optional[int] = None) -> None:
        """
        Mark this instance as rate limited.
        
        Args:
            retry_after: Seconds until the rate limit expires
        """
        self.status = InstanceStatus.RATE_LIMITED
        
        if retry_after is None:
            retry_after = 60  # Default to 1 minute
        
        self.rate_limited_until = time.time() + retry_after
        logger.warning(f"Instance {self.name} marked as rate limited for {retry_after} seconds")
    
    def mark_error(self, error_message: str) -> None:
        """
        Mark this instance as having an error.
        
        Args:
            error_message: Error message
        """
        self.error_count += 1
        self.last_error = error_message
        
        if self.error_count >= 3:  # Three strikes and you're out
            self.status = InstanceStatus.ERROR
            logger.error(f"Instance {self.name} marked as error after {self.error_count} consecutive errors")
    
    def mark_healthy(self) -> None:
        """Mark this instance as healthy."""
        self.status = InstanceStatus.HEALTHY
        self.error_count = 0
        self.last_error = None
        self.rate_limited_until = None
    
    def build_azure_url(self, endpoint: str, deployment_name: str) -> str:
        """
        Build the Azure OpenAI API URL for this instance.
        
        Args:
            endpoint: The API endpoint
            deployment_name: The Azure deployment name
            
        Returns:
            The full Azure OpenAI API URL
        """
        # Map OpenAI endpoints to Azure endpoints
        endpoint_mapping = {
            "/v1/chat/completions": f"/openai/deployments/{deployment_name}/chat/completions",
            "/v1/completions": f"/openai/deployments/{deployment_name}/completions",
            "/v1/embeddings": f"/openai/deployments/{deployment_name}/embeddings",
        }
        
        azure_endpoint = endpoint_mapping.get(endpoint)
        if not azure_endpoint:
            # For unsupported endpoints, try to extract the operation name
            parts = endpoint.split("/")
            if len(parts) >= 3 and parts[-2] == "deployments":
                # Already in Azure format
                return f"{self.api_base}{endpoint}"
            else:
                # Unknown endpoint format
                raise ValueError(f"Unsupported endpoint: {endpoint}")
        
        return f"{self.api_base}{azure_endpoint}?api-version={self.api_version}"


class InstanceManager:
    """Manager for multiple Azure OpenAI service instances with load balancing and failover."""
    
    def __init__(self, routing_strategy: RoutingStrategy = RoutingStrategy.FAILOVER):
        """
        Initialize the instance manager.
        
        Args:
            routing_strategy: Strategy for selecting instances
        """
        self.instances: Dict[str, AzureOpenAIInstance] = {}
        self.routing_strategy = routing_strategy
        self.round_robin_index = 0
        self._load_instances_from_env()
        
        if not self.instances:
            logger.warning("No Azure OpenAI instances configured. Falling back to legacy single instance configuration.")
            self._load_legacy_instance()
        
        logger.info(f"Initialized {len(self.instances)} Azure OpenAI instances with {routing_strategy} routing strategy")
    
    def _load_instances_from_env(self) -> None:
        """Load instances from environment variables."""
        instance_names = os.getenv("AZURE_INSTANCES", "").split(",")
        instance_names = [name.strip() for name in instance_names if name.strip()]        
        
        for name in instance_names:
            prefix = f"AZURE_INSTANCE_{name.upper()}_"
            
            if not os.getenv(f"{prefix}API_KEY") or not os.getenv(f"{prefix}API_BASE"):
                logger.warning(f"Skipping incomplete instance configuration for {name}")
                continue
            
            try:
                instance = AzureOpenAIInstance(
                    name=name,
                    api_key=os.getenv(f"{prefix}API_KEY", ""),
                    api_base=os.getenv(f"{prefix}API_BASE", ""),
                    api_version=os.getenv(f"{prefix}API_VERSION", "2023-07-01-preview"),
                    priority=int(os.getenv(f"{prefix}PRIORITY", "100")),
                    weight=int(os.getenv(f"{prefix}WEIGHT", "100")),
                    max_tpm=int(os.getenv(f"{prefix}MAX_TPM", "240000")),
                )
                instance.initialize_client()
                self.instances[name] = instance
                logger.info(f"Loaded Azure OpenAI instance {name} with priority {instance.priority}, weight {instance.weight}, max TPM {instance.max_tpm}")
            except Exception as e:
                logger.error(f"Error loading instance {name}: {str(e)}")
    
    def _load_legacy_instance(self) -> None:
        """Load the legacy single instance configuration."""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_API_BASE")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        
        if not api_key or not api_base:
            logger.error("No Azure OpenAI instances configured and legacy configuration is incomplete")
            return
        
        instance = AzureOpenAIInstance(
            name="default",
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            priority=1,
            weight=100,
            max_tpm=240000,  # Default to 240K TPM
        )
        instance.initialize_client()
        self.instances["default"] = instance
        logger.info("Loaded legacy Azure OpenAI instance configuration")
    
    def select_instance(self, required_tokens: int) -> Optional[AzureOpenAIInstance]:
        """
        Select an Azure OpenAI instance based on the configured routing strategy.
        
        Args:
            required_tokens: Estimated number of tokens required for the request
            
        Returns:
            Selected instance or None if no suitable instance is available
        """
        # Filter healthy instances with enough TPM capacity
        available_instances = [
            instance for instance in self.instances.values()
            if (instance.status == InstanceStatus.HEALTHY or 
                (instance.status == InstanceStatus.RATE_LIMITED and time.time() >= (instance.rate_limited_until or 0)))
            and (instance.current_tpm + required_tokens) <= instance.max_tpm
        ]
        
        if not available_instances:
            logger.warning(f"No healthy instances available with enough TPM capacity for {required_tokens} tokens")
            # Try to find any instance that's not in error state as a fallback
            available_instances = [
                instance for instance in self.instances.values()
                if instance.status != InstanceStatus.ERROR
            ]
            
            if not available_instances:
                logger.error("No available instances found")
                return None
        
        # If we only have one instance, return it
        if len(available_instances) == 1:
            return available_instances[0]
        
        # Select instance based on routing strategy
        if self.routing_strategy == RoutingStrategy.PRIORITY:
            # Sort by priority (lower is higher priority)
            available_instances.sort(key=lambda x: x.priority)
            return available_instances[0]
            
        elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin
            self.round_robin_index = (self.round_robin_index + 1) % len(available_instances)
            return available_instances[self.round_robin_index]
            
        elif self.routing_strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection
            total_weight = sum(instance.weight for instance in available_instances)
            if total_weight == 0:
                # If all weights are 0, use simple round-robin
                self.round_robin_index = (self.round_robin_index + 1) % len(available_instances)
                return available_instances[self.round_robin_index]
            
            # Weighted random selection
            r = random.uniform(0, total_weight)
            upto = 0
            for instance in available_instances:
                upto += instance.weight
                if upto >= r:
                    return instance
            
            # Fallback to first instance
            return available_instances[0]
            
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            # Sort by current TPM usage (lower is better)
            available_instances.sort(key=lambda x: x.current_tpm)
            return available_instances[0]
            
        elif self.routing_strategy == RoutingStrategy.FAILOVER:
            # Sort by priority (lower is higher priority) for failover
            available_instances.sort(key=lambda x: x.priority)
            return available_instances[0]
            
        # Default to first instance
        return available_instances[0]
    
    async def try_instances(self, 
                           endpoint: str, 
                           deployment: str, 
                           payload: Dict[str, Any], 
                           required_tokens: int,
                           method: str = "POST") -> Tuple[Dict[str, Any], AzureOpenAIInstance]:
        """
        Try instances until one succeeds or all fail.
        
        Args:
            endpoint: The API endpoint
            deployment: The Azure deployment name
            payload: The request payload
            required_tokens: Estimated tokens required for the request
            method: HTTP method
            
        Returns:
            Tuple of (response, instance used)
            
        Raises:
            HTTPException: If all instances fail
        """
        # Filter out instances in error state
        available_instances = [instance for instance in self.instances.values() 
                               if instance.status != InstanceStatus.ERROR]
        
        if not available_instances:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No available Azure OpenAI instances",
            )
        
        # Try strategy-based instance first
        primary_instance = self.select_instance(required_tokens)
        if primary_instance:
            try:
                response = await self._forward_request(primary_instance, endpoint, deployment, payload, method)
                # Mark instance as healthy and update TPM
                primary_instance.mark_healthy()
                if "usage" in response and "total_tokens" in response["usage"]:
                    primary_instance.update_tpm_usage(response["usage"]["total_tokens"])
                return response, primary_instance
            except HTTPException as e:
                status_code = e.status_code
                detail = e.detail
                
                # Handle rate limiting specifically
                if status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                    retry_after = None
                    # Extract retry-after from headers if available
                    if hasattr(e, 'headers') and e.headers and 'retry-after' in e.headers:
                        try:
                            retry_after = int(e.headers['retry-after'])
                        except (ValueError, TypeError):
                            pass
                    
                    primary_instance.mark_rate_limited(retry_after)
                    logger.warning(f"Instance {primary_instance.name} rate limited: {detail}")
                else:
                    primary_instance.mark_error(str(e))
                    logger.error(f"Error from instance {primary_instance.name}: {detail}")
                
                # Fall through to try other instances
            except Exception as e:
                primary_instance.mark_error(str(e))
                logger.error(f"Unexpected error from instance {primary_instance.name}: {str(e)}")
                # Fall through to try other instances
        
        # If primary instance failed or not found, try all other instances in priority order
        available_instances.sort(key=lambda x: x.priority)
        errors = []
        
        for instance in available_instances:
            # Skip the instance we already tried
            if primary_instance and instance.name == primary_instance.name:
                continue
                
            # Skip instances that are rate limited
            if instance.is_rate_limited():
                logger.debug(f"Skipping rate-limited instance {instance.name}")
                continue
                
            try:
                response = await self._forward_request(instance, endpoint, deployment, payload, method)
                # Mark instance as healthy and update TPM
                instance.mark_healthy()
                if "usage" in response and "total_tokens" in response["usage"]:
                    instance.update_tpm_usage(response["usage"]["total_tokens"])
                return response, instance
            except HTTPException as e:
                status_code = e.status_code
                detail = e.detail
                
                # Handle rate limiting specifically
                if status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                    retry_after = None
                    if hasattr(e, 'headers') and e.headers and 'retry-after' in e.headers:
                        try:
                            retry_after = int(e.headers['retry-after'])
                        except (ValueError, TypeError):
                            pass
                    
                    instance.mark_rate_limited(retry_after)
                    logger.warning(f"Instance {instance.name} rate limited: {detail}")
                else:
                    instance.mark_error(str(e))
                    logger.error(f"Error from instance {instance.name}: {detail}")
                
                errors.append(f"{instance.name}: {detail}")
            except Exception as e:
                instance.mark_error(str(e))
                logger.error(f"Unexpected error from instance {instance.name}: {str(e)}")
                errors.append(f"{instance.name}: {str(e)}")
        
        # If we get here, all instances failed
        error_message = f"All Azure OpenAI instances failed: {'; '.join(errors)}"
        logger.error(error_message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_message,
        )
    
    async def _forward_request(self, 
                             instance: AzureOpenAIInstance, 
                             endpoint: str, 
                             deployment: str, 
                             payload: Dict[str, Any],
                             method: str = "POST") -> Dict[str, Any]:
        """
        Forward a request to an Azure OpenAI instance.
        
        Args:
            instance: The instance to use
            endpoint: The API endpoint
            deployment: The Azure deployment name
            payload: The request payload
            method: HTTP method
            
        Returns:
            The API response
            
        Raises:
            HTTPException: If the request fails
        """
        url = instance.build_azure_url(endpoint, deployment)
        instance.last_used = time.time()
        logger.debug(f"Forwarding request to instance {instance.name}: {url}")
        
        try:
            if method.upper() == "POST":
                response = await instance.client.post(url, json=payload)
            elif method.upper() == "GET":
                response = await instance.client.get(url, params=payload)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Raise exception for non-2xx responses
            response.raise_for_status()
            
            # Parse response
            if response.headers.get("content-type", "").startswith("application/json"):
                result = response.json()
            else:
                result = {"text": response.text}
            
            return result
        
        except httpx.HTTPStatusError as e:
            # Handle API errors
            error_detail = "Unknown error"
            headers = {}
            
            try:
                error_response = e.response.json()
                error_detail = error_response.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            
            status_code = e.response.status_code
            logger.error(f"Instance {instance.name} API error: {status_code} - {error_detail}")
            
            # Pass along retry-after header for rate limiting
            if status_code == 429 and "retry-after" in e.response.headers:
                headers["retry-after"] = e.response.headers["retry-after"]
            
            # Map Azure errors to appropriate status codes
            raise HTTPException(
                status_code=status_code,
                detail=f"Azure OpenAI API error from instance {instance.name}: {error_detail}",
                headers=headers,
            )
        
        except (httpx.RequestError, httpx.TimeoutException) as e:
            # Handle connection errors
            logger.error(f"Connection error to instance {instance.name}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Connection error to Azure OpenAI instance {instance.name}: {str(e)}",
            )
    
    def get_instance_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all instances."""
        return [
            {
                "name": instance.name,
                "status": instance.status,
                "current_tpm": instance.current_tpm,
                "max_tpm": instance.max_tpm,
                "tpm_usage_percent": round((instance.current_tpm / instance.max_tpm) * 100, 2) if instance.max_tpm > 0 else 0,
                "error_count": instance.error_count,
                "last_error": instance.last_error,
                "rate_limited_until": instance.rate_limited_until,
                "priority": instance.priority,
                "weight": instance.weight,
                "last_used": instance.last_used,
            }
            for instance in self.instances.values()
        ]


# Create a singleton instance with the default routing strategy
instance_manager = InstanceManager(
    routing_strategy=RoutingStrategy(os.getenv("AZURE_ROUTING_STRATEGY", RoutingStrategy.FAILOVER))
)
