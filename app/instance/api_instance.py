from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import time
import httpx
import logging

from app.utils.endpoint_mappings import EndpointMapper

logger = logging.getLogger(__name__)

class InstanceStatus(str, Enum):
    """Status of an API instance."""
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited" 
    ERROR = "error"

class APIInstance(BaseModel):
    """Configuration and state for an OpenAI-compatible service instance."""
    name: str = Field(..., description="Unique identifier for this instance")
    provider_type: str = Field(default="azure", description="Provider type (azure or generic)")
    api_key: str = Field(..., description="API key for the service")
    api_base: str = Field(..., description="API base URL")
    api_version: str = Field(..., description="API version")
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
        proxies = {"http://": self.proxy_url} if self.proxy_url else None

        if self.client is None:
            self.client = httpx.AsyncClient(
                proxies=proxies,
                timeout=httpx.Timeout(300.0),
                headers={"api-key": self.api_key} if self.api_key != 'NONE' else {},
            )
    
    def update_tpm_usage(self, tokens: int) -> None:
        current_time = int(time.time())
        window_start = current_time - 60
        self.usage_window = {ts: usage for ts, usage in self.usage_window.items() if ts >= window_start}
        self.usage_window[current_time] = self.usage_window.get(current_time, 0) + tokens
        self.current_tpm = sum(self.usage_window.values())
    
    def is_rate_limited(self) -> bool:
        if self.status == InstanceStatus.RATE_LIMITED:
            current_time = time.time()
            if self.rate_limited_until and current_time >= self.rate_limited_until:
                logger.info(f"Instance {self.name} rate limit has expired, marking as healthy")
                self.status = InstanceStatus.HEALTHY
                self.rate_limited_until = None
                return False
            return True
        if self.current_tpm >= self.max_tpm * 0.95:
            logger.warning(f"Instance {self.name} is approaching TPM limit: {self.current_tpm}/{self.max_tpm}")
            return True
        return False
    
    def mark_rate_limited(self, retry_after: Optional[int] = None) -> None:
        self.status = InstanceStatus.RATE_LIMITED
        if retry_after is None:
            retry_after = 60
        self.rate_limited_until = time.time() + retry_after
        logger.warning(f"Instance {self.name} marked as rate limited for {retry_after} seconds")
    
    def mark_error(self, error_message: str) -> None:
        self.error_count += 1
        self.last_error = error_message
        if self.error_count >= 3:
            self.status = InstanceStatus.ERROR
            logger.error(f"Instance {self.name} marked as error after {self.error_count} consecutive errors")
    
    def mark_healthy(self) -> None:
        self.status = InstanceStatus.HEALTHY
        self.error_count = 0
        self.last_error = None
        self.rate_limited_until = None
    
    def build_url(self, endpoint: str, deployment_name: str) -> str:
        logger.debug(f"Building URL for instance {self.name}, provider_type={self.provider_type}, endpoint={endpoint}, deployment={deployment_name}")
        if self.provider_type == "generic":
            url = f"{self.api_base}{endpoint}"
            logger.debug(f"Generic provider URL: {url}")
            return url
        if deployment_name == "DeepSeek-R1":
            if endpoint == "/v1/chat/completions":
                return f"{self.api_base}/deepseek/chat/completions?api-version={self.api_version}"
            elif endpoint == "/v1/completions":
                return f"{self.api_base}/deepseek/completions?api-version={self.api_version}"
            elif endpoint == "/v1/embeddings":
                return f"{self.api_base}/deepseek/embeddings?api-version={self.api_version}"
            else:
                endpoint_part = endpoint.split("/")[-1]
                return f"{self.api_base}/deepseek/{endpoint_part}?api-version={self.api_version}"
        mapper = EndpointMapper()
        endpoint_type = None
        if "/chat/completions" in endpoint:
            endpoint_type = "chat"
        elif "/completions" in endpoint:
            endpoint_type = "completions"
        elif "/embeddings" in endpoint:
            endpoint_type = "embeddings"
        else:
            parts = endpoint.split("/")
            if len(parts) >= 3 and parts[-2] == "deployments":
                return f"{self.api_base}{endpoint}"
            else:
                logger.error(f"Unsupported endpoint: {endpoint}")
                raise ValueError(f"Unsupported endpoint: {endpoint}")
        try:
            provider_endpoint = mapper.get_endpoint(
                provider=self.provider_type,
                endpoint_type=endpoint_type,
                deployment=deployment_name
            )
            return f"{self.api_base}{provider_endpoint}?api-version={self.api_version}"
        except ValueError as e:
            logger.error(f"Error building URL: {str(e)}")
            raise 