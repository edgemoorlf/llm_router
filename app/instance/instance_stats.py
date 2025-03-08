from typing import Dict, Any, Optional
import time
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class InstanceStats(BaseModel):
    """Statistics tracker for an API instance."""
    
    # Configuration
    rpm_window_minutes: int = Field(default=5, description="Time window in minutes for statistics calculation")
    
    # Basic usage stats
    current_tpm: int = Field(default=0, description="Current TPM usage (resets each minute)")
    current_rpm: int = Field(default=0, description="Current requests per minute based on configured window")
    total_tokens_served: int = Field(default=0, description="Total tokens served by this instance")
    
    # Windows for sliding statistics
    usage_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of token usage (timestamp -> tokens)")
    request_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of request counts (timestamp -> requests)")
    
    # Instance-level Error statistics - errors encountered by this instance
    total_errors_500: int = Field(default=0, description="Total number of 500 errors encountered by this instance")
    total_errors_503: int = Field(default=0, description="Total number of 503 errors encountered by this instance")
    total_other_errors: int = Field(default=0, description="Total number of other errors encountered by this instance")
    error_500_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 500 errors (timestamp -> count)")
    error_503_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 503 errors (timestamp -> count)")
    error_other_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of other errors (timestamp -> count)")
    current_error_rate: float = Field(default=0.0, description="Current error rate for this instance (errors/requests) based on time window")
    current_500_rate: float = Field(default=0.0, description="Current 500 error rate for this instance based on time window")
    current_503_rate: float = Field(default=0.0, description="Current 503 error rate for this instance based on time window")
    
    # Client-level Error statistics - errors actually returned to clients
    total_client_errors_500: int = Field(default=0, description="Total number of 500 errors returned to clients")
    total_client_errors_503: int = Field(default=0, description="Total number of 503 errors returned to clients")
    total_client_errors_other: int = Field(default=0, description="Total number of other errors returned to clients")
    client_error_500_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of client 500 errors (timestamp -> count)")
    client_error_503_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of client 503 errors (timestamp -> count)")
    client_error_other_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of client other errors (timestamp -> count)")
    current_client_error_rate: float = Field(default=0.0, description="Current client error rate (errors/requests) based on time window")
    current_client_500_rate: float = Field(default=0.0, description="Current client 500 error rate based on time window")
    current_client_503_rate: float = Field(default=0.0, description="Current client 503 error rate based on time window")
    
    # Upstream Error statistics - errors received from endpoint APIs
    total_upstream_429_errors: int = Field(default=0, description="Total 429 rate limit errors from upstream APIs")
    total_upstream_400_errors: int = Field(default=0, description="Total 400 bad request errors from upstream APIs")
    total_upstream_500_errors: int = Field(default=0, description="Total 500 internal errors from upstream APIs")
    total_upstream_other_errors: int = Field(default=0, description="Total other errors from upstream APIs")
    upstream_429_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 429 errors (timestamp -> count)")
    upstream_400_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 400 errors (timestamp -> count)")
    upstream_500_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of 500 errors (timestamp -> count)")
    upstream_other_window: Dict[int, int] = Field(default_factory=dict, description="Sliding window of other errors (timestamp -> count)")
    current_upstream_error_rate: float = Field(default=0.0, description="Current upstream error rate based on time window")
    current_upstream_429_rate: float = Field(default=0.0, description="Current upstream 429 error rate based on time window")
    current_upstream_400_rate: float = Field(default=0.0, description="Current upstream 400 error rate based on time window")
    
    def update_tpm_usage(self, tokens: int) -> None:
        """Update the tokens per minute usage with sliding window."""
        current_time = int(time.time())
        window_start = current_time - 60
        self.usage_window = {ts: usage for ts, usage in self.usage_window.items() if ts >= window_start}
        self.usage_window[current_time] = self.usage_window.get(current_time, 0) + tokens
        self.current_tpm = sum(self.usage_window.values())
        
        # Update total tokens served
        self.total_tokens_served += tokens
    
    def update_rpm_usage(self, instance_name: str = "") -> None:
        """
        Update the requests per minute counter with sliding window.
        Uses the configured time window (rpm_window_minutes) for calculation.
        
        Args:
            instance_name: Optional name of the instance for logging purposes
        """
        current_time = int(time.time())
        # Convert window minutes to seconds
        window_seconds = self.rpm_window_minutes * 60
        window_start = current_time - window_seconds
        
        # Update sliding window
        self.request_window = {ts: count for ts, count in self.request_window.items() if ts >= window_start}
        self.request_window[current_time] = self.request_window.get(current_time, 0) + 1
        
        # Calculate RPM as total requests in window divided by window size in minutes
        total_requests_in_window = sum(self.request_window.values())
        if self.rpm_window_minutes > 0:
            self.current_rpm = int(total_requests_in_window / self.rpm_window_minutes)
        else:
            self.current_rpm = total_requests_in_window
        
        # Update error rates as well, since they share the same window
        self.update_error_rates(instance_name)
    
    def update_error_rates(self, instance_name: str = "") -> None:
        """
        Update error rates based on the time window.
        Should be called after updating the request window.
        
        Args:
            instance_name: Optional name of the instance for logging purposes
        """
        current_time = int(time.time())
        window_seconds = self.rpm_window_minutes * 60
        window_start = current_time - window_seconds
        
        # Calculate request count in the window (should already be up to date)
        total_requests_in_window = sum(self.request_window.values())
        
        # Clean up old entries in instance-level error windows
        self.error_500_window = {ts: count for ts, count in self.error_500_window.items() if ts >= window_start}
        self.error_503_window = {ts: count for ts, count in self.error_503_window.items() if ts >= window_start}
        self.error_other_window = {ts: count for ts, count in self.error_other_window.items() if ts >= window_start}
        
        # Calculate instance-level error count in the window
        errors_500_in_window = sum(self.error_500_window.values())
        errors_503_in_window = sum(self.error_503_window.values())
        errors_other_in_window = sum(self.error_other_window.values())
        total_errors_in_window = errors_500_in_window + errors_503_in_window + errors_other_in_window
        
        # Update instance-level error rates
        if total_requests_in_window > 0:
            self.current_error_rate = round(total_errors_in_window / total_requests_in_window, 4)
            self.current_500_rate = round(errors_500_in_window / total_requests_in_window, 4)
            self.current_503_rate = round(errors_503_in_window / total_requests_in_window, 4)
        else:
            self.current_error_rate = 0.0
            self.current_500_rate = 0.0
            self.current_503_rate = 0.0
            
        # Clean up old entries in client-level error windows
        self.client_error_500_window = {ts: count for ts, count in self.client_error_500_window.items() if ts >= window_start}
        self.client_error_503_window = {ts: count for ts, count in self.client_error_503_window.items() if ts >= window_start}
        self.client_error_other_window = {ts: count for ts, count in self.client_error_other_window.items() if ts >= window_start}
        
        # Calculate client-level error count in the window
        client_errors_500_in_window = sum(self.client_error_500_window.values())
        client_errors_503_in_window = sum(self.client_error_503_window.values())
        client_errors_other_in_window = sum(self.client_error_other_window.values())
        total_client_errors_in_window = client_errors_500_in_window + client_errors_503_in_window + client_errors_other_in_window
        
        # Update client-level error rates
        if total_requests_in_window > 0:
            self.current_client_error_rate = round(total_client_errors_in_window / total_requests_in_window, 4)
            self.current_client_500_rate = round(client_errors_500_in_window / total_requests_in_window, 4)
            self.current_client_503_rate = round(client_errors_503_in_window / total_requests_in_window, 4)
        else:
            self.current_client_error_rate = 0.0
            self.current_client_500_rate = 0.0
            self.current_client_503_rate = 0.0
            
        # Clean up old entries in upstream error windows
        self.upstream_429_window = {ts: count for ts, count in self.upstream_429_window.items() if ts >= window_start}
        self.upstream_400_window = {ts: count for ts, count in self.upstream_400_window.items() if ts >= window_start}
        self.upstream_500_window = {ts: count for ts, count in self.upstream_500_window.items() if ts >= window_start}
        self.upstream_other_window = {ts: count for ts, count in self.upstream_other_window.items() if ts >= window_start}
        
        # Calculate upstream error count in the window
        upstream_429_in_window = sum(self.upstream_429_window.values())
        upstream_400_in_window = sum(self.upstream_400_window.values())
        upstream_500_in_window = sum(self.upstream_500_window.values())
        upstream_other_in_window = sum(self.upstream_other_window.values())
        total_upstream_errors_in_window = upstream_429_in_window + upstream_400_in_window + upstream_500_in_window + upstream_other_in_window
        
        # Update upstream error rates
        if total_requests_in_window > 0:
            self.current_upstream_error_rate = round(total_upstream_errors_in_window / total_requests_in_window, 4)
            self.current_upstream_429_rate = round(upstream_429_in_window / total_requests_in_window, 4)
            self.current_upstream_400_rate = round(upstream_400_in_window / total_requests_in_window, 4)
        else:
            self.current_upstream_error_rate = 0.0
            self.current_upstream_429_rate = 0.0
            self.current_upstream_400_rate = 0.0
    
    def record_error(self, status_code: int, instance_name: str = "") -> None:
        """
        Record an instance-level error encountered by this instance.
        These errors may not be sent to clients if the request is retried on another instance.
        
        Args:
            status_code: HTTP status code of the error
            instance_name: Optional name of the instance for logging purposes
        """
        current_time = int(time.time())
        
        if status_code == 500:
            self.total_errors_500 += 1
            self.error_500_window[current_time] = self.error_500_window.get(current_time, 0) + 1
        elif status_code == 503:
            self.total_errors_503 += 1
            self.error_503_window[current_time] = self.error_503_window.get(current_time, 0) + 1
        else:
            self.total_other_errors += 1
            self.error_other_window[current_time] = self.error_other_window.get(current_time, 0) + 1
            
        # Update error rates
        self.update_error_rates(instance_name)
        
        if instance_name:
            logger.debug(f"Recorded instance-level error {status_code} for instance {instance_name}, current error rate: {self.current_error_rate:.2%}")
    
    def record_client_error(self, status_code: int, instance_name: str = "", timestamp: Optional[int] = None) -> None:
        """
        Record a client-level error that was actually returned to the client.
        This happens when all instances have failed, and the error is returned to the client.
        
        Args:
            status_code: HTTP status code of the error
            instance_name: Optional name of the instance for logging purposes
            timestamp: Optional timestamp for when the error occurred
        """
        if timestamp is None:
            timestamp = int(time.time())
            
        error_type = "other"
        if status_code == 500:
            self.total_client_errors_500 += 1
            self.client_error_500_window[timestamp] = self.client_error_500_window.get(timestamp, 0) + 1
            error_type = "500"
        elif status_code == 503:
            self.total_client_errors_503 += 1
            self.client_error_503_window[timestamp] = self.client_error_503_window.get(timestamp, 0) + 1
            error_type = "503"
        else:
            self.total_client_errors_other += 1
            self.client_error_other_window[timestamp] = self.client_error_other_window.get(timestamp, 0) + 1
            
        # Update error rates
        self.update_error_rates(instance_name)
        
        if instance_name:
            logger.debug(f"Recorded client-level error {error_type} for instance {instance_name}, current client error rate: {self.current_client_error_rate:.2%}")
    
    def record_upstream_error(self, status_code: int, instance_name: str = "") -> None:
        """
        Record an error received from an upstream API endpoint.
        
        Args:
            status_code: HTTP status code of the error
            instance_name: Optional name of the instance for logging purposes
        """
        current_time = int(time.time())
        
        if status_code == 429:
            self.total_upstream_429_errors += 1
            self.upstream_429_window[current_time] = self.upstream_429_window.get(current_time, 0) + 1
        elif status_code == 400:
            self.total_upstream_400_errors += 1
            self.upstream_400_window[current_time] = self.upstream_400_window.get(current_time, 0) + 1
        elif status_code == 500:
            self.total_upstream_500_errors += 1
            self.upstream_500_window[current_time] = self.upstream_500_window.get(current_time, 0) + 1
        else:
            self.total_upstream_other_errors += 1
            self.upstream_other_window[current_time] = self.upstream_other_window.get(current_time, 0) + 1
            
        # Update error rates
        self.update_error_rates(instance_name)
        
        if instance_name:
            logger.debug(f"Recorded upstream error {status_code} for instance {instance_name}, current upstream error rate: {self.current_upstream_error_rate:.2%}")
    
    def set_rpm_window(self, minutes: int, instance_name: str = "") -> None:
        """
        Set the time window for RPM calculation.
        
        Args:
            minutes: New time window in minutes
            instance_name: Optional name of the instance for logging purposes
        """
        if minutes <= 0:
            raise ValueError("RPM window must be positive")
        
        self.rpm_window_minutes = minutes
        # Recalculate current RPM with the new window
        self.update_rpm_usage(instance_name)
        
        if instance_name:
            logger.info(f"Set RPM window for instance {instance_name} to {minutes} minutes")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "current_tpm": self.current_tpm,
            "current_rpm": self.current_rpm,
            "total_tokens_served": self.total_tokens_served,
            "error_rates": {
                "instance": {
                    "total": self.current_error_rate,
                    "500": self.current_500_rate,
                    "503": self.current_503_rate
                },
                "client": {
                    "total": self.current_client_error_rate,
                    "500": self.current_client_500_rate,
                    "503": self.current_client_503_rate
                },
                "upstream": {
                    "total": self.current_upstream_error_rate,
                    "429": self.current_upstream_429_rate,
                    "400": self.current_upstream_400_rate
                }
            },
            "error_counts": {
                "instance": {
                    "500": self.total_errors_500,
                    "503": self.total_errors_503,
                    "other": self.total_other_errors
                },
                "client": {
                    "500": self.total_client_errors_500,
                    "503": self.total_client_errors_503,
                    "other": self.total_client_errors_other
                },
                "upstream": {
                    "429": self.total_upstream_429_errors,
                    "400": self.total_upstream_400_errors,
                    "500": self.total_upstream_500_errors,
                    "other": self.total_upstream_other_errors
                }
            }
        } 