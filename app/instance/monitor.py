import logging
import time
from typing import Dict, List, Any, Optional

from .api_instance import APIInstance

logger = logging.getLogger(__name__)

class InstanceMonitor:
    """Monitors and provides statistics for API instances."""
    
    @staticmethod
    def get_instance_stats(instances: Dict[str, APIInstance]) -> List[Dict[str, Any]]:
        """
        Get statistics for all instances.
        
        Args:
            instances: Dictionary of API instances
            
        Returns:
            List of instance statistics dictionaries
        """
        return [
            {
                "name": instance.name,
                "provider_type": instance.provider_type,
                "status": instance.status,
                # Request statistics
                "current_tpm": instance.instance_stats.current_tpm,
                "current_rpm": instance.instance_stats.current_rpm,
                "rpm_window_minutes": instance.instance_stats.rpm_window_minutes,
                "max_tpm": instance.max_tpm,
                "tpm_usage_percent": round((instance.instance_stats.current_tpm / instance.max_tpm) * 100, 2) if instance.max_tpm > 0 else 0,
                "total_tokens_served": instance.instance_stats.total_tokens_served,
                
                # Instance-level Error statistics (errors encountered by this instance)
                "instance_error_rate": instance.instance_stats.current_error_rate,
                "instance_500_rate": instance.instance_stats.current_500_rate,
                "instance_503_rate": instance.instance_stats.current_503_rate,
                "total_instance_errors_500": instance.instance_stats.total_errors_500,
                "total_instance_errors_503": instance.instance_stats.total_errors_503,
                "total_instance_errors_other": instance.instance_stats.total_other_errors,
                
                # Client-level Error statistics (errors returned to clients)
                "client_error_rate": instance.instance_stats.current_client_error_rate,
                "client_500_rate": instance.instance_stats.current_client_500_rate,
                "client_503_rate": instance.instance_stats.current_client_503_rate,
                "total_client_errors_500": instance.instance_stats.total_client_errors_500,
                "total_client_errors_503": instance.instance_stats.total_client_errors_503,
                "total_client_errors_other": instance.instance_stats.total_client_errors_other,
                
                # Upstream Error statistics (errors from endpoint APIs)
                "upstream_error_rate": instance.instance_stats.current_upstream_error_rate,
                "upstream_429_rate": instance.instance_stats.current_upstream_429_rate,
                "upstream_400_rate": instance.instance_stats.current_upstream_400_rate,
                "total_upstream_429_errors": instance.instance_stats.total_upstream_429_errors,
                "total_upstream_400_errors": instance.instance_stats.total_upstream_400_errors,
                "total_upstream_500_errors": instance.instance_stats.total_upstream_500_errors,
                "total_upstream_other_errors": instance.instance_stats.total_upstream_other_errors,
                
                # Instance status
                "error_count": instance.error_count,
                "last_error": instance.last_error,
                "rate_limited_until": instance.rate_limited_until,
                
                # Configuration
                "priority": instance.priority,
                "weight": instance.weight,
                "last_used": instance.last_used,
                "supported_models": instance.supported_models,
                "model_deployments": instance.model_deployments,
            }
            for instance in instances.values()
        ]
    
    @staticmethod
    def get_health_status(instances: Dict[str, APIInstance]) -> Dict[str, Any]:
        """
        Get health status summary for all instances.
        
        Args:
            instances: Dictionary of API instances
            
        Returns:
            Summary of instance health status
        """
        total = len(instances)
        healthy = sum(1 for i in instances.values() if i.status == "healthy")
        rate_limited = sum(1 for i in instances.values() if i.status == "rate_limited")
        error = sum(1 for i in instances.values() if i.status == "error")
        
        # Find the average RPM window if there are instances
        avg_rpm_window = 0
        if total > 0:
            avg_rpm_window = sum(i.instance_stats.rpm_window_minutes for i in instances.values()) / total
        
        # Calculate total instance-level errors
        total_instance_errors_500 = sum(i.instance_stats.total_errors_500 for i in instances.values())
        total_instance_errors_503 = sum(i.instance_stats.total_errors_503 for i in instances.values())
        total_instance_errors_other = sum(i.instance_stats.total_other_errors for i in instances.values())
        total_instance_errors = total_instance_errors_500 + total_instance_errors_503 + total_instance_errors_other
        
        # Calculate total client-level errors
        total_client_errors_500 = sum(i.instance_stats.total_client_errors_500 for i in instances.values())
        total_client_errors_503 = sum(i.instance_stats.total_client_errors_503 for i in instances.values())
        total_client_errors_other = sum(i.instance_stats.total_client_errors_other for i in instances.values())
        total_client_errors = total_client_errors_500 + total_client_errors_503 + total_client_errors_other
        
        # Calculate total upstream errors
        total_upstream_429 = sum(i.instance_stats.total_upstream_429_errors for i in instances.values())
        total_upstream_400 = sum(i.instance_stats.total_upstream_400_errors for i in instances.values())
        total_upstream_500 = sum(i.instance_stats.total_upstream_500_errors for i in instances.values())
        total_upstream_other = sum(i.instance_stats.total_upstream_other_errors for i in instances.values())
        total_upstream_errors = total_upstream_429 + total_upstream_400 + total_upstream_500 + total_upstream_other
        
        # Calculate average error rates
        avg_instance_error_rate = 0.0
        avg_instance_500_rate = 0.0
        avg_instance_503_rate = 0.0
        avg_client_error_rate = 0.0
        avg_client_500_rate = 0.0
        avg_client_503_rate = 0.0
        avg_upstream_error_rate = 0.0
        avg_upstream_429_rate = 0.0
        avg_upstream_400_rate = 0.0
        
        if total > 0:
            avg_instance_error_rate = sum(i.instance_stats.current_error_rate for i in instances.values()) / total
            avg_instance_500_rate = sum(i.instance_stats.current_500_rate for i in instances.values()) / total
            avg_instance_503_rate = sum(i.instance_stats.current_503_rate for i in instances.values()) / total
            avg_client_error_rate = sum(i.instance_stats.current_client_error_rate for i in instances.values()) / total
            avg_client_500_rate = sum(i.instance_stats.current_client_500_rate for i in instances.values()) / total
            avg_client_503_rate = sum(i.instance_stats.current_client_503_rate for i in instances.values()) / total
            avg_upstream_error_rate = sum(i.instance_stats.current_upstream_error_rate for i in instances.values()) / total
            avg_upstream_429_rate = sum(i.instance_stats.current_upstream_429_rate for i in instances.values()) / total
            avg_upstream_400_rate = sum(i.instance_stats.current_upstream_400_rate for i in instances.values()) / total
        
        return {
            "total_instances": total,
            "healthy_instances": healthy,
            "rate_limited_instances": rate_limited,
            "error_instances": error,
            "health_percentage": round((healthy / total) * 100, 2) if total > 0 else 0,
            "has_available_capacity": any(
                i.status == "healthy" and i.instance_stats.current_tpm < i.max_tpm * 0.9
                for i in instances.values()
            ),
            "total_tokens_served": sum(i.instance_stats.total_tokens_served for i in instances.values()),
            "total_current_rpm": sum(i.instance_stats.current_rpm for i in instances.values()),
            "avg_rpm_window_minutes": round(avg_rpm_window, 1),
            
            # Instance-level Error statistics (errors encountered by instances)
            "total_instance_errors": total_instance_errors,
            "total_instance_errors_500": total_instance_errors_500,
            "total_instance_errors_503": total_instance_errors_503,
            "total_instance_errors_other": total_instance_errors_other,
            "avg_instance_error_rate": round(avg_instance_error_rate, 4),
            "avg_instance_500_rate": round(avg_instance_500_rate, 4),
            "avg_instance_503_rate": round(avg_instance_503_rate, 4),
            
            # Client-level Error statistics (errors returned to clients)
            "total_client_errors": total_client_errors,
            "total_client_errors_500": total_client_errors_500,
            "total_client_errors_503": total_client_errors_503,
            "total_client_errors_other": total_client_errors_other,
            "avg_client_error_rate": round(avg_client_error_rate, 4),
            "avg_client_500_rate": round(avg_client_500_rate, 4),
            "avg_client_503_rate": round(avg_client_503_rate, 4),
            
            # Upstream Error statistics (errors from endpoint APIs)
            "total_upstream_errors": total_upstream_errors,
            "total_upstream_429_errors": total_upstream_429,
            "total_upstream_400_errors": total_upstream_400,
            "total_upstream_500_errors": total_upstream_500,
            "total_upstream_other_errors": total_upstream_other,
            "avg_upstream_error_rate": round(avg_upstream_error_rate, 4),
            "avg_upstream_429_rate": round(avg_upstream_429_rate, 4),
            "avg_upstream_400_rate": round(avg_upstream_400_rate, 4),
        }
        
    @staticmethod
    def get_service_metrics(instances: Dict[str, APIInstance], window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Get overall service performance metrics for a specific time window.
        
        Args:
            instances: Dictionary of API instances
            window_minutes: Time window in minutes for calculations, default is the average 
                            of all instances' rpm_window_minutes
            
        Returns:
            Overall service metrics for the specified time window
        """
        if not instances:
            return {
                "window_minutes": 0,
                "tokens_per_minute": 0,
                "requests_per_minute": 0,
                "error_rate": 0.0,
                "success_rate": 1.0,  # 没有实例认为是100%成功率
                "upstream_error_rate": 0.0,
                "active_instances": 0
            }
            
        # Determine the window size to use
        if window_minutes is None:
            # Use the average of all instances' rpm_window_minutes
            window_minutes = int(sum(i.instance_stats.rpm_window_minutes for i in instances.values()) / len(instances))
            
        if window_minutes <= 0:
            window_minutes = 5  # Default to 5 minutes if invalid
            
        current_time = int(time.time())
        window_seconds = window_minutes * 60
        window_start = current_time - window_seconds
        
        # 处理的唯一请求的时间戳集合
        # 由于我们无法精确地知道哪些请求是唯一的（没有请求ID），我们使用时间戳来近似
        unique_request_timestamps = set()
        
        # Collect all request timestamps across all instances
        for instance in instances.values():
            for ts in instance.instance_stats.request_window.keys():
                if ts >= window_start:
                    unique_request_timestamps.add(ts)
        
        # 估计的总唯一请求数
        total_unique_requests = len(unique_request_timestamps)
        
        # 累积所有时间窗口内处理的总令牌数
        total_tokens = 0
        for instance in instances.values():
            for ts, count in instance.instance_stats.usage_window.items():
                if ts >= window_start:
                    total_tokens += count
        
        # 收集客户端错误时间戳
        # 这些是真正返回给客户端的错误，被记录在一个特殊的字段中
        all_client_error_timestamps = set()
        for instance in instances.values():
            for ts, count in instance.instance_stats.client_error_500_window.items():
                if ts >= window_start:
                    for _ in range(count):
                        all_client_error_timestamps.add(ts)
            for ts, count in instance.instance_stats.client_error_503_window.items():
                if ts >= window_start:
                    for _ in range(count):
                        all_client_error_timestamps.add(ts)
            for ts, count in instance.instance_stats.client_error_other_window.items():
                if ts >= window_start:
                    for _ in range(count):
                        all_client_error_timestamps.add(ts)
        
        # 确保我们只统计每个时间戳一次，避免重复计数
        client_errors = len(all_client_error_timestamps)
            
        # 收集上游错误（从上游 API 收到的错误）时间戳
        # 我们关心的是遇到上游错误的请求数，而不是错误总数
        # 因为一个请求可能经过多个实例，多个实例可能都遇到上游错误
        upstream_error_timestamps = set()
        for instance in instances.values():
            for ts, count in instance.instance_stats.upstream_429_window.items():
                if ts >= window_start:
                    for _ in range(count):
                        upstream_error_timestamps.add(ts)
            for ts, count in instance.instance_stats.upstream_400_window.items():
                if ts >= window_start:
                    for _ in range(count):
                        upstream_error_timestamps.add(ts)
            for ts, count in instance.instance_stats.upstream_500_window.items():
                if ts >= window_start:
                    for _ in range(count):
                        upstream_error_timestamps.add(ts)
            for ts, count in instance.instance_stats.upstream_other_window.items():
                if ts >= window_start:
                    for _ in range(count):
                        upstream_error_timestamps.add(ts)
                        
        total_upstream_errors = len(upstream_error_timestamps)
        
        # 计算成功请求：总请求 - 客户端错误
        successful_requests = total_unique_requests - client_errors
        
        # 计算成功率
        success_rate = 1.0
        if total_unique_requests > 0:
            success_rate = round(successful_requests / total_unique_requests, 4)
            
        # 计算错误率
        error_rate = 0.0
        if total_unique_requests > 0:
            error_rate = round(client_errors / total_unique_requests, 4)
            
        # 计算上游错误率（遇到上游错误的请求比例）
        upstream_error_rate = 0.0
        if total_unique_requests > 0:
            upstream_error_rate = round(total_upstream_errors / total_unique_requests, 4)
        
        # Calculate rates based on the window
        requests_per_minute = int(total_unique_requests / window_minutes) if window_minutes > 0 else 0
        tokens_per_minute = int(total_tokens / window_minutes) if window_minutes > 0 else 0
        
        # Count active instances (those with requests in the window)
        active_instances = sum(1 for i in instances.values() 
                              if any(ts >= window_start for ts in i.instance_stats.request_window.keys()))
        
        return {
            "window_minutes": window_minutes,
            "tokens_per_minute": tokens_per_minute,
            "requests_per_minute": requests_per_minute,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "upstream_error_rate": upstream_error_rate,
            "active_instances": active_instances,
            "total_requests_in_window": total_unique_requests,
            "successful_requests_in_window": successful_requests,
            "failed_requests_in_window": client_errors,
            "total_tokens_in_window": total_tokens,
            "total_upstream_errors_in_window": total_upstream_errors
        }
    
    @staticmethod
    def get_multiple_window_metrics(instances: Dict[str, APIInstance], 
                                   windows: List[int] = [5, 15, 30, 60]) -> Dict[str, Dict[str, Any]]:
        """
        Get service metrics for multiple time windows.
        
        Args:
            instances: Dictionary of API instances
            windows: List of time windows in minutes to calculate metrics for
            
        Returns:
            Dictionary of service metrics for each time window
        """
        metrics = {}
        for window in windows:
            metrics[f"{window}min"] = InstanceMonitor.get_service_metrics(instances, window)
        return metrics 