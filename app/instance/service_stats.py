import time
import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class ServiceStats:
    """全局服务统计，集中管理服务级指标，特别是客户端错误"""
    
    def __init__(self, default_window_minutes: int = 5):
        # 配置
        self.default_window_minutes = default_window_minutes
        
        # 请求统计
        self.total_requests = 0
        self.request_window: Dict[int, int] = {}  # 时间戳 -> 请求数
        
        # 令牌统计
        self.total_tokens_served = 0
        self.tokens_window: Dict[int, int] = {}  # 时间戳 -> 令牌数
        
        # 客户端错误统计（真正返回给客户端的错误）
        self.total_client_errors_500 = 0
        self.total_client_errors_503 = 0
        self.total_client_errors_other = 0
        self.client_error_500_window: Dict[int, int] = {}  # 时间戳 -> 错误数
        self.client_error_503_window: Dict[int, int] = {}  # 时间戳 -> 错误数
        self.client_error_other_window: Dict[int, int] = {}  # 时间戳 -> 错误数
        
        # 成功请求统计
        self.total_successful_requests = 0
        self.successful_requests_window: Dict[int, int] = {}  # 时间戳 -> 成功请求数
        
        # 上游错误统计（从上游 API 收到的错误）
        self.total_upstream_errors = 0
        self.upstream_error_window: Dict[int, int] = {}  # 时间戳 -> 错误数
    
    def record_request(self, timestamp: Optional[int] = None):
        """记录一个请求"""
        if timestamp is None:
            timestamp = int(time.time())
        
        self.total_requests += 1
        self.request_window[timestamp] = self.request_window.get(timestamp, 0) + 1
    
    def record_successful_request(self, timestamp: Optional[int] = None, tokens_used: int = 0):
        """记录一个成功的请求"""
        if timestamp is None:
            timestamp = int(time.time())
        
        self.total_successful_requests += 1
        self.successful_requests_window[timestamp] = self.successful_requests_window.get(timestamp, 0) + 1
        
        if tokens_used > 0:
            self.total_tokens_served += tokens_used
            self.tokens_window[timestamp] = self.tokens_window.get(timestamp, 0) + tokens_used
    
    def record_client_error(self, status_code: int, timestamp: Optional[int] = None):
        """记录一个返回给客户端的错误"""
        if timestamp is None:
            timestamp = int(time.time())
        
        if status_code == 500:
            self.total_client_errors_500 += 1
            self.client_error_500_window[timestamp] = self.client_error_500_window.get(timestamp, 0) + 1
        elif status_code == 503:
            self.total_client_errors_503 += 1
            self.client_error_503_window[timestamp] = self.client_error_503_window.get(timestamp, 0) + 1
        else:
            self.total_client_errors_other += 1
            self.client_error_other_window[timestamp] = self.client_error_other_window.get(timestamp, 0) + 1
            
        logger.debug(f"Recorded client error {status_code} at timestamp {timestamp}")
    
    def record_upstream_error(self, timestamp: Optional[int] = None):
        """记录一个从上游 API 收到的错误"""
        if timestamp is None:
            timestamp = int(time.time())
        
        self.total_upstream_errors += 1
        self.upstream_error_window[timestamp] = self.upstream_error_window.get(timestamp, 0) + 1
    
    def get_metrics(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """获取指定时间窗口的服务指标"""
        if window_minutes is None:
            window_minutes = self.default_window_minutes
            
        if window_minutes <= 0:
            window_minutes = 5  # 默认为 5 分钟
            
        current_time = int(time.time())
        window_seconds = window_minutes * 60
        window_start = current_time - window_seconds
        
        # 计算时间窗口内的请求数
        requests_in_window = sum(count for ts, count in self.request_window.items() if ts >= window_start)
        
        # 计算时间窗口内的令牌数
        tokens_in_window = sum(count for ts, count in self.tokens_window.items() if ts >= window_start)
        
        # 计算时间窗口内的客户端错误数
        client_errors_500 = sum(count for ts, count in self.client_error_500_window.items() if ts >= window_start)
        client_errors_503 = sum(count for ts, count in self.client_error_503_window.items() if ts >= window_start)
        client_errors_other = sum(count for ts, count in self.client_error_other_window.items() if ts >= window_start)
        client_errors_total = client_errors_500 + client_errors_503 + client_errors_other
        
        # 计算时间窗口内的成功请求数
        successful_requests = sum(count for ts, count in self.successful_requests_window.items() if ts >= window_start)
        
        # 计算时间窗口内的上游错误数
        upstream_errors = sum(count for ts, count in self.upstream_error_window.items() if ts >= window_start)
        
        # 计算速率
        requests_per_minute = round(requests_in_window / window_minutes, 2) if window_minutes > 0 else 0
        tokens_per_minute = round(tokens_in_window / window_minutes, 2) if window_minutes > 0 else 0
        
        # 计算错误率和成功率
        error_rate = round(client_errors_total / requests_in_window, 4) if requests_in_window > 0 else 0.0
        success_rate = round(successful_requests / requests_in_window, 4) if requests_in_window > 0 else 1.0
        upstream_error_rate = round(upstream_errors / requests_in_window, 4) if requests_in_window > 0 else 0.0
        
        # 清理旧数据
        self._cleanup_old_data(window_start)
        
        return {
            "window_minutes": window_minutes,
            "requests_per_minute": requests_per_minute,
            "tokens_per_minute": tokens_per_minute,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "upstream_error_rate": upstream_error_rate,
            "total_requests_in_window": requests_in_window,
            "successful_requests_in_window": successful_requests,
            "failed_requests_in_window": client_errors_total,
            "total_tokens_in_window": tokens_in_window,
            "client_errors_500": client_errors_500,
            "client_errors_503": client_errors_503,
            "client_errors_other": client_errors_other,
            "upstream_errors_in_window": upstream_errors,
            
            # 总计统计
            "total_requests": self.total_requests,
            "total_successful_requests": self.total_successful_requests,
            "total_client_errors_500": self.total_client_errors_500,
            "total_client_errors_503": self.total_client_errors_503,
            "total_client_errors_other": self.total_client_errors_other,
            "total_client_errors": self.total_client_errors_500 + self.total_client_errors_503 + self.total_client_errors_other,
            "total_upstream_errors": self.total_upstream_errors,
            "total_tokens_served": self.total_tokens_served
        }
    
    def get_multiple_window_metrics(self, windows: List[int] = [5, 15, 30, 60]) -> Dict[str, Dict[str, Any]]:
        """获取多个时间窗口的指标"""
        return {f"{window}min": self.get_metrics(window) for window in windows}
    
    def _cleanup_old_data(self, cutoff_timestamp: int):
        """清理旧数据以减少内存使用"""
        # 只保留最近的数据
        self.request_window = {ts: count for ts, count in self.request_window.items() if ts >= cutoff_timestamp}
        self.tokens_window = {ts: count for ts, count in self.tokens_window.items() if ts >= cutoff_timestamp}
        self.client_error_500_window = {ts: count for ts, count in self.client_error_500_window.items() if ts >= cutoff_timestamp}
        self.client_error_503_window = {ts: count for ts, count in self.client_error_503_window.items() if ts >= cutoff_timestamp}
        self.client_error_other_window = {ts: count for ts, count in self.client_error_other_window.items() if ts >= cutoff_timestamp}
        self.successful_requests_window = {ts: count for ts, count in self.successful_requests_window.items() if ts >= cutoff_timestamp}
        self.upstream_error_window = {ts: count for ts, count in self.upstream_error_window.items() if ts >= cutoff_timestamp}


# 创建全局单例
service_stats = ServiceStats() 