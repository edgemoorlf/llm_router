import time
import random
import logging
from typing import Dict, Any, Tuple, Optional
import httpx
from fastapi import HTTPException, status

from .api_instance import APIInstance
from .service_stats import service_stats

logger = logging.getLogger(__name__)

class RequestForwarder:
    """Forwards API requests to selected instances and handles responses."""

    @staticmethod
    async def forward_request(
            instance: APIInstance, 
            endpoint: str, 
            deployment: str, 
            payload: Dict[str, Any],
            method: str = "POST") -> Dict[str, Any]:
        """
        Forward a request to an API instance.
        
        Args:
            instance: The instance to use
            endpoint: The API endpoint
            deployment: The deployment name
            payload: The request payload
            method: HTTP method
            
        Returns:
            The API response
            
        Raises:
            HTTPException: If the request fails
        """
        url = instance.build_url(endpoint, deployment)
        instance.last_used = time.time()
        current_time = int(time.time())
        
        # Update request per minute counter for instance
        instance.update_rpm_usage()
        
        # Record request in global service stats
        service_stats.record_request(current_time)
        
        # Log request details at debug level
        request_id = f"req-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
        logger.debug(f"[{request_id}] Forwarding request to instance {instance.name} ({instance.provider_type}): {url}")
        
        # Log payload at trace level (if needed for debugging)
        if logger.isEnabledFor(logging.DEBUG):
            # Create a sanitized copy of the payload for logging
            sanitized_payload = payload.copy()
            # Remove sensitive fields if present
            if "api_key" in sanitized_payload:
                sanitized_payload["api_key"] = "***REDACTED***"
            logger.debug(f"[{request_id}] Request payload: {sanitized_payload}")
        
        start_time = time.time()
        try:
            if method.upper() == "POST":
                response = await instance.client.post(url, json=payload)
            elif method.upper() == "GET":
                response = await instance.client.get(url, params=payload)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Log response status and timing
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"[{request_id}] Response received: status={response.status_code}, time={elapsed_ms}ms")
            
            # Raise exception for non-2xx responses
            response.raise_for_status()
            
            # Parse response
            if response.headers.get("content-type", "").startswith("application/json"):
                result = response.json()
                
                # Log token usage if available
                tokens_used = 0
                if "usage" in result and "total_tokens" in result["usage"]:
                    tokens_used = result["usage"]["total_tokens"]
                    logger.debug(f"[{request_id}] Token usage: {tokens_used} total tokens")
                
                # Record successful request in global service stats
                service_stats.record_successful_request(current_time, tokens_used)
            else:
                result = {"text": response.text}
                logger.debug(f"[{request_id}] Non-JSON response received: {len(response.text)} bytes")
                
                # Record successful request in global service stats (without token count)
                service_stats.record_successful_request(current_time)
            
            return result
        
        except httpx.HTTPStatusError as e:
            # Handle API errors
            error_detail = "Unknown error"
            headers = {}
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            try:
                error_response = e.response.json()
                error_detail = error_response.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            
            status_code = e.response.status_code
            
            # Record upstream error from the endpoint in instance
            instance.record_upstream_error(status_code)
            
            # Record upstream error in global service stats
            service_stats.record_upstream_error(current_time)
            
            logger.error(f"[{request_id}] Instance {instance.name}[{instance.api_base}] API error: {status_code} - {error_detail} (time={elapsed_ms}ms)")
            
            # Pass along retry-after header for rate limiting
            if status_code == 429 and "retry-after" in e.response.headers:
                retry_after = e.response.headers["retry-after"]
                headers["retry-after"] = retry_after
                logger.warning(f"[{request_id}] Rate limit exceeded for instance {instance.name}, retry-after: {retry_after}")
            
            # Map Azure errors to appropriate status codes
            raise HTTPException(
                status_code=status_code,
                detail=f"API error from instance {instance.name}: {error_detail}",
                headers=headers,
            )
        
        except (httpx.RequestError, httpx.TimeoutException) as e:
            # Handle connection errors
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_type = type(e).__name__
            
            # Record as a downstream error (typically results in a 503 response to client)
            instance.record_error(503)
            
            logger.warning(f"[{request_id}] Connection error to instance {instance.name}: {error_type}: {str(e)} (time={elapsed_ms}ms)")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Connection error to API instance {instance.name}: {str(e)}",
            )
        except Exception as e:
            # Handle unexpected errors
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_type = type(e).__name__
            
            # Record as downnstream 500 error
            instance.record_upstream_error(500)
            
            # Record upstream error in global service stats
            service_stats.record_upstream_error(current_time)
            
            logger.error(f"[{request_id}] Unexpected error with instance {instance.name}: {error_type}: {str(e)} (time={elapsed_ms}ms)")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error with API instance {instance.name}: {str(e)}",
            )

    async def try_instances(self,
                          instances: Dict[str, APIInstance],
                          endpoint: str, 
                          deployment: str, 
                          payload: Dict[str, Any], 
                          required_tokens: int,
                          instance_router,
                          method: str = "POST",
                          provider_type: Optional[str] = None) -> Tuple[Dict[str, Any], APIInstance]:
        """
        Try instances until one succeeds or all fail.
        
        Args:
            instances: Dictionary of available API instances
            endpoint: The API endpoint
            deployment: The deployment name
            payload: The request payload
            required_tokens: Estimated tokens required for the request
            instance_router: The router to select instances
            method: HTTP method
            provider_type: Optional provider type to filter instances (e.g., "azure" or "generic")
            
        Returns:
            Tuple of (response, instance used)
            
        Raises:
            HTTPException: If all instances fail
        """
        current_time = int(time.time())
        
        # Extract model name from payload if available
        model_name = None
        if "model" in payload:
            model_name = payload["model"]
            logger.debug(f"Extracted model name from payload: {model_name}")
        
        # Try strategy-based instance first, prioritizing instances that support this model
        primary_instance = instance_router.select_instance(instances, required_tokens, model_name)
        if primary_instance:
            logger.debug(f"Selected primary instance: {primary_instance.name if primary_instance else 'None'} for model: {model_name} with max input tokens: {primary_instance.max_input_tokens} vs required tokens: {required_tokens}")
            try:
                response = await self.forward_request(primary_instance, endpoint, deployment, payload, method)
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
                    logger.warning(f"Error from instance {primary_instance.name}: {detail}")
                
                # Fall through to try other instances
            except Exception as e:
                # Instance errors already recorded in forward_request
                primary_instance.mark_error(str(e))
                logger.warning(f"Unexpected error from instance {primary_instance.name}: {str(e)}")
                # Fall through to try other instances
        else: 
            error_message = "No instance found"
            logger.error(error_message)
            
            # 全局记录客户端错误，不再依赖单个实例
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            service_stats.record_client_error(status_code, current_time)
                
            raise HTTPException(
                status_code=status_code,
                detail=error_message,
            )

        # If primary instance failed, try all other instances that support this model
        available_instances = instance_router.get_available_instances_for_model(instances, model_name, provider_type)
        
        # Remove the primary instance from the list if we already tried it
        if primary_instance:
            available_instances = [i for i in available_instances if i.name != primary_instance.name]
        
        errors = []
        
        for instance in available_instances:
            # Skip instances that are rate limited
            if instance.is_rate_limited():
                logger.debug(f"Skipping rate-limited instance {instance.name}")
                continue
                
            try:
                response = await self.forward_request(instance, endpoint, deployment, payload, method)
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
                    logger.warning(f"Error from instance {instance.name}: {detail}")
                
                errors.append(f"{instance.name}: {detail}")
            except Exception as e:
                # Instance errors already recorded in forward_request
                instance.mark_error(str(e))
                logger.error(f"Unexpected error from instance {instance.name}: {str(e)}")
                errors.append(f"{instance.name}: {str(e)}")
        
        # 所有实例都失败了 - 这是将发送给客户端的错误
        error_message = f"All API instances failed: {'; '.join(errors)}"
        logger.error(error_message)
        
        # 全局记录客户端错误，不再依赖单个实例
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        service_stats.record_client_error(status_code, current_time)
        
        # No instance available, return 503 Service Unavailable
        raise HTTPException(
            status_code=status_code,
            detail=error_message,
        )

    async def try_specific_instance(self,
                                  instance_name: str,
                                  endpoint: str, 
                                  deployment: str, 
                                  payload: Dict[str, Any], 
                                  required_tokens: int,
                                  method: str = "POST") -> Tuple[Dict[str, Any], APIInstance]:
        """
        Try a specific instance to handle a request.
        
        This is useful for testing or verification purposes when you want to ensure
        a request is handled by a specific instance rather than using the load balancing.
        
        Args:
            instance_name: The name of the instance to use
            endpoint: The API endpoint
            deployment: The deployment name
            payload: The request payload
            required_tokens: Estimated tokens required for the request
            method: HTTP method
            
        Returns:
            Tuple of (response, instance used)
            
        Raises:
            HTTPException: If the instance fails or doesn't exist
        """
        from app.instance.manager import instance_manager
        
        # Check if the instance exists
        if instance_name not in instance_manager.instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance '{instance_name}' not found"
            )
            
        # Get the instance
        instance = instance_manager.instances[instance_name]
        
        # Try to forward the request to the specific instance
        try:
            response = await self.forward_request(instance, endpoint, deployment, payload, method)
            
            # Mark instance as healthy and update TPM
            instance.mark_healthy()
            if "usage" in response and "total_tokens" in response["usage"]:
                instance.update_tpm_usage(response["usage"]["total_tokens"])
                
            return response, instance
        except HTTPException as e:
            # If the instance fails, mark it accordingly and re-raise the exception
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
                
                instance.mark_rate_limited(retry_after)
                logger.warning(f"Instance {instance.name} rate limited: {detail}")
            else:
                instance.mark_error(str(e))
                logger.warning(f"Error from instance {instance.name}: {detail}")
            
            raise
        except Exception as e:
            # If there's an unexpected error, mark the instance as having an error
            instance.mark_error(str(e))
            logger.error(f"Unexpected error from instance {instance.name}: {str(e)}")
            
            # Re-raise the exception with a more informative message
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error with API instance {instance.name}: {str(e)}"
            )

# Create a singleton instance
instance_forwarder = RequestForwarder() 