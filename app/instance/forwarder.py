import time
import random
import logging
from typing import Dict, Any, Tuple, Optional
import httpx
from fastapi import HTTPException, status

from app.models.instance import InstanceConfig, InstanceState
from .service_stats import service_stats

# Import instance manager from context
from app.instance.instance_context import instance_manager

logger = logging.getLogger(__name__)

class RequestForwarder:
    """Forwards API requests to selected instances and handles responses."""

    @staticmethod
    async def forward_request(
            instance: Tuple[InstanceConfig, InstanceState], 
            endpoint: str, 
            payload: Dict[str, Any],
            method: str = "POST") -> Dict[str, Any]:
        """
        Forward a request to an API instance.
        
        Args:
            instance: The instance to use as (config, state) tuple
            endpoint: The API endpoint
            payload: The request payload
            method: HTTP method
            
        Returns:
            The API response
            
        Raises:
            HTTPException: If the request fails
        """
        config, state = instance
        
        # Extract model from payload
        model_name = payload.get("model", "").lower() if payload and "model" in payload else ""
        
        # Determine deployment name based on the model
        deployment = ""
        if model_name and config.provider_type == "azure" and config.model_deployments:
            # Look up the deployment name for this model in this specific instance
            deployment = config.model_deployments.get(model_name, "")
            if deployment:
                logger.debug(f"Resolved deployment '{deployment}' for model '{model_name}' in instance '{config.name}'")
            else:
                logger.warning(f"No deployment mapping found for model '{model_name}' in instance '{config.name}'")
        
        url = config.build_url(endpoint, deployment)
        state.last_used = time.time()
        current_time = int(time.time())
        
        # Update request per minute counter for instance
        state.update_rpm_usage()
        
        # Record request in global service stats
        service_stats.record_request(current_time)
        
        # Log request details at debug level
        request_id = f"req-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
        logger.debug(f"[{request_id}] Forwarding request to instance {config.name} ({config.provider_type}): {url}")
        
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
                response = await config.client.post(url, json=payload)
            elif method.upper() == "GET":
                response = await config.client.get(url, params=payload)
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
            state.record_upstream_error(status_code)
            
            # Record upstream error in global service stats
            service_stats.record_upstream_error(current_time)
            
            logger.error(f"[{request_id}] Instance {config.name}[{config.api_base}] API error: {status_code} - {error_detail} (time={elapsed_ms}ms)")
            
            # Pass along retry-after header for rate limiting
            if status_code == 429 and "retry-after" in e.response.headers:
                retry_after = e.response.headers["retry-after"]
                headers["retry-after"] = retry_after
                logger.warning(f"[{request_id}] Rate limit exceeded for instance {config.name}, retry-after: {retry_after}")
            
            # Map Azure errors to appropriate status codes
            raise HTTPException(
                status_code=status_code,
                detail=f"API error from instance {config.name}: {error_detail}",
                headers=headers,
            )
        
        except (httpx.RequestError, httpx.TimeoutException) as e:
            # Handle connection errors
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_type = type(e).__name__
            
            # Record as a downstream error (typically results in a 503 response to client)
            state.record_error(503)
            
            logger.warning(f"[{request_id}] Connection error to instance {config.name}: {error_type}: {str(e)} (time={elapsed_ms}ms)")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Connection error to API instance {config.name}: {str(e)}",
            )
        except Exception as e:
            # Handle unexpected errors
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_type = type(e).__name__
            
            # Record as downnstream 500 error
            state.record_upstream_error(500)
            
            # Record upstream error in global service stats
            service_stats.record_upstream_error(current_time)
            
            logger.error(f"[{request_id}] Unexpected error with instance {config.name}: {error_type}: {str(e)} (time={elapsed_ms}ms)")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error with API instance {config.name}: {str(e)}",
            )

    async def try_instances(self,
                          instances: Dict[str, Tuple[InstanceConfig, InstanceState]],
                          endpoint: str, 
                          payload: Dict[str, Any], 
                          required_tokens: int,
                          instance_router,
                          method: str = "POST",
                          provider_type: Optional[str] = None) -> Tuple[Dict[str, Any], Tuple[InstanceConfig, InstanceState]]:
        """
        Try instances until one succeeds or all fail.
        
        Args:
            instances: Dictionary of available API instances as (config, state) tuples
            endpoint: The API endpoint
            payload: The request payload
            required_tokens: Estimated tokens required for the request
            instance_router: The router to select instances
            method: HTTP method
            provider_type: Optional provider type to filter instances (e.g., "azure" or "generic")
            
        Returns:
            Tuple of (response, instance tuple used)
            
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
            config, state = primary_instance
            logger.debug(f"Selected primary instance: {config.name} for model: {model_name} with max input tokens: {config.max_input_tokens} vs required tokens: {required_tokens}")
            try:
                response = await self.forward_request(primary_instance, endpoint, payload, method)
                # Mark instance as healthy and update TPM
                state.mark_healthy()
                if "usage" in response and "total_tokens" in response["usage"]:
                    state.update_tpm_usage(response["usage"]["total_tokens"])
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
                    
                    state.mark_rate_limited(retry_after)
                    logger.warning(f"Instance {config.name} rate limited: {detail}")
                else:
                    state.mark_error(str(e))
                    logger.warning(f"Error from instance {config.name}: {detail}")
                
                # Fall through to try other instances
            except Exception as e:
                # Instance errors already recorded in forward_request
                state.mark_error(str(e))
                logger.warning(f"Unexpected error from instance {config.name}: {str(e)}")
                # Fall through to try other instances
        else: 
            error_message = "No instance found"
            if model_name:
                error_message += f" supporting model {model_name}"
            logger.error(error_message)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_message
            )

    async def try_specific_instance(
        self, 
        instance_name: str, 
        endpoint: str, 
        payload: Dict[str, Any],
        method: str = "POST"
    ) -> Dict[str, Any]:
        """
        Try a specific instance by name.
        
        Args:
            instance_name: The name of the instance to use
            endpoint: The API endpoint
            payload: The request payload
            method: HTTP method
            
        Returns:
            API response
            
        Raises:
            HTTPException: If the instance fails or doesn't exist
        """
        from app.instance.manager import instance_manager
        
        # Check if the instance exists
        instance = instance_manager.get_instance(instance_name)
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance '{instance_name}' not found"
            )
        
        # Try to forward the request to the specific instance
        try:
            response = await self.forward_request(instance, endpoint, payload, method)
            
            # Mark instance as healthy and update TPM
            instance.mark_healthy()
            if "usage" in response and "total_tokens" in response["usage"]:
                instance.update_tpm_usage(response["usage"]["total_tokens"])
                
            return response
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