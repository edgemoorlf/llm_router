"""Instance manager for multiple OpenAI-compatible service instances."""
import os
import time
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
import httpx
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from .routing_strategy import RoutingStrategy
from .api_instance import APIInstance, InstanceStatus

class InstanceManager:
    """Manager for multiple API service instances with load balancing and failover."""
    
    def __init__(self, routing_strategy: RoutingStrategy = RoutingStrategy.FAILOVER):
        """
        Initialize the instance manager.
        
        Args:
            routing_strategy: Strategy for selecting instances
        """
        self.instances: Dict[str, APIInstance] = {}
        self.routing_strategy = routing_strategy
        self.round_robin_index = 0
        self._load_instances_from_env()
        
        if not self.instances:
            logger.warning("No API instances configured. Falling back to legacy single instance configuration.")
            self._load_legacy_instance()
        
        logger.info(f"Initialized {len(self.instances)} API instances with {routing_strategy} routing strategy")
    
    def load_from_csv(self, csv_path: str) -> None:
        """Load instances from a CSV file, replacing existing configuration."""
        self.instances.clear()
        self._load_instances_from_csv(csv_path)
        if not self.instances:
            logger.warning("No instances loaded from CSV. Falling back to environment variables.")
            self._load_instances_from_env()
        logger.info(f"Loaded {len(self.instances)} instances from CSV")

    def _load_instances_from_csv(self, csv_path: str) -> None:
        """Load instances from a CSV file."""
        try:
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_idx, row in enumerate(reader, start=1):
                    try:
                        name = row.get('name', f'instance-{row_idx}')
                        instance = APIInstance(
                            name=name,
                            provider_type=row.get('provider_type', 'azure').lower(),
                            api_key=row['API_KEY'],
                            api_base=row['API_BASE'],
                            api_version=row.get('api_version', '2023-07-01-preview'),
                            proxy_url=row.get('proxy_url'),
                            priority=int(row.get('priority', 100)),
                            weight=int(row.get('weight', 100)),
                            max_tpm=int(row.get('max_tpm', 240000)),
                            max_input_tokens=int(row.get('max_input_tokens', 0)),
                            supported_models=[m.strip() for m in row.get('model_name', '').split(',') if m.strip()],
                        )
                        instance.initialize_client()
                        self.instances[name] = instance
                        logger.info(f"Loaded API instance {name} from CSV")
                    except KeyError as e:
                        logger.error(f"Missing required column {e} in CSV row {row_idx}")
                    except Exception as e:
                        logger.error(f"Error loading instance from CSV row {row_idx}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load instances from CSV: {str(e)}")
            raise

    def _load_instances_from_env(self) -> None:
        """Load instances from environment variables."""
        instance_names = os.getenv("API_INSTANCES", "").split(",")
        instance_names = [name.strip() for name in instance_names if name.strip()]        
        
        for name in instance_names:
            prefix = f"API_INSTANCE_{name.upper()}_"
            
            if not os.getenv(f"{prefix}API_KEY") or not os.getenv(f"{prefix}API_BASE"):
                logger.warning(f"Skipping incomplete instance configuration for {name}")
                continue
            
            try:
                # Create the instance with basic configuration
                instance = APIInstance(
                    name=name,
                    provider_type=os.getenv(f"{prefix}PROVIDER_TYPE", "azure").lower(),
                    api_key=os.getenv(f"{prefix}API_KEY", ""),
                    api_base=os.getenv(f"{prefix}API_BASE", ""),
                    api_version=os.getenv(f"{prefix}API_VERSION", "2023-07-01-preview"),
                    priority=int(os.getenv(f"{prefix}PRIORITY", "100")),
                    weight=int(os.getenv(f"{prefix}WEIGHT", "100")),
                    max_tpm=int(os.getenv(f"{prefix}MAX_TPM", "240000")),
                    max_input_tokens=int(os.getenv(f"{prefix}MAX_INPUT_TOKENS", "0")),
                )
                
                # Load supported models for this instance
                models_str = os.getenv(f"{prefix}SUPPORTED_MODELS", "")
                if models_str:
                    instance.supported_models = [model.strip() for model in models_str.split(",") if model.strip()]
                    logger.info(f"Instance {name} supports models: {', '.join(instance.supported_models)}")
                
                # Load model to deployment mappings for this instance
                # Format: MODEL_MAP_<model>=<deployment>
                for key, value in os.environ.items():
                    if key.startswith(f"{prefix}MODEL_MAP_"):
                        model_name = key[len(f"{prefix}MODEL_MAP_"):].lower().replace("_", "-")
                        deployment_name = value
                        instance.model_deployments[model_name] = deployment_name
                        logger.info(f"Instance {name} maps model '{model_name}' to deployment '{deployment_name}'")
                
                instance.initialize_client()
                self.instances[name] = instance
                logger.info(f"Loaded API instance {name} with priority {instance.priority}, weight {instance.weight}, max TPM {instance.max_tpm}")
            except Exception as e:
                logger.error(f"Error loading instance {name}: {str(e)}")
    
    def _load_legacy_instance(self) -> None:
        """Load the legacy single instance configuration."""
        api_key = os.getenv("API_KEY")
        api_base = os.getenv("API_BASE")
        api_version = os.getenv("API_VERSION", "2023-07-01-preview")
        
        if not api_key or not api_base:
            logger.error("No API instances configured and legacy configuration is incomplete")
            return
        
        instance = APIInstance(
            name="default",
            provider_type=os.getenv("API_PROVIDER_TYPE", "azure").lower(),
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            priority=1,
            weight=100,
            max_tpm=240000,  # Default to 240K TPM
        )
        instance.initialize_client()
        self.instances["default"] = instance
        logger.info("Loaded legacy API instance configuration")
    
    def select_instance(self, required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        """
        Select an API instance based on the configured routing strategy and model support.
        
        Args:
            required_tokens: Estimated number of tokens required for the request
            model_name: The model name requested (optional)
            
        Returns:
            Selected instance or None if no suitable instance is available
        """
        # Filter healthy instances with enough capacity
        available_instances = [
            instance for instance in self.instances.values()
            if (instance.status == InstanceStatus.HEALTHY or 
                (instance.status == InstanceStatus.RATE_LIMITED and time.time() >= (instance.rate_limited_until or 0)))
            and (instance.current_tpm + required_tokens) <= instance.max_tpm
            and (instance.max_input_tokens == 0 or required_tokens <= instance.max_input_tokens)
        ]
        
        # Enforce max input tokens strictly
        available_instances = [
            instance for instance in available_instances
            if instance.max_input_tokens == 0 or required_tokens <= instance.max_input_tokens
        ]
        
        # If a model name is provided, filter instances that support this model
        if model_name:
            # Normalize the model name
            normalized_model = model_name.lower().split(':')[0]
            
            # Filter instances that explicitly support this model
            model_instances = [
                instance for instance in available_instances
                if (
                    # Instance has this model in its supported_models list
                    normalized_model in [m.lower() for m in instance.supported_models] or
                    # Instance has a deployment mapping for this model
                    normalized_model in instance.model_deployments or
                    # Instance has no model restrictions (empty supported_models means it supports all models)
                    not instance.supported_models
                )
            ]
            
            # If we found instances that support this model, use them
            if model_instances:
                available_instances = model_instances
                logger.debug(f"Found {len(model_instances)} instances supporting model '{model_name}'")
            else:
                logger.warning(f"No instances explicitly support model '{model_name}', falling back to all available instances")
        
        if not available_instances:
            logger.warning(f"No healthy instances available with enough TPM capacity for {required_tokens} tokens")
            # Try to find any instance that's not in error state but still respects token limits
            available_instances = [
                instance for instance in self.instances.values()
                if instance.status != InstanceStatus.ERROR
                and (instance.current_tpm + required_tokens) <= instance.max_tpm
                and (instance.max_input_tokens == 0 or required_tokens <= instance.max_input_tokens)
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
            
        elif self.routing_strategy == RoutingStrategy.MODEL_SPECIFIC:
            # For model-specific routing, we've already filtered instances by model support above
            # Now sort by priority within those instances
            available_instances.sort(key=lambda x: x.priority)
            
            # If we have a model name, log which instance we're using for it
            if model_name:
                logger.info(f"Using model-specific routing: selected instance {available_instances[0].name} for model {model_name}")
            
            return available_instances[0]
            
        # Default to first instance
        return available_instances[0]
    
    async def try_instances(self, 
                           endpoint: str, 
                           deployment: str, 
                           payload: Dict[str, Any], 
                           required_tokens: int,
                           method: str = "POST",
                           provider_type: Optional[str] = None) -> Tuple[Dict[str, Any], APIInstance]:
        """
        Try instances until one succeeds or all fail.
        
        Args:
            endpoint: The API endpoint
            deployment: The deployment name
            payload: The request payload
            required_tokens: Estimated tokens required for the request
            method: HTTP method
            provider_type: Optional provider type to filter instances (e.g., "azure" or "generic")
            
        Returns:
            Tuple of (response, instance used)
            
        Raises:
            HTTPException: If all instances fail
        """
        # Filter out instances in error state and by provider type if specified
        available_instances = [
            instance for instance in self.instances.values() 
            if instance.status != InstanceStatus.ERROR and
               (provider_type is None or instance.provider_type == provider_type)
        ]
        
        if not available_instances:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No available API instances",
            )
        
        # Extract model name from payload if available
        model_name = None
        if "model" in payload:
            model_name = payload["model"]
            logger.debug(f"Extracted model name from payload: {model_name}")
        
        # Try strategy-based instance first, prioritizing instances that support this model
        primary_instance = self.select_instance(required_tokens, model_name)
        if primary_instance:
            logger.debug(f"Selected primary instance: {primary_instance.name if primary_instance else 'None'} for model: {model_name} with max input tokens: {primary_instance.max_input_tokens} vs required tokens: {required_tokens}")
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
        else: 
            error_message = "No instance found"
            logger.error(error_message)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_message,
        )


        # If primary instance failed or not found, try all other instances in priority order
        # Sort by priority and filter by model support if possible
        available_instances.sort(key=lambda x: x.priority)
        
        # If we have a model name, prioritize instances that support this model
        if model_name:
            # Normalize the model name
            normalized_model = model_name.lower().split(':')[0]
            
            # Move instances that support this model to the front of the list
            model_supporting_instances = [
                instance for instance in available_instances
                if (
                    # Instance has this model in its supported_models list
                    normalized_model in [m.lower() for m in instance.supported_models] or
                    # Instance has a deployment mapping for this model
                    normalized_model in instance.model_deployments or
                    # Instance has no model restrictions (empty supported_models means it supports all models)
                    not instance.supported_models
                )
            ]
            
            # Only use model-supporting instances if we found any
            if model_supporting_instances:
                logger.debug(f"Prioritizing {len(model_supporting_instances)} instances that support model '{model_name}' for failover")
                available_instances = model_supporting_instances + [
                    instance for instance in available_instances 
                    if instance not in model_supporting_instances
                ]
        
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
        error_message = f"All API instances failed: {'; '.join(errors)}"
        logger.error(error_message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_message,
        )
    
    async def _forward_request(self, 
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
                if "usage" in result and "total_tokens" in result["usage"]:
                    logger.debug(f"[{request_id}] Token usage: {result['usage']['total_tokens']} total tokens")
            else:
                result = {"text": response.text}
                logger.debug(f"[{request_id}] Non-JSON response received: {len(response.text)} bytes")
            
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
            logger.error(f"[{request_id}] Connection error to instance {instance.name}: {error_type}: {str(e)} (time={elapsed_ms}ms)")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Connection error to API instance {instance.name}: {str(e)}",
            )
        except Exception as e:
            # Handle unexpected errors
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_type = type(e).__name__
            logger.error(f"[{request_id}] Unexpected error with instance {instance.name}: {error_type}: {str(e)} (time={elapsed_ms}ms)")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error with API instance {instance.name}: {str(e)}",
            )
    
    def get_instance_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all instances."""
        return [
            {
                "name": instance.name,
                "provider_type": instance.provider_type,
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
                "supported_models": instance.supported_models,
                "model_deployments": instance.model_deployments,
            }
            for instance in self.instances.values()
        ]


# Create a singleton instance with the default routing strategy
instance_manager = InstanceManager(
    routing_strategy=RoutingStrategy(os.getenv("API_ROUTING_STRATEGY", RoutingStrategy.FAILOVER))
)
