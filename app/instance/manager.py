"""Instance manager for multiple OpenAI-compatible service instances."""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import uuid
import time
import traceback

logger = logging.getLogger(__name__)

from .routing_strategy import RoutingStrategy
from .api_instance import APIInstance
from .router import InstanceRouter
from .forwarder import RequestForwarder
from .monitor import InstanceMonitor
from .state_manager import StateManager, create_state_manager

# Import the configuration system
from app.config import config_loader
from app.config.config_hierarchy import config_hierarchy

class InstanceManager:
    """
    Manages OpenAI-compatible API instances.
    
    This class handles the loading, tracking, and selection of API instances
    for forwarding requests to OpenAI-compatible APIs. Instances can be loaded from:
    
    1. SQLite database (highest priority)
    2. State files (temporary persistence)
    3. YAML configuration files in app/config/instances/
    4. Default values (lowest priority)
    
    The manager also tracks instance health, usage statistics, and handles
    the selection of the appropriate instance for a request based on the
    current routing strategy.
    """
    
    def __init__(self, routing_strategy: Optional[RoutingStrategy] = None, state_manager: Optional[StateManager] = None):
        """
        Initialize the instance manager.
        
        Args:
            routing_strategy: Strategy for selecting instances
            state_manager: StateManager for persisting instance state
        """
        self.instances: Dict[str, APIInstance] = {}
        self.instances_lock = threading.RLock()  # Lock for thread-safe access to instances
        self.router = None  # Will be initialized later
        self.forwarder = RequestForwarder()
        self.monitor = InstanceMonitor()
        
        # Create worker ID and initialize last state check time
        self.worker_id = str(uuid.uuid4())
        self.last_state_check = 0
        
        # Initialize state manager (use SQLite-based by default)
        self.state_manager = state_manager or create_state_manager("sqlite")
        
        try:
            # Load configuration using the config hierarchy
            config = config_hierarchy.get_configuration()
            
            # Use routing strategy from config if not provided
            if routing_strategy is None and 'routing' in config and 'strategy' in config['routing']:
                routing_strategy = RoutingStrategy(config['routing']['strategy'])
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Fall back to default routing strategy
            if routing_strategy is None:
                routing_strategy = RoutingStrategy.FAILOVER
            config = None
            
        # Initialize router now that we have the routing strategy
        self.router = InstanceRouter(routing_strategy)
        
        # First try to load state from the state manager (highest priority)
        if self._load_state():
            logger.info(f"Loaded instance state from state manager: {len(self.instances)} instances")
        else:
            # Load instances from configuration hierarchy
            try:
                if config and 'instances' in config:
                    self._load_instances_from_hierarchy(config['instances'])
                else:
                    logger.warning("No valid configuration found or no instances defined in config")
            except Exception as e:
                logger.error(f"Error loading instances from config: {str(e)}")
            
            # If no instances found, provide a clear error
            if not self.instances:
                logger.error("No API instances found in configuration hierarchy. The application requires at least one instance defined.")
                self.instances = {}
            
            # Set RPM window for all instances
            try:
                if config and 'monitoring' in config and 'stats_window_minutes' in config['monitoring'] and config['monitoring']['stats_window_minutes'] > 0:
                    self.set_rpm_window_for_all(config['monitoring']['stats_window_minutes'])
            except Exception as e:
                logger.error(f"Error setting RPM window: {str(e)}")
            
            # Save initial state
            self._save_state()
        
        logger.info(f"Initialized {len(self.instances)} API instances with {routing_strategy} routing strategy")
    
    def _load_instances_from_hierarchy(self, instances_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Load instances from configuration hierarchy.
        
        Args:
            instances_config: Dictionary of instance configurations
        """
        for name, config in instances_config.items():
            try:
                # Convert dictionary config to APIInstance
                instance = APIInstance(
                    name=name,
                    provider_type=config.get('provider_type', 'generic'),
                    api_key=config.get('api_key', ''),
                    api_base=config.get('api_base', ''),
                    api_version=config.get('api_version', '2023-05-15'),
                    proxy_url=config.get('proxy_url'),
                    priority=config.get('priority', 100),
                    weight=config.get('weight', 100),
                    max_tpm=config.get('max_tpm', 240000),
                    max_input_tokens=config.get('max_input_tokens', 0),
                    supported_models=config.get('supported_models', []),
                    model_deployments=config.get('model_deployments', {})
                )
                instance.initialize_client()
                self.instances[name] = instance
                
                # Get instance source information for logging
                source_info = config_hierarchy.get_instance_source(name)
                sources = source_info.get('sources', {})
                source_list = [f"{key}: {source}" for key, source in sources.items()]
                
                logger.info(f"Loaded API instance {name} from configuration hierarchy")
                logger.debug(f"Instance {name} configuration sources: {', '.join(source_list)}")
            except Exception as e:
                logger.error(f"Error loading instance {name} from configuration: {str(e)}")
    
    def _load_state(self) -> bool:
        """
        Load instance state from the state manager.
        
        Returns:
            True if state was loaded successfully, False otherwise
        """
        try:
            state = self.state_manager.load_state()
            if not state or 'instances' not in state:
                return False
                
            with self.instances_lock:
                # Clear existing instances
                self.instances.clear()
                
                # Load instances from state
                for instance_data in state.get('instances', []):
                    try:
                        instance = APIInstance(
                            name=instance_data['name'],
                            provider_type=instance_data['provider_type'],
                            api_key=instance_data['api_key'],
                            api_base=instance_data['api_base'],
                            api_version=instance_data.get('api_version', '2023-05-15'),
                            proxy_url=instance_data.get('proxy_url'),
                            priority=instance_data.get('priority', 100),
                            weight=instance_data.get('weight', 100),
                            max_tpm=instance_data.get('max_tpm', 240000),
                            max_input_tokens=instance_data.get('max_input_tokens', 0),
                            supported_models=instance_data.get('supported_models', []),
                            model_deployments=instance_data.get('model_deployments', {})
                        )
                        instance.initialize_client()
                        
                        # Load stats if available
                        if 'stats' in instance_data:
                            instance.error_count = instance_data['stats'].get('error_count', 0)
                            instance.last_error = instance_data['stats'].get('last_error')
                            instance.rate_limited_until = instance_data['stats'].get('rate_limited_until')
                            
                        self.instances[instance_data['name']] = instance
                    except Exception as e:
                        logger.error(f"Error loading instance {instance_data.get('name', 'unknown')} from state: {str(e)}")
                
            # Update the database source in configuration hierarchy
            config_hierarchy.sources['database'].timestamp = time.time()
            
            return True
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False
    
    def _save_state(self) -> bool:
        """
        Save instance state using the state manager.
        
        Returns:
            True if state was saved successfully, False otherwise
        """
        try:
            instances_data = []
            
            with self.instances_lock:
                for name, instance in self.instances.items():
                    instance_data = {
                        'name': instance.name,
                        'provider_type': instance.provider_type,
                        'api_key': instance.api_key,
                        'api_base': instance.api_base,
                        'api_version': instance.api_version,
                        'proxy_url': instance.proxy_url,
                        'priority': instance.priority,
                        'weight': instance.weight,
                        'max_tpm': instance.max_tpm,
                        'max_input_tokens': instance.max_input_tokens,
                        'supported_models': instance.supported_models,
                        'model_deployments': instance.model_deployments,
                        'stats': {
                            'error_count': instance.error_count,
                            'last_error': instance.last_error,
                            'rate_limited_until': instance.rate_limited_until,
                        }
                    }
                    instances_data.append(instance_data)
            
            return self.state_manager.save_state(instances_data, self.worker_id)
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return False
    
    def reload_config(self) -> None:
        """Reload configuration from disk and update instances."""
        try:
            # Reload configuration using the config hierarchy
            config = config_hierarchy.get_configuration()
            if not config or 'instances' not in config:
                logger.error("Failed to reload configuration")
                return
                
            # Clear existing instances
            with self.instances_lock:
                self.instances.clear()
            
            # Load instances from configuration hierarchy
            if 'instances' in config and config['instances']:
                self._load_instances_from_hierarchy(config['instances'])
            else:
                logger.warning("No instances defined in reloaded configuration")
            
            # If no instances found, provide a clear error
            if not self.instances:
                logger.error("No API instances found in configuration. The application requires at least one instance defined.")
                self.instances = {}
            
            # Update routing strategy
            try:
                if 'routing' in config and 'strategy' in config['routing']:
                    self.router.strategy = RoutingStrategy(config['routing']['strategy'])
            except Exception as e:
                logger.error(f"Error updating routing strategy: {str(e)}")
            
            # Set RPM window for all instances
            try:
                if 'monitoring' in config and 'stats_window_minutes' in config['monitoring'] and config['monitoring']['stats_window_minutes'] > 0:
                    self.set_rpm_window_for_all(config['monitoring']['stats_window_minutes'])
            except Exception as e:
                logger.error(f"Error setting RPM window: {str(e)}")
            
            # Save state after reloading config
            self._save_state()
            
            logger.info(f"Reloaded configuration with {len(self.instances)} instances")
        except Exception as e:
            logger.error(f"Error reloading configuration: {str(e)}")
            raise
    
    def check_for_updates(self) -> bool:
        """
        Check for updates from other workers.
        
        Returns:
            True if updates were loaded, False otherwise
        """
        has_updates, current_time = self.state_manager.check_for_updates(self.last_state_check)
        
        # Update last check time
        self.last_state_check = current_time
        
        # If updates are available, reload state
        if has_updates:
            return self._load_state()
            
        return False

    def add_instance(self, name: str, instance: APIInstance) -> None:
        """
        Add an instance to the manager.
        
        Args:
            name: Name of the instance
            instance: The APIInstance object
        """
        with self.instances_lock:
            self.instances[name] = instance
            # Save state after modifying instances
            self._save_state()
            
    def remove_instance(self, name: str) -> bool:
        """
        Remove an instance from the manager.
        
        Args:
            name: Name of the instance to remove
            
        Returns:
            True if the instance was removed, False if it was not found
        """
        with self.instances_lock:
            if name in self.instances:
                del self.instances[name]
                # Save state after modifying instances
                self._save_state()
                return True
            return False
    
    def has_instance(self, name: str) -> bool:
        """
        Check if an instance exists in the manager.
        
        Args:
            name: Name of the instance to check
            
        Returns:
            True if the instance exists, False otherwise
        """
        with self.instances_lock:
            return name in self.instances
    
    def get_instance(self, name: str) -> Optional[APIInstance]:
        """
        Get an instance by name.
        
        Args:
            name: Name of the instance to get
            
        Returns:
            The instance if it exists, None otherwise
        """
        with self.instances_lock:
            return self.instances.get(name)
    
    def get_all_instances(self) -> Dict[str, APIInstance]:
        """
        Get all instances.
        
        Returns:
            Dictionary of instance name to APIInstance
        """
        with self.instances_lock:
            # Return a copy to avoid external modifications
            return dict(self.instances)

    def select_instance(self, required_tokens: int, model_name: Optional[str] = None) -> Optional[APIInstance]:
        """
        Select an instance based on the router strategy.
        
        Args:
            required_tokens: Estimated number of tokens required
            model_name: Optional model name to filter instances
            
        Returns:
            Selected instance or None if no suitable instance is available
        """
        if not self.instances:
            logger.error("No instances available for selection")
            return None
            
        logger.debug(f"Selecting instance for model '{model_name}' requiring {required_tokens} tokens")
        
        # Lock access to the instances dictionary to prevent concurrent modification during selection
        with self.instances_lock:
            instance = self.router.select_instance(self.instances, required_tokens, model_name)
            
        if instance:
            logger.info(f"Selected instance '{instance.name}' for model '{model_name}'")
            if model_name:
                # Log the exact model deployment that will be used
                if model_name.lower() in instance.model_deployments:
                    deployment = instance.model_deployments[model_name.lower()]
                    logger.info(f"Using deployment '{deployment}' for model '{model_name}'")
                else:
                    # Check if model_name is in supported_models
                    if model_name.lower() in [m.lower() for m in instance.supported_models]:
                        logger.info(f"Model '{model_name}' is in supported_models but has no specific deployment mapping - instance will use default deployments")
                    else:
                        logger.warning(f"Model '{model_name}' is neither in supported_models nor has a mapping in model_deployments for instance '{instance.name}'")
        else:
            logger.warning(f"No suitable instance found for model '{model_name}' requiring {required_tokens} tokens")
            # Add a stronger warning if we have instances but none match this model
            if self.instances:
                supported_models = set()
                for inst in self.instances.values():
                    if inst.supported_models:
                        supported_models.update([m.lower() for m in inst.supported_models])
                    if inst.model_deployments:
                        supported_models.update([k.lower() for k in inst.model_deployments.keys()])
                
                if supported_models:
                    logger.error(f"Available models: {sorted(list(supported_models))}")
                    logger.error(f"The requested model '{model_name}' is not supported by any configured instance - request will fail")
            
        return instance
    
    async def try_instances(self, 
                           endpoint: str, 
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
        return await self.forwarder.try_instances(
            self.instances,
            endpoint,
            payload,
            required_tokens,
            self.router,
            method,
            provider_type
        )
    
    def get_instance_stats(self, instance_name: Optional[str] = None) -> Union[List[Dict[str, Any]], Dict[str, Any], None, Dict[str, Any]]:
        """
        Get statistics for all instances or a specific instance.
        
        Note: This is an enhanced version that supports both getting stats for all instances
        and for a single instance by name.
        
        Args:
            instance_name: Optional name of a specific instance. If provided, returns stats
                           for only that instance. If None, returns stats for all instances.
        
        Returns:
            If instance_name is None: List of instance statistics dictionaries
            If instance_name is provided: Dictionary with statistics for the specified instance,
                                         or None if the instance doesn't exist
            If error occurs: Dictionary with error status and message
        """
        try:
            if instance_name is None:
                # Return stats for all instances
                stats = self.monitor.get_instance_stats(self.instances)
                return {
                    "status": "success",
                    "instances": stats,
                    "count": len(stats) if stats else 0
                }
            
            # Get stats for a specific instance
            if instance_name not in self.instances:
                return {
                    "status": "error",
                    "message": f"Instance '{instance_name}' not found"
                }
                
            # Create a dictionary with just this instance
            single_instance = {instance_name: self.instances[instance_name]}
            
            # Get stats for this instance
            all_stats = self.monitor.get_instance_stats(single_instance)
            
            # Return the first (and only) item in the list, or None if empty
            if all_stats:
                return {
                    "status": "success",
                    "instance": all_stats[0],
                    "name": instance_name
                }
            return {
                "status": "error",
                "message": f"No stats available for instance '{instance_name}'"
            }
        except Exception as e:
            error_msg = f"Error getting instance stats: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status summary for all instances.
        
        Returns:
            Dictionary with status and health information or error message
        """
        try:
            health_status = self.monitor.get_health_status(self.instances)
            return {
                "status": "success",
                "health_status": health_status
            }
        except Exception as e:
            error_msg = f"Error getting health status: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}
    
    def get_service_metrics(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Get overall service performance metrics for a specific time window.
        
        Args:
            window_minutes: Time window in minutes for calculations, default uses the average 
                           of all instances' rpm_window_minutes
            
        Returns:
            Dictionary with status and service metrics or error message
        """
        try:
            metrics = self.monitor.get_service_metrics(self.instances, window_minutes)
            return {
                "status": "success",
                "metrics": metrics
            }
        except Exception as e:
            error_msg = f"Error getting service metrics: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}
    
    def get_metrics_for_multiple_windows(self, windows: List[int] = None) -> Dict[str, Any]:
        """
        Get service metrics for multiple time windows.
        
        Args:
            windows: List of time windows in minutes to calculate metrics for
            
        Returns:
            Dictionary with status and metrics for each window or error message
        """
        try:
            # Use windows from config hierarchy if not provided
            if windows is None:
                config = config_hierarchy.get_configuration()
                stats_window = config.get('monitoring', {}).get('stats_window_minutes', 5)
                additional_windows = config.get('monitoring', {}).get('additional_windows', [15, 60])
                windows = [stats_window] + additional_windows
            
            metrics = self.monitor.get_multiple_window_metrics(self.instances, windows)
            return {
                "status": "success",
                "windows": metrics
            }
        except Exception as e:
            error_msg = f"Error getting metrics for multiple windows: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"status": "error", "message": error_msg}
    
    def set_rpm_window_for_all(self, minutes: int) -> None:
        """
        Set the RPM calculation window for all instances.
        
        Args:
            minutes: The number of minutes to use for the RPM calculation window
        """
        if minutes <= 0:
            raise ValueError("RPM window must be positive")
            
        for instance in self.instances.values():
            instance.set_rpm_window(minutes)
            
        logger.info(f"Set RPM window for all instances to {minutes} minutes")


# Create a singleton instance with the default routing strategy from config
instance_manager = InstanceManager() 