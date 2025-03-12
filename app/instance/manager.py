"""Instance manager for multiple OpenAI-compatible service instances."""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import threading

logger = logging.getLogger(__name__)

from .routing_strategy import RoutingStrategy
from .api_instance import APIInstance
from .config_loader import InstanceConfigLoader
from .router import InstanceRouter
from .forwarder import RequestForwarder
from .monitor import InstanceMonitor

# Import the new configuration system
from app.config import config_loader, InstanceConfig

# Add new imports for file-based state management
import json
import tempfile
import os.path
import time
import uuid

class InstanceManager:
    """Manager for multiple API service instances with load balancing and failover."""
    
    # Add configuration for shared state file
    SHARED_STATE_FILE = os.environ.get('INSTANCE_MANAGER_STATE_FILE', 
                                      os.path.join(tempfile.gettempdir(), 'azure_openai_proxy_instances.json'))
    STATE_FILE_POLL_INTERVAL = 5  # seconds
    
    def __init__(self, routing_strategy: Optional[RoutingStrategy] = None):
        """
        Initialize the instance manager.
        
        Args:
            routing_strategy: Strategy for selecting instances
        """
        self.instances: Dict[str, APIInstance] = {}
        self.instances_lock = threading.RLock()  # Lock for thread-safe access to instances
        self.router = None  # Will be initialized later
        self.forwarder = RequestForwarder()
        self.config_loader = InstanceConfigLoader()
        self.monitor = InstanceMonitor()
        
        # Add worker ID for state file operations
        self.worker_id = str(uuid.uuid4())
        self.last_state_check = 0
        
        try:
            # Load configuration (may raise an exception)
            config = config_loader.load_config()
            
            # Use routing strategy from config if not provided
            if routing_strategy is None:
                routing_strategy = RoutingStrategy(config.routing.strategy)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Fall back to default routing strategy
            if routing_strategy is None:
                routing_strategy = RoutingStrategy.FAILOVER
            config = None
            
        # Initialize router now that we have the routing strategy
        self.router = InstanceRouter(routing_strategy)
        
        # First try to load state from shared file
        if self._load_state_from_file():
            logger.info(f"Loaded instance state from shared file: {len(self.instances)} instances")
        else:
            # Load instances from configuration
            try:
                if config and config.instances:
                    self._load_instances_from_config(config.instances)
                else:
                    logger.warning("No valid configuration found or no instances defined in config")
            except Exception as e:
                logger.error(f"Error loading instances from config: {str(e)}")
            
            # If no instances found, try loading from environment variables (backward compatibility)
            if not self.instances:
                logger.warning("No API instances found in configuration. Falling back to environment variables.")
                self.instances = self.config_loader.load_from_env()
                
                # If still no instances, try legacy configuration
                if not self.instances:
                    logger.warning("No API instances configured. Falling back to legacy single instance configuration.")
                    self.instances = self.config_loader.load_legacy_instance()
            
            # Set RPM window for all instances
            try:
                if config and config.monitoring and config.monitoring.stats_window_minutes > 0:
                    self.set_rpm_window_for_all(config.monitoring.stats_window_minutes)
            except Exception as e:
                logger.error(f"Error setting RPM window: {str(e)}")
            
            # Save initial state to file
            self._save_state_to_file()
        
        logger.info(f"Initialized {len(self.instances)} API instances with {routing_strategy} routing strategy")
    
    def _load_instances_from_config(self, instance_configs: List[InstanceConfig]) -> None:
        """
        Load instances from configuration objects.
        
        Args:
            instance_configs: List of instance configurations
        """
        for config in instance_configs:
            try:
                instance = APIInstance(
                    name=config.name,
                    provider_type=config.provider_type,
                    api_key=config.api_key,
                    api_base=config.api_base,
                    api_version=config.api_version,
                    proxy_url=config.proxy_url,
                    priority=config.priority,
                    weight=config.weight,
                    max_tpm=config.max_tpm,
                    max_input_tokens=config.max_input_tokens,
                    supported_models=config.supported_models,
                    model_deployments=config.model_deployments
                )
                instance.initialize_client()
                self.instances[config.name] = instance
                logger.info(f"Loaded API instance {config.name} from configuration")
            except Exception as e:
                logger.error(f"Error loading instance {config.name} from configuration: {str(e)}")
    
    def load_from_csv(self, csv_path: str) -> None:
        """
        Load instances from a CSV file, replacing existing configuration.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.instances.clear()
        instances = self.config_loader.load_from_csv(csv_path)
        self.instances.update(instances)
        
        if not self.instances:
            logger.warning("No instances loaded from CSV. Falling back to environment variables.")
            self.instances = self.config_loader.load_from_env()
            
        logger.info(f"Loaded {len(self.instances)} instances from CSV")
    
    def reload_config(self) -> None:
        """Reload configuration from disk."""
        try:
            # Reload configuration
            config = config_loader.reload()
            if not config:
                logger.error("Failed to reload configuration")
                return
                
            # Clear existing instances
            self.instances.clear()
            
            # Load instances from configuration
            if config.instances:
                self._load_instances_from_config(config.instances)
            else:
                logger.warning("No instances defined in reloaded configuration")
            
            # If no instances found, try loading from environment variables
            if not self.instances:
                logger.warning("No API instances found in configuration. Falling back to environment variables.")
                self.instances = self.config_loader.load_from_env()
            
            # Update routing strategy
            try:
                if config.routing and config.routing.strategy:
                    self.router.strategy = RoutingStrategy(config.routing.strategy)
            except Exception as e:
                logger.error(f"Error updating routing strategy: {str(e)}")
            
            # Set RPM window for all instances
            try:
                if config.monitoring and config.monitoring.stats_window_minutes > 0:
                    self.set_rpm_window_for_all(config.monitoring.stats_window_minutes)
            except Exception as e:
                logger.error(f"Error setting RPM window: {str(e)}")
            
            # Save state after reloading config
            self._save_state_to_file()
            
            logger.info(f"Reloaded configuration with {len(self.instances)} instances")
        except Exception as e:
            logger.error(f"Error reloading configuration: {str(e)}")
            raise
    
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
        return await self.forwarder.try_instances(
            self.instances,
            endpoint,
            deployment,
            payload,
            required_tokens,
            self.router,
            method,
            provider_type
        )
    
    def get_instance_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all instances.
        
        Returns:
            List of instance statistics dictionaries
        """
        return self.monitor.get_instance_stats(self.instances)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status summary for all instances.
        
        Returns:
            Summary of instance health status
        """
        return self.monitor.get_health_status(self.instances)
    
    def get_service_metrics(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Get overall service performance metrics for a specific time window.
        
        Args:
            window_minutes: Time window in minutes for calculations, default uses the average 
                           of all instances' rpm_window_minutes
            
        Returns:
            Overall service metrics for the specified time window
        """
        return self.monitor.get_service_metrics(self.instances, window_minutes)
    
    def get_metrics_for_multiple_windows(self, windows: List[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get service metrics for multiple time windows.
        
        Args:
            windows: List of time windows in minutes to calculate metrics for
            
        Returns:
            Dictionary of service metrics for each time window
        """
        # Use windows from config if not provided
        if windows is None:
            config = config_loader.get_config()
            windows = [config.monitoring.stats_window_minutes] + config.monitoring.additional_windows
        
        return self.monitor.get_multiple_window_metrics(self.instances, windows)
    
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

    def _load_state_from_file(self) -> bool:
        """
        Load instance state from shared file.
        
        Returns:
            True if state was loaded successfully, False otherwise
        """
        if not os.path.exists(self.SHARED_STATE_FILE):
            return False
            
        try:
            # Check file modification time to avoid unnecessary reads
            mtime = os.path.getmtime(self.SHARED_STATE_FILE)
            if mtime <= self.last_state_check and self.instances:
                return True
                
            self.last_state_check = time.time()
            
            with open(self.SHARED_STATE_FILE, 'r') as f:
                state = json.load(f)
                
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
                        logger.error(f"Error loading instance {instance_data.get('name', 'unknown')} from state file: {str(e)}")
                
            return True
        except Exception as e:
            logger.error(f"Error loading state from file: {str(e)}")
            return False
    
    def _save_state_to_file(self) -> bool:
        """
        Save instance state to shared file.
        
        Returns:
            True if state was saved successfully, False otherwise
        """
        try:
            state = {
                'timestamp': time.time(),
                'worker_id': self.worker_id,
                'instances': []
            }
            
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
                    state['instances'].append(instance_data)
            
            # Write to temporary file first, then rename for atomicity
            temp_file = f"{self.SHARED_STATE_FILE}.{self.worker_id}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(state, f)
                
            # Atomic rename
            os.replace(temp_file, self.SHARED_STATE_FILE)
            
            self.last_state_check = time.time()
            return True
        except Exception as e:
            logger.error(f"Error saving state to file: {str(e)}")
            return False
    
    def check_for_updates(self) -> bool:
        """
        Check for updates from other workers.
        
        Returns:
            True if updates were loaded, False otherwise
        """
        # Only check every STATE_FILE_POLL_INTERVAL seconds
        if time.time() - self.last_state_check < self.STATE_FILE_POLL_INTERVAL:
            return False
            
        return self._load_state_from_file()

    # Add state saving to all methods that modify instances
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
            self._save_state_to_file()
            
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
                self._save_state_to_file()
                return True
            return False

# Create a singleton instance with the default routing strategy from config
instance_manager = InstanceManager() 