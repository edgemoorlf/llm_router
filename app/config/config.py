import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, HttpUrl, SecretStr, root_validator, validator
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class InstanceConfig(BaseModel):
    """Configuration for an API instance."""
    name: str = Field(..., description="Unique identifier for this instance")
    provider_type: str = Field(default="azure", description="Provider type (azure or generic)")
    api_key: str = Field(..., description="API key for the service")
    api_base: str = Field(..., description="API base URL")
    api_version: str = Field(default="2024-08-01-preview", description="API version")
    proxy_url: Optional[str] = Field(
        default=None,
        description="Proxy URL for HTTP requests (e.g. http://user:pass@host:port)",
    )
    priority: int = Field(default=100, description="Priority (lower is higher priority)")
    weight: int = Field(default=100, description="Weight for weighted distribution (higher gets more traffic)")
    max_tpm: int = Field(default=240000, description="Maximum TPM (tokens per minute) for this instance")
    max_input_tokens: int = Field(default=0, description="Maximum input tokens allowed (0=unlimited)")
    supported_models: List[str] = Field(default_factory=list, description="List of models supported by this instance")
    model_deployments: Dict[str, str] = Field(default_factory=dict, description="Mapping of model names to deployment names (for Azure)")
    
    @validator('max_tpm')
    def validate_max_tpm(cls, v):
        if v <= 0:
            raise ValueError("max_tpm must be greater than 0")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        if v < 0:
            raise ValueError("priority must be non-negative")
        return v

class RoutingConfig(BaseModel):
    """Configuration for request routing."""
    strategy: str = Field(default="failover", description="Routing strategy (failover, weighted, round_robin)")
    retries: int = Field(default=3, description="Max retries for failed requests")
    timeout: int = Field(default=60, description="Default timeout in seconds")

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Log level")
    file: str = Field(default="../logs/app.log", description="Log file path")
    max_size: int = Field(default=5242880, description="Max log file size (5MB)")
    backup_count: int = Field(default=3, description="Number of backup log files")
    feishu_webhook: Optional[str] = Field(default=None, description="Feishu webhook URL for alerts")

class MonitoringConfig(BaseModel):
    """Configuration for monitoring."""
    stats_window_minutes: int = Field(default=5, description="Default time window for statistics in minutes")
    additional_windows: List[int] = Field(default=[15, 30, 60], description="Additional time windows for statistics")

class AppConfig(BaseModel):
    """Main application configuration."""
    name: str = Field(default="Azure OpenAI Proxy", description="Application name")
    version: str = Field(default="1.0.3", description="Application version")
    port: int = Field(default=3010, description="Server port")
    instances: List[InstanceConfig] = Field(default_factory=list, description="API instances")
    routing: RoutingConfig = Field(default_factory=RoutingConfig, description="Routing configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")

class ConfigLoader:
    """Configuration loader with YAML support and environment variable fallback."""
    
    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.config_path: Optional[str] = None
        self.env: str = os.getenv("ENVIRONMENT", "production")
        
    def load_config(self, config_dir: Optional[str] = None) -> AppConfig:
        """
        Load configuration from YAML files with environment variable fallback.
        
        Args:
            config_dir: Directory containing config files, defaults to app/config
            
        Returns:
            Loaded configuration
        """
        # Default config directory to app/config if not specified
        if config_dir is None:
            # Find config directory relative to this file
            current_dir = Path(__file__).parent
            config_dir = str(current_dir)
        
        self.config_path = config_dir
        logger.info(f"Loading configuration from {config_dir}")
        
        # Try to load from YAML first
        config_dict = self._load_from_yaml(config_dir)
        
        # Ensure config_dict is at least an empty dict if YAML loading failed
        if config_dict is None:
            config_dict = {}
            logger.warning("Failed to load YAML configuration, starting with empty configuration")
        
        # If no instances in YAML config, fallback to env vars
        if not config_dict.get("instances"):
            logger.info("No instances found in YAML configuration, falling back to environment variables")
            instances = self._load_instances_from_env()
            config_dict["instances"] = []
            if instances:
                config_dict["instances"].extend(instances)
        
        try:
            self.config = AppConfig.parse_obj(config_dict)
            logger.info(f"Loaded configuration with {len(self.config.instances)} instances")
            return self.config
        except Exception as e:
            logger.error(f"Error parsing configuration: {str(e)}")
            # Provide a minimal default configuration
            self.config = AppConfig()
            return self.config
    
    def _load_from_yaml(self, config_dir: str) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config_dict = {}
        
        # Try to load base.yaml
        base_path = os.path.join(config_dir, "base.yaml")
        if os.path.exists(base_path):
            try:
                with open(base_path, "r") as f:
                    base_config = yaml.safe_load(f)
                    if base_config:
                        config_dict.update(base_config)
                logger.info(f"Loaded base configuration from {base_path}")
            except Exception as e:
                logger.error(f"Error loading base configuration: {str(e)}")
        else:
            logger.warning(f"Base configuration file not found at {base_path}")
        
        # Try to load environment-specific YAML (e.g. production.yaml)
        env_path = os.path.join(config_dir, f"{self.env}.yaml")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r") as f:
                    env_config = yaml.safe_load(f)
                    if env_config:
                        # Deep merge with base config
                        config_dict = self._deep_merge(config_dict, env_config)
                logger.info(f"Loaded environment configuration from {env_path}")
            except Exception as e:
                logger.error(f"Error loading environment configuration: {str(e)}")
        else:
            logger.info(f"Environment configuration file not found at {env_path}")
        
        # Handle environment variable placeholders in YAML
        config_dict = self._resolve_env_vars(config_dict)
        
        return config_dict
    
    def _resolve_env_vars(self, config: Any) -> Any:
        """Resolve environment variable placeholders in configuration values."""
        if isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            # Extract environment variable name
            env_var = config[2:-1]
            return os.getenv(env_var, "")
        elif isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        else:
            return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
                
        return result
    
    def _load_instances_from_env(self) -> List[Dict[str, Any]]:
        """Load instance configurations from environment variables."""
        instances = []
        
        # First try the multi-instance format
        instance_names = os.getenv("API_INSTANCES", "").split(",")
        instance_names = [name.strip() for name in instance_names if name.strip()]
        
        if not instance_names:
            logger.info("No API_INSTANCES defined in environment variables")
        
        for name in instance_names:
            prefix = f"API_INSTANCE_{name.upper()}_"
            
            if not os.getenv(f"{prefix}API_KEY") or not os.getenv(f"{prefix}API_BASE"):
                logger.warning(f"Skipping incomplete instance configuration for {name}")
                continue
                
            instance = {
                "name": name,
                "provider_type": os.getenv(f"{prefix}PROVIDER_TYPE", "azure").lower(),
                "api_key": os.getenv(f"{prefix}API_KEY", ""),
                "api_base": os.getenv(f"{prefix}API_BASE", ""),
                "api_version": os.getenv(f"{prefix}API_VERSION", "2024-08-01-preview"),
                "proxy_url": os.getenv(f"{prefix}PROXY_URL"),
                "priority": int(os.getenv(f"{prefix}PRIORITY", "100")),
                "weight": int(os.getenv(f"{prefix}WEIGHT", "100")),
                "max_tpm": int(os.getenv(f"{prefix}MAX_TPM", "240000")),
                "max_input_tokens": int(os.getenv(f"{prefix}MAX_INPUT_TOKENS", "0")),
            }
            
            # Load supported models
            models_str = os.getenv(f"{prefix}SUPPORTED_MODELS", "")
            if models_str:
                instance["supported_models"] = [model.strip() for model in models_str.split(",") if model.strip()]
            
            # Load model to deployment mappings
            model_deployments = {}
            for key, value in os.environ.items():
                if key.startswith(f"{prefix}MODEL_MAP_"):
                    model_name = key[len(f"{prefix}MODEL_MAP_"):].lower().replace("_", "-")
                    model_deployments[model_name] = value
            
            if model_deployments:
                instance["model_deployments"] = model_deployments
            
            instances.append(instance)
            logger.info(f"Loaded instance {name} from environment variables")
        
        # If no instances found, try legacy single instance format
        if not instances:
            logger.info("No multi-instance configuration found, checking for legacy single instance format")
            api_key = os.getenv("API_KEY")
            api_base = os.getenv("API_BASE")
            
            if api_key and api_base:
                instance = {
                    "name": "default",
                    "provider_type": os.getenv("API_PROVIDER_TYPE", "azure").lower(),
                    "api_key": api_key,
                    "api_base": api_base,
                    "api_version": os.getenv("API_VERSION", "2024-08-01-preview"),
                    "priority": 1,
                    "weight": 100,
                    "max_tpm": 240000,
                }
                instances.append(instance)
                logger.info("Loaded legacy instance configuration from environment variables")
            else:
                logger.warning("No instance configuration found in environment variables")
        
        return instances
    
    def reload(self) -> AppConfig:
        """Reload configuration from disk."""
        if self.config_path:
            logger.info("Reloading configuration")
            return self.load_config(self.config_path)
        else:
            logger.warning("Cannot reload configuration, no config path set")
            return self.config
    
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        if not self.config:
            return self.load_config()
        return self.config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for API responses)."""
        if not self.config:
            logger.warning("No configuration loaded, returning empty dictionary")
            return {}
            
        try:
            # Convert to dict and redact sensitive fields
            config_dict = self.config.dict()
            
            # Redact API keys
            if "instances" in config_dict:
                for instance in config_dict["instances"]:
                    if "api_key" in instance:
                        instance["api_key"] = "********"
            
            return config_dict
        except Exception as e:
            logger.error(f"Error converting configuration to dictionary: {str(e)}")
            return {}
    
    def save_config(self, config: AppConfig) -> bool:
        """Save configuration to disk."""
        if not self.config_path:
            logger.error("Cannot save configuration, no config path set")
            return False
            
        try:
            # Ensure the config directory exists
            os.makedirs(self.config_path, exist_ok=True)
            
            # Save to environment-specific file
            config_path = os.path.join(self.config_path, f"{self.env}.yaml")
            
            # Convert to dict
            config_dict = config.dict(exclude_none=True)
            
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                
            logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False

# Create a singleton instance
config_loader = ConfigLoader() 