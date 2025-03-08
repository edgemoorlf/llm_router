import os
import csv
import logging
from typing import Dict

from .api_instance import APIInstance

logger = logging.getLogger(__name__)

class InstanceConfigLoader:
    """Responsible for loading API instances from different configuration sources."""
    
    @staticmethod
    def load_from_csv(csv_path: str) -> Dict[str, APIInstance]:
        """Load instances from a CSV file."""
        instances = {}
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
                            api_version=row.get('api_version', '2024-08-01-preview'),
                            proxy_url=row.get('proxy_url'),
                            priority=int(row.get('priority', 100)),
                            weight=int(row.get('weight', 100)),
                            max_tpm=int(row.get('max_tpm', 240000)),
                            max_input_tokens=int(row.get('max_input_tokens', 0)),
                            supported_models=[m.strip() for m in row.get('model_name', '').split(',') if m.strip()],
                        )
                        instance.initialize_client()
                        instances[name] = instance
                        logger.info(f"Loaded API instance {name} from CSV")
                    except KeyError as e:
                        logger.error(f"Missing required column {e} in CSV row {row_idx}")
                    except Exception as e:
                        logger.error(f"Error loading instance from CSV row {row_idx}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load instances from CSV: {str(e)}")
            raise
        
        return instances
    
    @staticmethod
    def load_from_env() -> Dict[str, APIInstance]:
        """Load instances from environment variables."""
        instances = {}
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
                    api_version=os.getenv(f"{prefix}API_VERSION", "2024-08-01-preview"),
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
                instances[name] = instance
                logger.info(f"Loaded API instance {name} with priority {instance.priority}, weight {instance.weight}, max TPM {instance.max_tpm}")
            except Exception as e:
                logger.error(f"Error loading instance {name}: {str(e)}")
        
        return instances
    
    @staticmethod
    def load_legacy_instance() -> Dict[str, APIInstance]:
        """Load the legacy single instance configuration."""
        instances = {}
        api_key = os.getenv("API_KEY")
        api_base = os.getenv("API_BASE")
        api_version = os.getenv("API_VERSION", "2023-07-01-preview")
        
        if not api_key or not api_base:
            logger.error("No API instances configured and legacy configuration is incomplete")
            return instances
        
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
        instances["default"] = instance
        logger.info("Loaded legacy API instance configuration")
        
        return instances 