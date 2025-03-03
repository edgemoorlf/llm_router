"""Utilities for mapping OpenAI model names to Azure OpenAI deployments."""
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ModelMapper:
    """Maps OpenAI model names to Azure OpenAI deployments."""

    def __init__(self):
        """Initialize the model mapper by loading mappings from environment variables."""
        self.model_map: Dict[str, str] = {}
        self._load_mappings_from_env()

    def _load_mappings_from_env(self) -> None:
        """
        Load model mappings from environment variables.
        
        Environment variables should be in the format:
        MODEL_MAP_<OPENAI_MODEL_NAME>=<AZURE_DEPLOYMENT_NAME>
        
        For example:
        MODEL_MAP_GPT4o=my-gpt4o-deployment
        MODEL_MAP_GPT35TURBO=my-gpt35turbo-deployment
        """
        for key, value in os.environ.items():
            if key.startswith("MODEL_MAP_"):
                openai_model = key[len("MODEL_MAP_"):].lower().replace("_", "-")
                azure_deployment = value
                self.model_map[openai_model] = azure_deployment
                logger.info(f"Mapped model '{openai_model}' to Azure deployment '{azure_deployment}'")

    def get_deployment_name(self, model_name: str) -> Optional[str]:
        """
        Get the Azure deployment name for the given OpenAI model name.
        
        Args:
            model_name: The OpenAI model name, e.g., 'gpt-4o' or 'gpt-3.5-turbo'
            
        Returns:
            The Azure deployment name or None if no mapping exists.
        """
        # Special case for DeepSeek R1 model
        if model_name.lower() == "deepseek-r1":
            return "DeepSeek-R1"  # This is a special case handled differently in the routing
        
        # Normalize the model name by removing the version suffix if present
        normalized_name = model_name.split(':')[0]
        
        # Check if we have a direct mapping
        if normalized_name in self.model_map:
            return self.model_map[normalized_name]
        
        # Convert common model names to our mapped format
        mapping_key = normalized_name.lower().replace(".", "").replace("-", "")
        
        # Check common model patterns
        if mapping_key == "gpt4o":
            return self.model_map.get("gpt4o")
        elif mapping_key in ["gpt35turbo", "gpt3turbo"]:
            return self.model_map.get("gpt35turbo")
            
        logger.warning(f"No Azure deployment mapped for model '{model_name}'")
        return None

# Create a singleton instance
model_mapper = ModelMapper()
