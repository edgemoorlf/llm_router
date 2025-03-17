"""
Migration script to separate instance configuration from state.

This script extracts configuration and state from the existing instance data
and stores them in separate files according to the new architecture.
"""

import os
import json
import time
import logging
import sys
from typing import Dict, Any

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.instance import InstanceConfig, InstanceState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migration")

# Define which fields belong to config vs state
CONFIG_FIELDS = {
    "name", "provider_type", "api_key", "api_base", "api_version", 
    "proxy_url", "priority", "weight", "max_tpm", "max_input_tokens",
    "supported_models", "model_deployments", "enabled", "timeout_seconds", 
    "retry_count"
}

STATE_FIELDS = {
    "name", "status", "error_count", "last_error", "rate_limited_until",
    "current_tpm", "current_rpm", "total_requests", "successful_requests",
    "last_used", "last_error_time", "avg_latency_ms", "utilization_percentage",
    "connection_status", "health_status"
}

def migrate_instance_data(
    input_file: str = "instance_data.json",
    config_file: str = "instance_configs.json",
    state_file: str = "instance_states.json"
) -> bool:
    """
    Migrate instance data from the old format to separated config/state.
    
    Args:
        input_file: Path to the old instance data file
        config_file: Path to write configurations to
        state_file: Path to write states to
        
    Returns:
        True if the migration was successful, False otherwise
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        logger.warning(f"Input file {input_file} not found. Nothing to migrate.")
        return False
    
    try:
        # Load existing instance data
        logger.info(f"Loading instance data from {input_file}")
        with open(input_file, "r") as f:
            old_data = json.load(f)
        
        # Extract the array of instances if it's in the new format
        instances_to_process = {}
        if isinstance(old_data, dict) and "instances" in old_data and isinstance(old_data["instances"], list):
            # This is the format where instances are in an array under "instances" key
            logger.info(f"Detected nested format with instances array")
            for instance in old_data["instances"]:
                if "name" in instance:
                    name = instance["name"]
                    instances_to_process[name] = instance
        else:
            # Assume it's the old flat format
            instances_to_process = old_data
        
        # Create containers for new data
        config_data = {}
        state_data = {}
        
        # Process each instance
        for name, instance in instances_to_process.items():
            logger.info(f"Processing instance '{name}'")
            
            # Extract config properties
            config_dict = {"name": name}
            for prop in CONFIG_FIELDS:
                if prop in instance and prop != "name":
                    config_dict[prop] = instance[prop]
            
            # Extract state properties (some may be in stats sub-object)
            state_dict = {"name": name}
            for prop in STATE_FIELDS:
                if prop in instance and prop != "name":
                    state_dict[prop] = instance[prop]
                # Check if it's in the stats object
                elif "stats" in instance and prop in instance["stats"]:
                    state_dict[prop] = instance["stats"][prop]
            
            # Validate and create models
            try:
                config = InstanceConfig(**config_dict)
                config_data[name] = config.dict()
                logger.debug(f"Created config for {name}")
            except Exception as e:
                logger.error(f"Error creating config for {name}: {e}")
                continue
            
            try:
                state = InstanceState(**state_dict)
                state_data[name] = state.dict()
                logger.debug(f"Created state for {name}")
            except Exception as e:
                logger.warning(f"Error creating state for {name}: {e}")
                # Continue anyway - state can be rebuilt
        
        # Create backup of the old data
        backup_name = f"{input_file}.backup_{int(time.time())}"
        logger.info(f"Creating backup of original data at {backup_name}")
        with open(backup_name, "w") as f:
            json.dump(old_data, f, indent=2)
        
        # Save the new data
        logger.info(f"Saving {len(config_data)} configurations to {config_file}")
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saving {len(state_data)} states to {state_file}")
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"Migration complete. Processed {len(instances_to_process)} instances.")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

if __name__ == "__main__":
    # Create the migration directory
    os.makedirs("migration", exist_ok=True)
    
    # Default file paths relative to the workspace
    input_file = "instance_data.json"
    config_file = "instance_configs.json"
    state_file = "instance_states.json"
    
    # Allow overriding file paths via command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        config_file = sys.argv[2]
    if len(sys.argv) > 3:
        state_file = sys.argv[3]
    
    success = migrate_instance_data(input_file, config_file, state_file)
    
    if success:
        logger.info("Migration completed successfully!")
    else:
        logger.error("Migration failed.")
        sys.exit(1) 