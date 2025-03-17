"""
Initialize the application to use the new implementation.

This script sets up environment variables to use the new implementation and
starts the application with the new instance management system.
"""

import os
import sys
import shutil
import logging
import argparse

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("initialize")

def initialize_new_implementation(
    config_file: str = "instance_configs.json",
    state_file: str = "instance_states.json",
    env_file: str = ".env"
) -> bool:
    """
    Initialize the application to use the new implementation.
    
    Args:
        config_file: Path to the configuration file
        state_file: Path to the state file
        env_file: Path to the environment file to create/modify
        
    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        # Check if files exist
        if not os.path.exists(config_file):
            logger.error(f"Configuration file {config_file} not found. Run the migration script first.")
            return False
            
        if not os.path.exists(state_file):
            logger.error(f"State file {state_file} not found. Run the migration script first.")
            return False
        
        # Create full paths
        config_file_full = os.path.abspath(config_file)
        state_file_full = os.path.abspath(state_file)
        
        # Create/modify .env file
        env_variables = {
            "USE_NEW_IMPLEMENTATION": "true",
            "INSTANCE_CONFIG_FILE": config_file_full,
            "INSTANCE_STATE_FILE": state_file_full
        }
        
        # Check if .env exists
        env_path = os.path.join("app", env_file)
        if not os.path.exists(os.path.dirname(env_path)):
            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            
        # Create backup if .env exists
        if os.path.exists(env_path):
            backup_path = f"{env_path}.backup"
            logger.info(f"Creating backup of {env_path} as {backup_path}")
            shutil.copy2(env_path, backup_path)
            
            # Read existing variables
            existing_vars = {}
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        existing_vars[key.strip()] = value.strip()
            
            # Update with new variables
            for key, value in env_variables.items():
                existing_vars[key] = value
                
            # Write back to .env
            with open(env_path, "w") as f:
                for key, value in existing_vars.items():
                    f.write(f"{key}={value}\n")
        else:
            # Create new .env file
            with open(env_path, "w") as f:
                for key, value in env_variables.items():
                    f.write(f"{key}={value}\n")
        
        logger.info(f"Updated {env_path} with new implementation settings")
        
        # Create initialization script
        init_script = """
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.instance.factory import InstanceFactory

# Initialize factory with new implementation
InstanceFactory.initialize(
    use_new_implementation=True,
    config_file_new=os.environ.get("INSTANCE_CONFIG_FILE", "instance_configs.json"),
    state_file_new=os.environ.get("INSTANCE_STATE_FILE", "instance_states.json")
)

print("Initialized InstanceFactory with new implementation")
"""
        
        # Write initialization script
        init_path = os.path.join("migration", "init_factory.py")
        with open(init_path, "w") as f:
            f.write(init_script)
            
        logger.info(f"Created initialization script at {init_path}")
        logger.info("To use the new implementation:")
        logger.info(f"1. Make sure the environment variables in {env_path} are set")
        logger.info(f"2. Import and run the initialization script before starting your application")
        logger.info(f"3. Or use the API endpoint to switch implementations at runtime")
        
        return True
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize the application to use the new implementation")
    parser.add_argument("--config-file", default="instance_configs.json", help="Path to the configuration file")
    parser.add_argument("--state-file", default="instance_states.json", help="Path to the state file")
    parser.add_argument("--env-file", default=".env", help="Path to the environment file to create/modify")
    args = parser.parse_args()
    
    success = initialize_new_implementation(args.config_file, args.state_file, args.env_file)
    
    if success:
        logger.info("Initialization completed successfully!")
    else:
        logger.error("Initialization failed.")
        sys.exit(1) 