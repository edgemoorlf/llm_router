"""
Instance Context Module

This module initializes and provides access to the instance manager and router.
It serves as a central point for accessing these components throughout the application,
eliminating circular dependencies between modules.
"""

import os
import logging
from app.instance.new_manager import NewInstanceManager
from app.instance.new_router import InstanceRouter

# Configure logger
logger = logging.getLogger(__name__)

# Get configuration from environment variables
config_file_path = os.environ.get("INSTANCE_CONFIG_FILE", "instance_configs.json")
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

# Initialize the instance manager
instance_manager = NewInstanceManager(
    config_file=config_file_path,
    redis_url=redis_url
)

# Initialize the router
instance_router = InstanceRouter(instance_manager)

# Log initialization
logger.info(f"Initialized instance manager with config file: {config_file_path}")
logger.info(f"Using Redis at: {redis_url}")
logger.info(f"Instance manager loaded {len(instance_manager.get_instance_stats())} instances")

def check_for_updates():
    """
    Check for updates to instance configurations.
    This function should be called periodically to ensure the instance manager 
    has the latest instance data.
    """
    try:
        instance_manager.reload_config()
    except Exception as e:
        logger.error(f"Error checking for instance updates: {e}") 