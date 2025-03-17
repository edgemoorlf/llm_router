#!/usr/bin/env python3
"""
Deployment script for the new implementation.

This script helps set up the new implementation in a production environment by:
1. Backing up existing files
2. Migrating to the new implementation if needed
3. Creating necessary directories
4. Setting up appropriate permissions
5. Installing monitoring and backup jobs
"""

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)

logger = logging.getLogger('deployment')

def backup_existing_files(source_dir: str, backup_dir: str) -> bool:
    """
    Back up existing configuration and instance files.
    
    Args:
        source_dir: Directory containing files to back up
        backup_dir: Directory to store backups
        
    Returns:
        True if backup was successful, False otherwise
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join(backup_dir, f"backup_{timestamp}")
        
        # Create backup directory
        os.makedirs(target_dir, exist_ok=True)
        
        # List of files to back up
        files_to_backup = [
            "instances.json",  # Old format
            "instance_configs.json",  # New config
            "instance_states.json",   # New state
            ".env"  # Environment variables
        ]
        
        # Back up each file if it exists
        for filename in files_to_backup:
            source_path = os.path.join(source_dir, filename)
            if os.path.exists(source_path):
                shutil.copy2(source_path, os.path.join(target_dir, filename))
                logger.info(f"Backed up {filename} to {target_dir}")
        
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False

def migrate_instances(api_url: str, admin_key: str) -> Tuple[bool, str]:
    """
    Migrate instances from old to new implementation.
    
    Args:
        api_url: Base URL for the API
        admin_key: Admin API key
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Check current implementation
        url = f"{api_url}/admin/instances/implementation"
        headers = {"X-Admin-API-Key": admin_key}
        
        response = requests.get(url, headers=headers)
        if not response.ok:
            return False, f"Failed to check implementation status: {response.text}"
        
        status = response.json()
        
        # If already using new implementation, return success
        if status.get("using_new_implementation", False):
            return True, "Already using the new implementation"
        
        # Migrate to new implementation
        url = f"{api_url}/admin/instances/migrate"
        response = requests.post(url, headers=headers)
        
        if not response.ok:
            return False, f"Migration failed: {response.text}"
        
        result = response.json()
        return True, f"Migration completed: {result.get('message', 'Success')}"
    except Exception as e:
        return False, f"Migration error: {e}"

def setup_environment(env_file: str, config_file: str, state_file: str) -> bool:
    """
    Set up environment variables for the new implementation.
    
    Args:
        env_file: Path to the .env file
        config_file: Path to the configuration file
        state_file: Path to the state file
        
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Read existing .env file if it exists
        env_vars = {}
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip()
        
        # Update with new implementation variables
        env_vars["USE_NEW_IMPLEMENTATION"] = "true"
        env_vars["INSTANCE_CONFIG_FILE"] = os.path.abspath(config_file)
        env_vars["INSTANCE_STATE_FILE"] = os.path.abspath(state_file)
        
        # Write back to .env file
        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Updated environment variables in {env_file}")
        return True
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False

def setup_monitoring(config_file: str, state_file: str) -> bool:
    """
    Set up monitoring and backup jobs.
    
    Args:
        config_file: Path to the configuration file
        state_file: Path to the state file
        
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Create backup directory
        backup_dir = "./backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create cron job file
        cron_file = "instance_monitoring.cron"
        with open(cron_file, "w") as f:
            # Daily backup at midnight
            f.write(f"0 0 * * * cd {os.getcwd()} && python scripts/backup_monitor.py backup --config-file {config_file} --state-file {state_file} --backup-dir {backup_dir}\n")
            
            # Monitor state file every hour
            f.write(f"0 * * * * cd {os.getcwd()} && python scripts/backup_monitor.py monitor --file {state_file} --interval 3600 --alert-after 7200 >> monitor.log 2>&1\n")
        
        # Display instructions for installing cron job
        logger.info(f"Created cron job file: {cron_file}")
        logger.info("To install the cron job, run:")
        logger.info(f"  crontab {cron_file}")
        
        return True
    except Exception as e:
        logger.error(f"Monitoring setup failed: {e}")
        return False

def verify_deployment(api_url: str, admin_key: str) -> Tuple[bool, str]:
    """
    Verify that the deployment was successful.
    
    Args:
        api_url: Base URL for the API
        admin_key: Admin API key
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Check implementation status
        url = f"{api_url}/admin/instances/implementation"
        headers = {"X-Admin-API-Key": admin_key}
        
        response = requests.get(url, headers=headers)
        if not response.ok:
            return False, f"Failed to check implementation status: {response.text}"
        
        status = response.json()
        
        # Verify using new implementation
        if not status.get("using_new_implementation", False):
            return False, "Not using the new implementation"
        
        # Get all instances
        url = f"{api_url}/admin/instances/new"
        response = requests.get(url, headers=headers)
        
        if not response.ok:
            return False, f"Failed to get instances: {response.text}"
        
        result = response.json()
        instance_count = result.get("count", 0)
        
        return True, f"Deployment verified: Using new implementation with {instance_count} instances"
    except Exception as e:
        return False, f"Verification error: {e}"

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy the new implementation")
    
    parser.add_argument("--api-url", default="http://localhost:3010", help="API base URL")
    parser.add_argument("--admin-key", required=True, help="Admin API key")
    parser.add_argument("--source-dir", default=".", help="Source directory")
    parser.add_argument("--backup-dir", default="./backups", help="Backup directory")
    parser.add_argument("--env-file", default=".env", help="Environment file")
    parser.add_argument("--config-file", default="instance_configs.json", help="Config file")
    parser.add_argument("--state-file", default="instance_states.json", help="State file")
    parser.add_argument("--migrate", action="store_true", help="Migrate instances")
    parser.add_argument("--setup-env", action="store_true", help="Set up environment")
    parser.add_argument("--setup-monitoring", action="store_true", help="Set up monitoring")
    parser.add_argument("--verify", action="store_true", help="Verify deployment")
    parser.add_argument("--all", action="store_true", help="Perform all deployment steps")
    
    args = parser.parse_args()
    
    # If --all is specified, set all individual steps
    if args.all:
        args.migrate = True
        args.setup_env = True
        args.setup_monitoring = True
        args.verify = True
    
    # Create absolute paths
    args.source_dir = os.path.abspath(args.source_dir)
    args.backup_dir = os.path.abspath(args.backup_dir)
    args.env_file = os.path.abspath(os.path.join(args.source_dir, args.env_file))
    args.config_file = os.path.abspath(os.path.join(args.source_dir, args.config_file))
    args.state_file = os.path.abspath(os.path.join(args.source_dir, args.state_file))
    
    logger.info("Starting deployment")
    
    # Back up existing files
    logger.info("Backing up existing files")
    if backup_existing_files(args.source_dir, args.backup_dir):
        logger.info("Backup completed successfully")
    else:
        logger.error("Backup failed")
        return 1
    
    # Migrate instances if requested
    if args.migrate:
        logger.info("Migrating instances")
        success, message = migrate_instances(args.api_url, args.admin_key)
        logger.info(message)
        if not success:
            logger.error("Migration failed")
            return 1
    
    # Set up environment if requested
    if args.setup_env:
        logger.info("Setting up environment")
        if setup_environment(args.env_file, args.config_file, args.state_file):
            logger.info("Environment setup completed successfully")
        else:
            logger.error("Environment setup failed")
            return 1
    
    # Set up monitoring if requested
    if args.setup_monitoring:
        logger.info("Setting up monitoring")
        if setup_monitoring(args.config_file, args.state_file):
            logger.info("Monitoring setup completed successfully")
        else:
            logger.error("Monitoring setup failed")
            return 1
    
    # Verify deployment if requested
    if args.verify:
        logger.info("Verifying deployment")
        success, message = verify_deployment(args.api_url, args.admin_key)
        logger.info(message)
        if not success:
            logger.error("Verification failed")
            return 1
    
    logger.info("Deployment completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 