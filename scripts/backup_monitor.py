#!/usr/bin/env python3
"""
Backup and monitoring script for instance configuration and state files.

This script provides functionality to:
1. Automatically back up configuration and state files
2. Monitor for changes to these files
3. Alert if the files haven't been updated in a certain time period
"""

import os
import sys
import time
import json
import shutil
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backup_monitor.log')
    ]
)

logger = logging.getLogger('backup_monitor')

def create_backup(source_file: str, backup_dir: str, max_backups: int = 10) -> Optional[str]:
    """
    Create a backup of the specified file.
    
    Args:
        source_file: Path to the file to back up
        backup_dir: Directory to store backups
        max_backups: Maximum number of backups to keep
        
    Returns:
        Path to the created backup file, or None if backup failed
    """
    try:
        # Ensure source file exists
        if not os.path.exists(source_file):
            logger.error(f"Source file does not exist: {source_file}")
            return None
        
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(source_file)
        backup_file = os.path.join(backup_dir, f"{filename}.{timestamp}")
        
        # Create the backup
        shutil.copy2(source_file, backup_file)
        logger.info(f"Created backup: {backup_file}")
        
        # Clean up old backups if needed
        cleanup_old_backups(backup_dir, filename, max_backups)
        
        return backup_file
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return None

def cleanup_old_backups(backup_dir: str, filename_prefix: str, max_backups: int) -> None:
    """
    Remove old backups to stay within the maximum limit.
    
    Args:
        backup_dir: Directory containing backups
        filename_prefix: Prefix of the backup files
        max_backups: Maximum number of backups to keep
    """
    try:
        # Get all backup files for this prefix
        backup_files = [f for f in os.listdir(backup_dir) 
                      if f.startswith(filename_prefix) and f != filename_prefix]
        
        # If we're within the limit, no need to clean up
        if len(backup_files) <= max_backups:
            return
        
        # Sort by modification time (oldest first)
        backup_files.sort(key=lambda f: os.path.getmtime(os.path.join(backup_dir, f)))
        
        # Remove oldest backups
        for old_file in backup_files[:len(backup_files) - max_backups]:
            old_path = os.path.join(backup_dir, old_file)
            os.remove(old_path)
            logger.info(f"Removed old backup: {old_path}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

def monitor_file_changes(file_path: str, check_interval: int = 300, alert_after: int = 3600) -> None:
    """
    Monitor a file for changes and alert if it hasn't changed in a while.
    
    Args:
        file_path: Path to the file to monitor
        check_interval: How often to check the file (in seconds)
        alert_after: Alert if the file hasn't changed in this many seconds
    """
    last_modified = 0
    
    while True:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                time.sleep(check_interval)
                continue
            
            # Get current modification time
            current_modified = os.path.getmtime(file_path)
            
            # On first run, just record the time
            if last_modified == 0:
                last_modified = current_modified
                logger.info(f"Started monitoring {file_path}")
                time.sleep(check_interval)
                continue
            
            # Check if the file has been modified
            if current_modified > last_modified:
                logger.info(f"File {file_path} was updated")
                last_modified = current_modified
            else:
                # Calculate time since last update
                time_since_update = time.time() - last_modified
                
                # Alert if it's been too long
                if time_since_update > alert_after:
                    logger.warning(
                        f"File {file_path} has not been updated in "
                        f"{time_since_update/3600:.1f} hours"
                    )
            
            # Wait for next check
            time.sleep(check_interval)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(check_interval)

def run_backup_job(config_file: str, state_file: str, backup_dir: str, 
                  interval: int, max_backups: int) -> None:
    """
    Run a continuous backup job for configuration and state files.
    
    Args:
        config_file: Path to the configuration file
        state_file: Path to the state file
        backup_dir: Directory to store backups
        interval: How often to create backups (in seconds)
        max_backups: Maximum number of backups to keep per file
    """
    logger.info(f"Starting backup job with interval {interval} seconds")
    
    while True:
        try:
            # Back up config file
            create_backup(config_file, backup_dir, max_backups)
            
            # Back up state file
            create_backup(state_file, backup_dir, max_backups)
            
            # Wait for next backup
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Backup job error: {e}")
            time.sleep(interval)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backup and monitor instance files")
    
    # Command options
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument("--config-file", required=True, help="Path to config file")
    backup_parser.add_argument("--state-file", required=True, help="Path to state file")
    backup_parser.add_argument("--backup-dir", default="./backups", help="Backup directory")
    backup_parser.add_argument("--max-backups", type=int, default=10, help="Maximum backups to keep")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor files for changes")
    monitor_parser.add_argument("--file", required=True, help="File to monitor")
    monitor_parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    monitor_parser.add_argument("--alert-after", type=int, default=3600, help="Alert if not changed in seconds")
    
    # Continuous backup job
    job_parser = subparsers.add_parser("job", help="Run continuous backup job")
    job_parser.add_argument("--config-file", required=True, help="Path to config file")
    job_parser.add_argument("--state-file", required=True, help="Path to state file")
    job_parser.add_argument("--backup-dir", default="./backups", help="Backup directory")
    job_parser.add_argument("--interval", type=int, default=3600, help="Backup interval in seconds")
    job_parser.add_argument("--max-backups", type=int, default=24, help="Maximum backups to keep")
    
    args = parser.parse_args()
    
    # Execute requested command
    if args.command == "backup":
        config_backup = create_backup(args.config_file, args.backup_dir, args.max_backups)
        state_backup = create_backup(args.state_file, args.backup_dir, args.max_backups)
        
        if config_backup and state_backup:
            logger.info("Backup completed successfully")
            return 0
        else:
            logger.error("Backup failed")
            return 1
            
    elif args.command == "monitor":
        try:
            logger.info(f"Starting to monitor {args.file}")
            monitor_file_changes(args.file, args.interval, args.alert_after)
            return 0
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
            return 0
            
    elif args.command == "job":
        try:
            run_backup_job(
                args.config_file, 
                args.state_file, 
                args.backup_dir, 
                args.interval, 
                args.max_backups
            )
            return 0
        except KeyboardInterrupt:
            logger.info("Backup job stopped")
            return 0
            
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 