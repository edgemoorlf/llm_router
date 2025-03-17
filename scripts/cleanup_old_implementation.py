#!/usr/bin/env python3
"""
Cleanup script for removing old implementation files and updating imports.

This script:
1. Identifies files with imports from old implementation
2. Updates imports to use the new instance_context module
3. Removes old implementation files that are no longer needed
"""

import os
import re
import sys
import glob
import shutil
from pathlib import Path

# Configure backup and working directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR = os.path.join(PROJECT_ROOT, "backup_old_implementation")

# Files to remove
FILES_TO_REMOVE = [
    "app/instance/factory.py",  # Old factory implementation
    "app/instance/manager.py",   # Old manager implementation
    "migration/initialize_new_implementation.py",  # Old migration script
    "migration/init_factory.py",  # Old factory initialization
]

# Patterns to update across all Python files
IMPORT_PATTERNS = [
    (r"from app\.instance\.manager import instance_manager", 
     "from app.instance.instance_context import instance_manager"),
    (r"from app\.instance\.factory import InstanceFactory", 
     "# Factory has been replaced with direct instance_context imports"),
    (r"from app\.main import instance_manager", 
     "from app.instance.instance_context import instance_manager"),
]

def create_backup():
    """Create a backup of the project before making changes."""
    print(f"Creating backup in {BACKUP_DIR}")
    
    # Create backup directory
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    # Copy all Python files
    for py_file in glob.glob(os.path.join(PROJECT_ROOT, "app/**/*.py"), recursive=True):
        # Preserve directory structure
        rel_path = os.path.relpath(py_file, PROJECT_ROOT)
        backup_file = os.path.join(BACKUP_DIR, rel_path)
        
        # Create directories if needed
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        
        # Copy the file
        shutil.copy2(py_file, backup_file)
        
    print(f"Backup completed")

def update_imports():
    """Update import statements in all Python files."""
    print("Updating imports...")
    
    # Get all Python files
    py_files = glob.glob(os.path.join(PROJECT_ROOT, "app/**/*.py"), recursive=True)
    
    for file_path in py_files:
        updated = False
        
        # Skip the files we'll be removing
        if any(file_path.endswith(f.lstrip("/")) for f in FILES_TO_REMOVE):
            continue
            
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Apply each pattern
        for pattern, replacement in IMPORT_PATTERNS:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                updated = True
                
        # Write back if modified
        if updated:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Updated imports in {file_path}")

def remove_old_files():
    """Remove files that are no longer needed."""
    print("Removing old implementation files...")
    
    for file_path in FILES_TO_REMOVE:
        full_path = os.path.join(PROJECT_ROOT, file_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"Removed {full_path}")

def update_env_file():
    """Update .env file to remove USE_NEW_IMPLEMENTATION variable."""
    env_file = os.path.join(PROJECT_ROOT, ".env")
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
            
        # Filter out USE_NEW_IMPLEMENTATION line
        updated_lines = [line for line in lines 
                        if not line.strip().startswith("USE_NEW_IMPLEMENTATION=")]
        
        # Write back if modified
        if len(updated_lines) != len(lines):
            with open(env_file, 'w') as f:
                f.writelines(updated_lines)
            print(f"Updated .env file (removed USE_NEW_IMPLEMENTATION)")

def main():
    """Main function."""
    print("Starting cleanup of old implementation files")
    
    # Create backup first
    create_backup()
    
    # Update imports
    update_imports()
    
    # Remove old files
    remove_old_files()
    
    # Update .env file
    update_env_file()
    
    print("\nCleanup completed successfully!")
    print(f"A backup of the original files was created in {BACKUP_DIR}")
    print("You can now run your application with the new implementation")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1) 