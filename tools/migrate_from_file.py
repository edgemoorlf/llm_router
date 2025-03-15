"""Migration tool to convert from file-based state to SQLite persistence."""
import os
import json
import logging
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the parent directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.instance.state_manager import FileStateManager
from app.instance.sqlite_state_manager import SQLiteStateManager

logger = logging.getLogger(__name__)

def migrate_from_file_to_sqlite(state_file: str, db_path: str, worker_id: str = None) -> bool:
    """
    Migrate instance data from a file-based state file to SQLite.
    
    Args:
        state_file: Path to the state file
        db_path: Path to the SQLite database
        worker_id: Optional worker ID to use for versioning
        
    Returns:
        True if migration was successful
    """
    logger.info(f"Starting migration from {state_file} to {db_path}")
    
    if not os.path.exists(state_file):
        logger.error(f"State file not found: {state_file}")
        return False
        
    # Create file state manager
    file_manager = FileStateManager(state_file_path=state_file)
    
    # Create SQLite state manager
    sqlite_manager = SQLiteStateManager(db_path=db_path)
    
    try:
        # Load state from file
        state = file_manager.load_state()
        if not state or 'instances' not in state:
            logger.error(f"Invalid or empty state file: {state_file}")
            return False
            
        instances_data = state.get('instances', [])
        logger.info(f"Loaded {len(instances_data)} instances from state file")
        
        # Set worker ID if not provided
        if not worker_id:
            worker_id = state.get('worker_id', f"migration-{int(time.time())}")
            
        # Save to SQLite
        result = sqlite_manager.save_state(instances_data, worker_id)
        
        if result:
            logger.info(f"Successfully migrated {len(instances_data)} instances to SQLite database")
            
            # Verify migration by loading from SQLite
            sqlite_state = sqlite_manager.load_state()
            if not sqlite_state or 'instances' not in sqlite_state:
                logger.error("Verification failed: Could not load state from SQLite after migration")
                return False
                
            sqlite_instances = sqlite_state.get('instances', [])
            if len(sqlite_instances) != len(instances_data):
                logger.error(f"Verification failed: Instance count mismatch - expected {len(instances_data)}, got {len(sqlite_instances)}")
                return False
                
            logger.info("Verification successful: Data migrated correctly")
            
            # Create backup of the original state file
            backup_path = f"{state_file}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                import shutil
                shutil.copy2(state_file, backup_path)
                logger.info(f"Created backup of original state file: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup of original state file: {str(e)}")
            
            return True
        else:
            logger.error("Failed to save state to SQLite database")
            return False
            
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Migrate from file-based state to SQLite")
    parser.add_argument("--state-file", type=str, help="Path to the state file")
    parser.add_argument("--db-path", type=str, help="Path to the SQLite database")
    parser.add_argument("--worker-id", type=str, help="Worker ID to use for versioning")
    
    args = parser.parse_args()
    
    if not args.state_file:
        default_state_file = os.path.join(os.path.expanduser('~'), '.temp', 'instance_state.json')
        if os.path.exists(default_state_file):
            args.state_file = default_state_file
            logger.info(f"Using default state file: {args.state_file}")
        else:
            temp_dir = os.path.join(os.getcwd(), '.temp')
            default_state_file = os.path.join(temp_dir, 'instance_state.json')
            if os.path.exists(default_state_file):
                args.state_file = default_state_file
                logger.info(f"Using default state file: {args.state_file}")
            else:
                parser.error("--state-file is required")
    
    if not args.db_path:
        default_db_path = os.path.join(os.path.expanduser('~'), '.azure_openai_proxy', 'instance_state.db')
        args.db_path = default_db_path
        logger.info(f"Using default database path: {args.db_path}")
    
    # Ensure database directory exists
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # Perform migration
    success = migrate_from_file_to_sqlite(
        state_file=args.state_file,
        db_path=args.db_path,
        worker_id=args.worker_id
    )
    
    if success:
        logger.info("Migration completed successfully")
        print("\n✅ Migration successful! Your instance data has been migrated to SQLite.")
        print(f"Database location: {args.db_path}")
        print(f"A backup of your original state file was created.")
        print("\nYou can now start your application with SQLite persistence.")
    else:
        logger.error("Migration failed")
        print("\n❌ Migration failed. Please check the logs for details.")
        sys.exit(1) 