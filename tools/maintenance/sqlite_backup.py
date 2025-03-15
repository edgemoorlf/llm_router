"""Backup utility for SQLite database."""
import os
import time
import logging
import shutil
import sqlite3
from datetime import datetime
import argparse
import glob

logger = logging.getLogger(__name__)

def backup_sqlite_database(
    db_path: str,
    backup_dir: str,
    max_backups: int = 7,
    vacuum: bool = True,
    check_integrity: bool = True
) -> str:
    """
    Create a backup of the SQLite database with optional vacuum and integrity check.
    
    Args:
        db_path: Path to the SQLite database
        backup_dir: Directory to store backups
        max_backups: Maximum number of backups to keep
        vacuum: Whether to vacuum the database before backup
        check_integrity: Whether to check database integrity before backup
        
    Returns:
        Path to the backup file
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"instance_state_{timestamp}.db"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    try:
        # Check integrity if requested
        if check_integrity:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            
            if result != "ok":
                logger.error(f"Database integrity check failed: {result}")
                conn.close()
                raise ValueError(f"Database integrity check failed: {result}")
            
            logger.info("Database integrity check passed")
            
            # Vacuum if requested
            if vacuum:
                cursor.execute("VACUUM")
                logger.info("Database vacuumed")
                
            conn.close()
        
        # Copy the database file
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
        
        # Rotate backups - remove old ones if exceeding max_backups
        if max_backups > 0:
            backup_files = sorted(glob.glob(os.path.join(backup_dir, "instance_state_*.db")))
            if len(backup_files) > max_backups:
                for old_backup in backup_files[:-max_backups]:
                    os.remove(old_backup)
                    logger.info(f"Removed old backup: {old_backup}")
        
        return backup_path
    
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        raise
        
def restore_sqlite_database(
    backup_path: str,
    db_path: str,
    create_backup: bool = True
) -> bool:
    """
    Restore a SQLite database from backup.
    
    Args:
        backup_path: Path to the backup file
        db_path: Path to restore the database to
        create_backup: Whether to create a backup of the current database
        
    Returns:
        True if restoration was successful
    """
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Backup file not found: {backup_path}")
    
    try:
        # Create a backup of the current database if it exists and requested
        if os.path.exists(db_path) and create_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pre_restore_backup = f"{db_path}.pre_restore.{timestamp}"
            shutil.copy2(db_path, pre_restore_backup)
            logger.info(f"Created pre-restore backup: {pre_restore_backup}")
        
        # Copy the backup to the target path
        shutil.copy2(backup_path, db_path)
        logger.info(f"Database restored from {backup_path} to {db_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Restore failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SQLite Database Backup Utility")
    parser.add_argument("--db-path", type=str, help="Path to the SQLite database")
    parser.add_argument("--backup-dir", type=str, help="Directory to store backups")
    parser.add_argument("--max-backups", type=int, default=7, help="Maximum number of backups to keep")
    parser.add_argument("--no-vacuum", action="store_true", help="Skip vacuum operation")
    parser.add_argument("--no-integrity-check", action="store_true", help="Skip integrity check")
    
    args = parser.parse_args()
    
    if not args.db_path:
        default_db_path = os.path.join(os.path.expanduser('~'), '.azure_openai_proxy', 'instance_state.db')
        args.db_path = default_db_path
        logger.info(f"Using default database path: {args.db_path}")
    
    if not args.backup_dir:
        default_backup_dir = os.path.join(os.path.expanduser('~'), '.azure_openai_proxy', 'backups')
        args.backup_dir = default_backup_dir
        logger.info(f"Using default backup directory: {args.backup_dir}")
    
    # Perform backup
    backup_path = backup_sqlite_database(
        db_path=args.db_path,
        backup_dir=args.backup_dir,
        max_backups=args.max_backups,
        vacuum=not args.no_vacuum,
        check_integrity=not args.no_integrity_check
    )
    
    logger.info(f"Backup completed: {backup_path}") 