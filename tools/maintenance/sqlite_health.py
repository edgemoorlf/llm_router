"""Health monitoring utilities for SQLite database."""
import os
import time
import logging
import sqlite3
import json
from datetime import datetime
import argparse

logger = logging.getLogger(__name__)

class SQLiteHealthCheck:
    """SQLite database health checker."""
    
    def __init__(self, db_path: str):
        """
        Initialize the health checker.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
            
    def check_integrity(self) -> bool:
        """
        Run SQLite integrity check.
        
        Returns:
            True if integrity check passes
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            conn.close()
            
            return result == "ok"
        except Exception as e:
            logger.error(f"Integrity check failed: {str(e)}")
            return False
            
    def get_db_stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary of database statistics
        """
        stats = {
            "file_size_bytes": os.path.getsize(self.db_path),
            "file_size_mb": os.path.getsize(self.db_path) / (1024 * 1024),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(self.db_path)).isoformat(),
            "tables": {},
            "integrity": None
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check database integrity
            cursor.execute("PRAGMA integrity_check")
            stats["integrity"] = cursor.fetchone()[0]
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get row count for each table
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Get schema information
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [{'name': row[1], 'type': row[2]} for row in cursor.fetchall()]
                
                stats["tables"][table] = {
                    "row_count": row_count,
                    "columns": columns
                }
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            stats["error"] = str(e)
            
        return stats
        
    def check_version_history(self) -> dict:
        """
        Check version history table.
        
        Returns:
            Dictionary with version history information
        """
        result = {
            "total_versions": 0,
            "current_version": None,
            "oldest_version": None,
            "newest_version": None
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if version history table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='instance_versions'")
            if not cursor.fetchone():
                result["error"] = "Instance versions table does not exist"
                return result
                
            # Get current version
            cursor.execute("SELECT version_id, timestamp FROM current_version WHERE id = 1")
            current = cursor.fetchone()
            if current:
                result["current_version"] = {
                    "version_id": current["version_id"],
                    "timestamp": datetime.fromtimestamp(current["timestamp"]).isoformat()
                }
            
            # Get version count
            cursor.execute("SELECT COUNT(*) FROM instance_versions")
            result["total_versions"] = cursor.fetchone()[0]
            
            # Get oldest version
            cursor.execute("SELECT version_id, timestamp FROM instance_versions ORDER BY timestamp ASC LIMIT 1")
            oldest = cursor.fetchone()
            if oldest:
                result["oldest_version"] = {
                    "version_id": oldest["version_id"],
                    "timestamp": datetime.fromtimestamp(oldest["timestamp"]).isoformat()
                }
                
            # Get newest version
            cursor.execute("SELECT version_id, timestamp FROM instance_versions ORDER BY timestamp DESC LIMIT 1")
            newest = cursor.fetchone()
            if newest:
                result["newest_version"] = {
                    "version_id": newest["version_id"],
                    "timestamp": datetime.fromtimestamp(newest["timestamp"]).isoformat()
                }
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to check version history: {str(e)}")
            result["error"] = str(e)
            
        return result

def run_health_check(db_path: str, output_file: str = None) -> dict:
    """
    Run all health checks and return results.
    
    Args:
        db_path: Path to the SQLite database
        output_file: Optional path to write results to JSON file
        
    Returns:
        Dictionary with health check results
    """
    try:
        checker = SQLiteHealthCheck(db_path)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "database_path": db_path,
            "integrity_check": checker.check_integrity(),
            "database_stats": checker.get_db_stats(),
            "version_history": checker.check_version_history()
        }
        
        # Write results to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        return results
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SQLite Database Health Check")
    parser.add_argument("--db-path", type=str, help="Path to the SQLite database")
    parser.add_argument("--output", type=str, help="Path to output JSON file")
    
    args = parser.parse_args()
    
    if not args.db_path:
        default_db_path = os.path.join(os.path.expanduser('~'), '.azure_openai_proxy', 'instance_state.db')
        args.db_path = default_db_path
        logger.info(f"Using default database path: {args.db_path}")
    
    # Run health check
    results = run_health_check(args.db_path, args.output)
    
    # Display results summary
    print("\nSQLite Database Health Check Results:")
    print(f"Database: {args.db_path}")
    print(f"Integrity Check: {'PASSED' if results['integrity_check'] else 'FAILED'}")
    
    db_stats = results['database_stats']
    print(f"Size: {db_stats['file_size_mb']:.2f} MB")
    print(f"Tables: {len(db_stats['tables'])}")
    
    version_history = results['version_history']
    print(f"Versions: {version_history['total_versions']}")
    
    if args.output:
        print(f"\nDetailed results written to: {args.output}") 