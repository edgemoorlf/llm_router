"""SQLite implementation of state management for instance configuration."""
import os
import json
import time
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .state_manager import StateManager

logger = logging.getLogger(__name__)

class SQLiteStateManager(StateManager):
    """SQLite-based implementation of state management with versioning support."""
    
    def __init__(self, 
                 db_path: Optional[str] = None, 
                 poll_interval: int = 5,
                 max_versions: int = 10):
        """
        Initialize the SQLite state manager.
        
        Args:
            db_path: Path to SQLite database file, defaults to app data directory
            poll_interval: Interval in seconds for polling updates
            max_versions: Maximum number of versions to keep
        """
        self.db_path = db_path or os.environ.get(
            'INSTANCE_MANAGER_DB_PATH', 
            os.path.join(os.path.expanduser('~'), '.azure_openai_proxy', 'instance_state.db')
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.poll_interval = poll_interval
        self.max_versions = max_versions
        
        # Initialize the database
        self._initialize_db()
        
        logger.info(f"Initialized SQLite state manager with database: {self.db_path}")
    
    def _initialize_db(self):
        """Initialize the SQLite database with necessary tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create instances table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                ''')
                
                # Create instance versions table for change history
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS instance_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data TEXT NOT NULL,
                    comment TEXT
                )
                ''')
                
                # Create index on name for faster lookups
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_instances_name ON instances(name)
                ''')
                
                # Create index on version_id for faster lookups
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_instance_versions_version_id ON instance_versions(version_id)
                ''')
                
                # Create current version table to track active version
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS current_version (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version_id TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                ''')
                
                # Initialize current version if not exists
                cursor.execute('''
                INSERT OR IGNORE INTO current_version (id, version_id, timestamp)
                VALUES (1, 'initial', ?)
                ''', (time.time(),))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {str(e)}")
            raise
    
    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        import uuid
        return f"v-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load instance state from SQLite database.
        
        Returns:
            Dictionary containing instance state, or None if not available/error
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get current version id
                cursor.execute('SELECT version_id FROM current_version WHERE id = 1')
                row = cursor.fetchone()
                if not row:
                    logger.error("No current version found in database")
                    return None
                    
                version_id = row['version_id']
                
                # Get version data
                cursor.execute(
                    'SELECT data, timestamp, worker_id FROM instance_versions WHERE version_id = ?',
                    (version_id,)
                )
                row = cursor.fetchone()
                if not row:
                    logger.error(f"Version data not found for version: {version_id}")
                    return None
                
                # Deserialize the JSON data
                instances_data = json.loads(row['data'])
                
                # Build complete state object
                state = {
                    'timestamp': row['timestamp'],
                    'worker_id': row['worker_id'],
                    'version': version_id,
                    'instances': instances_data
                }
                
                logger.debug(f"Loaded state from database with {len(instances_data)} instances (version: {version_id})")
                return state
                
        except Exception as e:
            logger.error(f"Error loading state from database: {str(e)}")
            return None
    
    def save_state(self, instances_data: List[Dict[str, Any]], worker_id: str) -> bool:
        """
        Save instance state to SQLite database.
        
        Args:
            instances_data: List of instance data dictionaries to save
            worker_id: Unique identifier for the worker saving the state
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            version_id = self._generate_version_id()
            timestamp = time.time()
            
            # JSON serialize the instances data
            instances_json = json.dumps(instances_data)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save the version data
                cursor.execute(
                    'INSERT INTO instance_versions (version_id, worker_id, timestamp, data) VALUES (?, ?, ?, ?)',
                    (version_id, worker_id, timestamp, instances_json)
                )
                
                # Update current version
                cursor.execute(
                    'UPDATE current_version SET version_id = ?, timestamp = ? WHERE id = 1',
                    (version_id, timestamp)
                )
                
                # Also store/update individual instances for querying
                for instance_data in instances_data:
                    name = instance_data.get('name')
                    if not name:
                        logger.warning("Instance data missing name field, skipping individual instance update")
                        continue
                        
                    # Check if instance exists
                    cursor.execute('SELECT id FROM instances WHERE name = ?', (name,))
                    row = cursor.fetchone()
                    
                    instance_json = json.dumps(instance_data)
                    
                    if row:
                        # Update existing instance
                        cursor.execute(
                            'UPDATE instances SET data = ?, updated_at = ? WHERE name = ?',
                            (instance_json, timestamp, name)
                        )
                    else:
                        # Insert new instance
                        cursor.execute(
                            'INSERT INTO instances (name, data, created_at, updated_at) VALUES (?, ?, ?, ?)',
                            (name, instance_json, timestamp, timestamp)
                        )
                
                # Prune old versions
                cursor.execute(
                    '''DELETE FROM instance_versions WHERE id NOT IN (
                        SELECT id FROM instance_versions ORDER BY timestamp DESC LIMIT ?
                    )''',
                    (self.max_versions,)
                )
                
                conn.commit()
                
                logger.debug(f"Saved state to database with {len(instances_data)} instances (version: {version_id})")
                return True
                
        except Exception as e:
            logger.error(f"Error saving state to database: {str(e)}")
            return False
    
    def check_for_updates(self, last_check_time: float) -> Tuple[bool, float]:
        """
        Check if there are updates to the state since last check.
        
        Args:
            last_check_time: Timestamp of last check
            
        Returns:
            Tuple of (has_updates, current_time)
        """
        current_time = time.time()
        
        # Only check if enough time has passed since last check
        if current_time - last_check_time < self.poll_interval:
            return (False, current_time)
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if current version timestamp is newer than last check
                cursor.execute('SELECT timestamp FROM current_version WHERE id = 1')
                row = cursor.fetchone()
                if not row:
                    return (False, current_time)
                
                version_timestamp = row[0]
                return (version_timestamp > last_check_time, current_time)
                
        except Exception as e:
            logger.error(f"Error checking for updates: {str(e)}")
            return (False, current_time)
    
    def get_version_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get version history.
        
        Args:
            limit: Maximum number of versions to retrieve
            
        Returns:
            List of version dictionaries with metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(
                    '''SELECT version_id, worker_id, timestamp, comment, 
                    (SELECT version_id FROM current_version WHERE id = 1) as current_version
                    FROM instance_versions 
                    ORDER BY timestamp DESC LIMIT ?''',
                    (limit,)
                )
                
                versions = []
                for row in cursor.fetchall():
                    versions.append({
                        'version_id': row['version_id'],
                        'worker_id': row['worker_id'],
                        'timestamp': row['timestamp'],
                        'comment': row['comment'],
                        'is_current': row['version_id'] == row['current_version']
                    })
                
                return versions
                
        except Exception as e:
            logger.error(f"Error retrieving version history: {str(e)}")
            return []
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a specific version.
        
        Args:
            version_id: Version ID to rollback to
            
        Returns:
            True if rollback was successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if version exists
                cursor.execute('SELECT COUNT(*) FROM instance_versions WHERE version_id = ?', (version_id,))
                if cursor.fetchone()[0] == 0:
                    logger.error(f"Version not found: {version_id}")
                    return False
                
                # Update current version
                cursor.execute(
                    'UPDATE current_version SET version_id = ?, timestamp = ? WHERE id = 1',
                    (version_id, time.time())
                )
                
                conn.commit()
                
                logger.info(f"Rolled back to version: {version_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error rolling back to version: {str(e)}")
            return False
    
    def get_instance(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific instance by name.
        
        Args:
            name: Instance name
            
        Returns:
            Instance data dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT data FROM instances WHERE name = ? AND is_active = 1', (name,))
                row = cursor.fetchone()
                if not row:
                    return None
                
                return json.loads(row['data'])
                
        except Exception as e:
            logger.error(f"Error retrieving instance {name}: {str(e)}")
            return None 