# Azure OpenAI Proxy

A proxy service to convert OpenAI API calls to Azure OpenAI API calls with rate limiting, load balancing, and failover capabilities.

## Features

- **Multiple Instance Support**: Configure multiple Azure OpenAI instances for load balancing and failover
- **Flexible Routing Strategies**: Choose between failover, weighted, and round-robin routing
- **Rate Limiting**: Automatic rate limiting based on TPM (tokens per minute)
- **Comprehensive Statistics**: Track usage, error rates, and performance metrics
- **Health Monitoring**: Monitor the health of all instances
- **Configuration Management**: YAML-based configuration with environment variable support

## Configuration

The application supports a flexible configuration system with multiple options:

### YAML Configuration (Recommended)

The application uses a hierarchical YAML configuration system with environment-specific overrides:

1. **Base Configuration**: `/config/base.yaml` contains default settings
2. **Environment-Specific**: `/config/production.yaml`, `/config/development.yaml`, etc.
3. **Environment Variables**: Values can be referenced using `${ENV_VAR}` syntax

Example configuration:

```yaml
# Base configuration (/config/base.yaml)
name: "Azure OpenAI Proxy"
version: "1.2.0"
port: 3010

routing:
  strategy: "failover"  # failover, weighted, round_robin
  retries: 3
  timeout: 60

logging:
  level: "INFO"
  file: "../logs/app.log"
  max_size: 5242880  # 5MB
  backup_count: 3
  feishu_webhook: "${FEISHU_WEBHOOK_URL}"

monitoring:
  stats_window_minutes: 5
  additional_windows: [15, 30, 60]

# Instance configuration
instances:
  - name: "instance1"
    provider_type: "azure"
    api_key: "${AZURE_API_KEY_1}"
    api_base: "https://example.openai.azure.com"
    api_version: "2024-08-01-preview"
    priority: 100
    weight: 100
    max_tpm: 240000
    supported_models:
      - "gpt-4o-mini"
      - "gpt-4o"
    model_deployments:
      gpt-4o-mini: "gpt4omini"
      gpt-4o: "gpt4o"

Environment variables can be referenced using the `${ENV_VAR}` syntax in YAML configuration files.

### Version-Specific Model Mappings

The proxy enforces strict model matching to ensure version-specific models are correctly routed:

```yaml
instances:
  - name: "instance1"
    # ... other configuration ...
    supported_models:
      - "gpt-4o-2024-11-20"  # Full versioned model name
      - "gpt-4o-2024-08-06"  # Different version of gpt-4o
    model_deployments:
      # Map each specific version to a different deployment
      gpt-4o-2024-11-20: "gpt4o-november"
      gpt-4o-2024-08-06: "gpt4o-august"
```

**Important:** The system uses **exact model name matching only**:
- Requests for `gpt-4o-2024-11-20` will be routed to the `gpt4o-november` deployment
- Requests for `gpt-4o-2024-08-06` will be routed to the `gpt4o-august` deployment
- Requests for generic `gpt-4o` will only work if `gpt-4o` is explicitly listed in supported_models and has its own mapping
- No normalization or fallback occurs - each model version must be explicitly configured

You must configure each specific model version you want to support. For example, if you want to support both the versioned model `gpt-4o-2024-11-20` and the base model `gpt-4o`, you need to include both in your configuration.

## API Endpoints

### OpenAI API Compatibility

- `/v1/chat/completions` - Chat completions API
- `/v1/completions` - Completions API
- `/v1/embeddings` - Embeddings API

### Instance Management

- `/instances` - Get configuration of all instances (with filtering options)
- `/instances/{instance_name}` - Get state of a specific instance including its health and stats
- `/instances/config/{instance_name}` - Get configuration of a specific instance
- `/instances/config/all` - Get configuration of all instances
- `/instances/verify/{instance_name}` - Verify an instance by running a comprehensive set of checks
- `/instances/add` - Add a new instance
- `/instances/add-many` - Add multiple instances
- `/instances/{instance_name}` (DELETE) - Remove an instance
- `/instances/{instance_name}` (PATCH) - Update instance configuration

### Statistics

- `/stats` - Get overall service metrics
- `/stats/windows` - Get metrics for multiple time windows
- `/stats/instances` - Get detailed stats and health info for all instances
- `/stats/health` - Get overall health status summary
- `/stats/reset` - Reset statistics

### Configuration Management

- `/config` - Get current configuration (excluding secrets)
- `/config/reload` - Reload configuration from disk

### Admin API

- `/admin/config/sources` - Get information about configuration sources
- `/admin/config/instances/{instance_name}/sources` - See where each instance setting comes from
- `/admin/config/effective` - Get effective configuration after applying the hierarchy
- `/admin/config/reload` - Reload configuration from all sources

## Implementation Status: Router Reorganization

The following changes have been implemented to reorganize the router implementations:

1. **Removed Endpoints**:
   - ✅ Deleted `/verification/instances/test` endpoint completely

2. **Moved Endpoints**:
   - ✅ Moved `/verification/instances/{instance_name}` → `/instances/verify/{instance_name}`
   - ✅ Moved `/health/instances/{instance_name}` → merged with `/instances/{instance_name}`
   - ✅ Moved `/health/instances` → merged with `/stats/instances`

3. **Added New Endpoints**:
   - ✅ Added `/instances/config/{instance_name}` - Get configuration of a specific instance
   - ✅ Added `/instances/config/all` - Get configuration of all instances 

4. **Updated Instance Management Router**:
   - ✅ Enhanced `/instances/{instance_name}` to include health and stats information
   - ✅ Implemented new config-specific endpoints
   - ✅ Added the verification functionality from the verification router

5. **Updated Statistics Router**:
   - ✅ Consolidated health information into `/stats/instances`
   - ✅ Added `/stats/health` endpoint for overall health status

6. **Removed Deprecated Routers**:
   - ✅ Removed the verification.py router
   - ✅ Removed the health.py router
   - ✅ Deleted the redundant router files from the codebase

7. **Backward Compatibility**:
   - ✅ Added redirects from old endpoints to new endpoints

## Deployment

### Docker

```bash
docker build -t azure-openai-proxy .
docker run -p 3010:3010 -v /path/to/config:/config azure-openai-proxy
```

### Environment Selection

Set the `ENVIRONMENT` variable to select the configuration environment:

```bash
ENVIRONMENT=production python -m app.main
```

## Hot Reloading Configuration

The configuration can be reloaded without restarting the service:

```bash
curl -X POST http://localhost:3010/config/reload
```

## Running with Gunicorn (Multiple Workers)

The proxy service supports running with multiple Gunicorn workers to handle more concurrent requests. When running with multiple workers, each worker has its own process space, so we use a file-based shared state mechanism to ensure that changes made to the instance configuration by one worker are visible to all other workers.

### Configuration

By default, the shared state file is stored in the system temporary directory. You can customize the location by setting the environment variable:

```bash
export INSTANCE_MANAGER_STATE_FILE=/path/to/shared/state.json
```

### Running with Gunicorn

To start the service with Gunicorn:

```bash
pip install gunicorn
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:3010
```

This will start 4 worker processes, each handling requests independently but sharing the instance state through the shared state file.

### Important Notes

- Each worker checks for updates from the shared state file before processing a request.
- Changes to instances (adding, removing, or modifying) are automatically saved to the shared state file.
- The shared state file includes all instance configuration and basic status information.
- For better performance, file access is minimized by checking modification times.

## SQLite Persistence

The proxy service now supports SQLite-based persistence for instance configuration and state management. This provides several advantages over the file-based approach:

- **Improved Reliability**: Atomic transactions ensure data integrity
- **Version History**: Automatic tracking of configuration changes with rollback capability
- **Better Concurrency**: More robust handling of multiple processes/workers
- **Performance**: Optimized for read-heavy workloads

### Configuration Hierarchy

Instance configuration follows a priority-based hierarchy:

1. **SQLite Database** (highest priority)
2. **State Files** (medium priority) 
3. **YAML Configuration** (lower priority)
4. **Default Values** (lowest priority)

Each setting is sourced from the highest-priority location where it's defined, allowing for flexible overrides.

### Database Location

By default, the SQLite database is stored at:

```
~/.azure_openai_proxy/instance_state.db
```

You can customize this location by setting the environment variable:

```bash
export INSTANCE_MANAGER_DB_PATH=/path/to/custom/database.db
```

### Admin API

A new admin API is available for managing configuration sources and viewing the hierarchy:

- `/admin/config/sources` - Get information about configuration sources
- `/admin/config/instances/{instance_name}/sources` - See where each instance setting comes from
- `/admin/config/effective` - Get effective configuration after applying the hierarchy
- `/admin/config/reload` - Reload configuration from all sources

These endpoints require admin authentication using an API key:

```bash
export ADMIN_API_KEY=your-secure-admin-key
```

API requests must include this key in the `X-Admin-API-Key` header.

## Maintenance Utilities

New maintenance utilities are provided for database management:

### Database Backup

The application includes a backup utility for the SQLite database:

```bash
# Basic backup with default settings
python -m app.maintenance.sqlite_backup

# Customized backup
python -m app.maintenance.sqlite_backup \
  --db-path /path/to/instance_state.db \
  --backup-dir /path/to/backup/directory \
  --max-backups 14 \
  --no-vacuum
```

Features:
- Integrity check before backup
- Database optimization (VACUUM)
- Automatic backup rotation
- Timestamped backup files

### Database Health Monitoring

A health check utility is available to monitor database status:

```bash
# Basic health check
python -m app.maintenance.sqlite_health

# Save detailed report to file
python -m app.maintenance.sqlite_health --output health_report.json
```

The health check reports on:
- Database integrity
- File size and growth
- Table structure and row counts
- Version history information

### Recommended Maintenance Schedule

For optimal operation, we recommend:

1. **Daily**: Run database backup with rotation
2. **Weekly**: Run health check and review reports
3. **Monthly**: Perform VACUUM operation to optimize database size
4. **As needed**: Review version history and prune if necessary

### Restoring from Backup

To restore from a backup:

```python
from app.maintenance.sqlite_backup import restore_sqlite_database

restore_sqlite_database(
    backup_path="/path/to/backup/instance_state_20240615_120000.db",
    db_path="/path/to/instance_state.db",
    create_backup=True  # Creates a pre-restore backup
)
```

### Automating Maintenance Tasks

You can automate maintenance tasks using cron jobs:

```bash
# Example crontab entries

# Daily backup at 1:00 AM
0 1 * * * cd /path/to/project && python -m app.maintenance.sqlite_backup --max-backups 14

# Weekly health check on Sunday at 2:00 AM
0 2 * * 0 cd /path/to/project && python -m app.maintenance.sqlite_health --output /var/log/azure_openai_proxy/health_$(date +\%Y\%m\%d).json

# Monthly VACUUM operation on the 1st at 3:00 AM
0 3 1 * * cd /path/to/project && python -c "import sqlite3; conn = sqlite3.connect('/path/to/instance_state.db'); conn.execute('VACUUM'); conn.close()"
```

For production environments, we recommend wrapping these commands in scripts with proper error handling and notifications.

### Migrating from File-Based State

If you're upgrading from a previous version that used file-based state management, migration happens automatically:

1. The system will attempt to load from SQLite first (which will be empty on first run)
2. If no data is found in SQLite, it will fall back to loading from the state file
3. On successful load from the state file, data will be saved to SQLite automatically
4. Future operations will use SQLite as the primary persistence layer

To manually migrate from an existing state file:

```bash
# Stop your application first
python -m app.tools.migrate_from_file \
  --state-file /path/to/existing/state.json \
  --db-path /path/to/new/instance_state.db
```

The migration tool performs these steps:
- Creates the SQLite database if it doesn't exist
- Loads data from the state file
- Converts the data to the new format if needed
- Saves the data to SQLite with proper versioning
- Verifies the migration was successful

## License

MIT
