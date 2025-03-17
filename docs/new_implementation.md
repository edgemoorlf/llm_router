# Instance Management Implementation

This document provides information about the instance management implementation with separated configuration and state architecture.

## Overview

The implementation separates instance data into two distinct components:

1. **Configuration**: Static settings defined by operators (rarely changes)
2. **State**: Dynamic runtime information that changes frequently

This separation provides several benefits:
- Clearer distinction between configuration and runtime state
- Easier to update configurations without affecting runtime state
- Better control over persistence strategies
- Improved debugging and monitoring capabilities

## Architecture

### Key Components

- **NewInstanceManager**: Main class that manages instance configurations and states
- **ConfigStore**: Manages persistence of instance configurations
- **StateStore**: Manages persistence of instance runtime states
- **InstanceRouter**: Routes requests to appropriate instances based on configuration and state

### Storage

By default, the system uses file-based JSON storage:
- `instance_configs.json`: Contains all instance configurations
- `instance_states.json`: Contains all instance runtime states

## Configuration

### Environment Variables

Configure the implementation using the following environment variables:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `INSTANCE_CONFIG_FILE` | Path to the configuration file | `instance_configs.json` |
| `INSTANCE_STATE_FILE` | Path to the state file | `instance_states.json` |

### File Structure

#### Configuration File (instance_configs.json)

```json
{
  "instance-name": {
    "name": "instance-name",
    "provider_type": "azure",
    "api_key": "your-api-key",
    "api_base": "https://example.openai.azure.com",
    "api_version": "2024-05-15",
    "priority": 100,
    "weight": 100,
    "max_tpm": 30000,
    "max_input_tokens": 4000,
    "supported_models": ["gpt-4"],
    "model_deployments": {
      "gpt-4": "gpt-4"
    },
    "enabled": true,
    "timeout_seconds": 60.0,
    "retry_count": 3
  }
}
```

#### State File (instance_states.json)

```json
{
  "instance-name": {
    "name": "instance-name",
    "status": "healthy",
    "error_count": 0,
    "last_error": null,
    "rate_limited_until": null,
    "current_tpm": 0,
    "current_rpm": 0,
    "total_requests": 0,
    "successful_requests": 0,
    "last_used": 0.0,
    "last_error_time": null,
    "avg_latency_ms": null,
    "utilization_percentage": 0.0,
    "connection_status": "unknown",
    "health_status": "unknown"
  }
}
```

## API Endpoints

The implementation provides the following admin API endpoints:

### Instance Management

- `GET /admin/instances`: Get all instances with their configurations and states
- `POST /admin/instances`: Add a new instance
- `GET /admin/instances/{instance_name}/config`: Get configuration for a specific instance
- `PUT /admin/instances/{instance_name}/config`: Update configuration for a specific instance
- `GET /admin/instances/{instance_name}/state`: Get state for a specific instance
- `PUT /admin/instances/{instance_name}/state`: Update state for a specific instance
- `DELETE /admin/instances/{instance_name}`: Delete an instance

## Usage Examples

### Getting All Instances

```bash
curl -X GET "http://localhost:3010/admin/instances" \
     -H "X-Admin-API-Key: youradminapikey"
```

### Getting a Specific Instance Configuration

```bash
curl -X GET "http://localhost:3010/admin/instances/azure2/config" \
     -H "X-Admin-API-Key: youradminapikey"
```

### Updating an Instance Configuration

```bash
curl -X PUT "http://localhost:3010/admin/instances/azure2/config" \
     -H "Content-Type: application/json" \
     -H "X-Admin-API-Key: youradminapikey" \
     -d '{"priority": 50}'
```

### Getting Instance State

```bash
curl -X GET "http://localhost:3010/admin/instances/azure2/state" \
     -H "X-Admin-API-Key: youradminapikey"
```

## Maintenance and Operations

### Backup and Monitoring

It's recommended to regularly back up the configuration and state files. A backup script is provided at `scripts/backup_monitor.py` that can create backups and monitor file changes:

```bash
# Create a backup
python scripts/backup_monitor.py backup --config-file instance_configs.json --state-file instance_states.json

# Monitor for changes
python scripts/backup_monitor.py monitor --file instance_configs.json --interval 300

# Run continuous backup job
python scripts/backup_monitor.py job --config-file instance_configs.json --state-file instance_states.json --interval 3600
```

### Testing

Run the comprehensive test suite to validate the implementation:

```bash
python tests/test_new_implementation.py
```

## Best Practices

1. **Regular Backups**: Always keep backups of both configuration and state files
2. **Configuration Management**: Treat the configuration file as a source-controlled asset
3. **API for Updates**: Prefer using the API endpoints over direct file editing
4. **Monitoring**: Set up monitoring for both files to ensure they're being updated
5. **Performance Testing**: Test under load to ensure the implementation can handle your traffic

## Troubleshooting

### Common Issues

1. **Files Not Found**: Ensure the paths to configuration and state files are correct in your environment variables
2. **Permission Issues**: Check file permissions if you encounter errors writing to the files
3. **Concurrency Problems**: If you see inconsistent data, ensure only one process is writing to the files

### Diagnostic Steps

1. Check application logs for errors
2. Verify file permissions and existence
3. Test API endpoints directly to isolate issues
4. Use the testing script to validate functionality

## Migration Notes

When migrating from the old implementation:

1. Run the migration endpoint (`POST /admin/instances/migrate`) to convert existing data
2. Verify that all instances were properly migrated
3. Monitor the system after migration to ensure proper functionality
4. Keep backups of the old data format until you're confident in the implementation 