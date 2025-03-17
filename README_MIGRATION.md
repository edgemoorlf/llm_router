# Migration to New Instance Management System

This document describes the migration process to the new instance management system with separated configuration and state architecture.

## Architecture Overview

The new instance management system separates instance data into two components:

1. **Configuration** - Static settings that rarely change (API keys, endpoints, models, capacities)
2. **State** - Dynamic runtime information that changes frequently (request counts, errors, token usage)

Key components include:

- `NewInstanceManager` - Manages instances with separated configuration and state
- `InstanceRouter` - Routes requests to appropriate instances based on criteria
- `InstanceFactory` - Facilitates transition between old and new implementations
- `ConfigStore` - Manages instance configuration persistence
- `StateStore` - Manages instance state persistence

## Migration Process

The migration involves these steps:

1. **Backup** - Backup existing configuration files
2. **Migration** - Use the migration endpoint to convert data
3. **Validation** - Verify instances were migrated correctly
4. **Deployment** - Update environment and set up monitoring

## Files and Scripts

### Main Components

- `app/instance/new_manager.py` - New instance manager implementation
- `app/instance/new_router.py` - New router implementation
- `app/instance/factory.py` - Factory for old/new implementation switching
- `app/storage/config_store.py` - Configuration persistence
- `app/storage/state_store.py` - State persistence

### Support Scripts

- `scripts/backup_monitor.py` - Handles backups and monitoring
- `scripts/deploy.py` - Deployment script for production environments

### Documentation

- `docs/new_implementation.md` - Detailed documentation on usage and architecture

## Usage Guide

### Deployment

To deploy in production:

```bash
python scripts/deploy.py --admin-key YOUR_ADMIN_KEY --all
```

This will:
1. Back up existing files
2. Migrate instances if using the old implementation
3. Set up environment variables
4. Configure monitoring and backups
5. Verify the deployment

For more options:

```bash
python scripts/deploy.py --help
```

### Backup and Monitoring

The backup and monitoring script provides three main functions:

1. **Backup** - Create backups of configuration and state files
2. **Monitor** - Monitor files for changes and alert if stale
3. **Continuous Backup** - Run regular backups on a schedule

Example:

```bash
# Create a backup
python scripts/backup_monitor.py backup --config-file instance_configs.json --state-file instance_states.json

# Monitor a file for changes
python scripts/backup_monitor.py monitor --file instance_states.json --alert-after 3600

# Run continuous backup job
python scripts/backup_monitor.py job --config-file instance_configs.json --state-file instance_states.json --interval 3600
```

### API Endpoints

The following admin endpoints are available:

- `GET /admin/instances/implementation` - Check implementation status
- `POST /admin/instances/migrate` - Migrate to new implementation
- `GET /admin/instances/new` - List all instances
- `POST /admin/instances/new` - Add a new instance
- `GET /admin/instances/new/{instance_name}` - Get instance details
- `DELETE /admin/instances/new/{instance_name}` - Delete an instance
- `PUT /admin/instances/new/{instance_name}/config` - Update instance configuration
- `PUT /admin/instances/new/{instance_name}/state` - Update instance state

## Benefits of New Implementation

1. **Cleaner Separation** - Clear distinction between configuration and state
2. **Improved Performance** - More efficient state updates
3. **Better Debugging** - Easier to identify and resolve issues
4. **Enhanced Reliability** - More robust persistence of instance data
5. **Scalability** - Better support for managing many instances

## Best Practices

1. Regularly back up configuration and state files
2. Use the API endpoints for updates rather than editing files directly
3. Monitor file changes to detect potential issues
4. Test thoroughly after making configuration changes
5. Follow the deployment script for production updates

## Troubleshooting

Common issues and solutions:

1. **File permissions** - Ensure the application has read/write access to configuration and state files
2. **Missing instances** - Check migration logs and verify instance data in files
3. **API errors** - Verify admin API key and check application logs
4. **Performance issues** - Monitor instance states for high traffic or errors

## Next Steps

After successful migration:

1. Monitor application performance with the new implementation
2. Consider implementing automated testing for instance management
3. Update documentation for any application-specific configurations
4. Train team members on the new architecture 