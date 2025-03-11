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

1. **Base Configuration**: `app/config/base.yaml` contains default settings
2. **Environment-Specific**: `app/config/production.yaml`, `app/config/development.yaml`, etc.
3. **Environment Variables**: Values can be referenced using `${ENV_VAR}` syntax

Example configuration:

```yaml
# Base configuration (app/config/base.yaml)
name: "Azure OpenAI Proxy"
version: "1.0.9"
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

You can also use environment variables for version-specific mappings:

```
API_INSTANCE_INSTANCE1_MODEL_MAP_GPT-4O-2024-11-20=gpt4o-november
API_INSTANCE_INSTANCE1_MODEL_MAP_GPT-4O-2024-08-06=gpt4o-august
```

### Environment Variables (Legacy Support)

For backward compatibility, the application also supports configuration via environment variables:

```
# Multiple instances
API_INSTANCES=instance1,instance2

# Instance 1 configuration
API_INSTANCE_INSTANCE1_API_KEY=your-api-key
API_INSTANCE_INSTANCE1_API_BASE=https://instance1.openai.azure.com
API_INSTANCE_INSTANCE1_API_VERSION=2024-08-01-preview
API_INSTANCE_INSTANCE1_PRIORITY=1
API_INSTANCE_INSTANCE1_WEIGHT=100
API_INSTANCE_INSTANCE1_MAX_TPM=240000
API_INSTANCE_INSTANCE1_SUPPORTED_MODELS=gpt-4o-mini,gpt-4o
API_INSTANCE_INSTANCE1_MODEL_MAP_GPT-4O-MINI=gpt4omini
API_INSTANCE_INSTANCE1_MODEL_MAP_GPT-4O=gpt4o

# Instance 2 configuration
API_INSTANCE_INSTANCE2_API_KEY=your-api-key-2
# ... and so on
```

### Legacy Single Instance (Deprecated)

For simple deployments, a single instance can be configured:

```
API_KEY=your-api-key
API_BASE=https://your-instance.openai.azure.com
API_VERSION=2024-08-01-preview
```

## API Endpoints

### OpenAI API Compatibility

- `/v1/chat/completions` - Chat completions API
- `/v1/completions` - Completions API
- `/v1/embeddings` - Embeddings API

### Statistics and Monitoring

- `/stats` - Get service statistics
- `/stats/windows` - Get statistics for multiple time windows
- `/stats/instances` - Get detailed instance statistics
- `/stats/health` - Get health status
- `/stats/status` - Get instance status

### Configuration Management

- `/config` - Get current configuration (excluding secrets)
- `/config/reload` - Reload configuration from disk

## Deployment

### Docker

```bash
docker build -t azure-openai-proxy .
docker run -p 3010:3010 -v /path/to/config:/app/config azure-openai-proxy
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

## License

MIT
