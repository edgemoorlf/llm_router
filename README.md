# Azure OpenAI Proxy

A FastAPI-based proxy service that re-routes standard OpenAI API calls to Azure OpenAI services with token rate limiting and multi-instance load balancing.

## Features

- Seamlessly redirects OpenAI API requests to Azure OpenAI services
- Maps OpenAI model names to Azure OpenAI deployments via configuration
- Supports multiple Azure OpenAI instances with load balancing and failover
- Provides different routing strategies (priority, failover, weighted, round-robin, least-loaded)
- Handles TPM (tokens per minute) limits for each instance with automatic rate limiting
- Provides automatic failover to healthy instances when rate limits are reached
- Supports both in-memory and Redis-based rate limiting for distributed deployments
- Handles streaming and non-streaming responses
- Supports chat completions, text completions, and embeddings endpoints
- Ensures compatibility with OpenAI clients like the official SDK, LangChain, and AI tools

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Tiktoken
- Redis (optional, for distributed rate limiting)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd azure-openai-proxy
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the `app` directory (you can copy from `.env.example`):

```bash
cp app/.env.example app/.env
```

4. Update the `.env` file with your Azure OpenAI credentials and model mappings.

## Configuration

The application is configured via environment variables. Here are the key settings:

### Multi-Instance Configuration

The proxy supports multiple Azure OpenAI instances with load balancing and automatic failover:

- `AZURE_INSTANCES`: Comma-separated list of instance names (e.g., `1,2,3`)
- `AZURE_ROUTING_STRATEGY`: Strategy for routing requests across instances. Options:
  - `failover`: Try instances in priority order (default)
  - `priority`: Always use highest priority instance unless unavailable
  - `round_robin`: Distribute requests evenly among instances
  - `weighted`: Distribute requests according to weight values
  - `least_loaded`: Use instance with lowest TPM consumption

For each instance, configure:
```
AZURE_INSTANCE_[NAME]_API_KEY=your_azure_api_key_1
AZURE_INSTANCE_[NAME]_API_BASE=https://your-resource1.openai.azure.com
AZURE_INSTANCE_[NAME]_API_VERSION=2023-07-01-preview
AZURE_INSTANCE_[NAME]_PRIORITY=10             # Lower number = higher priority
AZURE_INSTANCE_[NAME]_WEIGHT=100              # Higher number = more traffic (for weighted strategy)
AZURE_INSTANCE_[NAME]_MAX_TPM=240000          # Maximum tokens per minute for this instance
```

Example configuration for two instances:
```
AZURE_INSTANCES=primary,backup
AZURE_ROUTING_STRATEGY=failover

AZURE_INSTANCE_PRIMARY_API_KEY=your_primary_key
AZURE_INSTANCE_PRIMARY_API_BASE=https://your-primary.openai.azure.com
AZURE_INSTANCE_PRIMARY_PRIORITY=10
AZURE_INSTANCE_PRIMARY_MAX_TPM=240000

AZURE_INSTANCE_BACKUP_API_KEY=your_backup_key
AZURE_INSTANCE_BACKUP_API_BASE=https://your-backup.openai.azure.com
AZURE_INSTANCE_BACKUP_PRIORITY=20
AZURE_INSTANCE_BACKUP_MAX_TPM=300000
```

### Legacy Single Instance Configuration

For backward compatibility, you can still use a single instance:

- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_API_BASE`: Your Azure OpenAI endpoint (e.g., `https://your-resource.openai.azure.com`)
- `AZURE_OPENAI_API_VERSION`: API version to use (default: `2023-07-01-preview`)

### Model Mappings

Map OpenAI model names to your Azure deployments:

- `MODEL_MAP_GPT4o`: Your GPT-4o deployment name
- `MODEL_MAP_GPT35TURBO`: Your GPT-3.5 Turbo deployment name

You can add additional mappings using the format `MODEL_MAP_<UPPERCASE_MODEL_NAME>`.

### Rate Limiting

- `TOKEN_RATE_LIMIT`: Token rate limit per minute (default: 30000)
- `USE_REDIS_RATE_LIMITER`: Set to `true` to use Redis instead of in-memory rate limiter
- `REDIS_URL`: Redis connection string (if using Redis rate limiter)

### Server Settings

- `LOG_LEVEL`: Logging level (default: INFO)
- `PORT`: Port to run the server on (default: 3010)

## Running the Service

Start the service with:

```bash
cd azure-openai-proxy
python -m app.main
```

Or use Uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3010 --reload
```

## Usage Examples

Once the proxy is running, you can use it as a drop-in replacement for the OpenAI API. Just change the base URL to point to your proxy.

### Python Example

```python
import openai

# Instead of: openai.api_key = "your-openai-key"
# Use your proxy:
openai.api_base = "http://localhost:3010/v1"
openai.api_key = "dummy-value"  # Can be any value, as the proxy doesn't check it

# Standard OpenAI API calls will be redirected to Azure
response = openai.ChatCompletion.create(
    model="gpt-4o",  # Will be mapped to your Azure deployment
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    max_tokens=150
)

print(response)
```

### cURL Example

```bash
curl -X POST http://localhost:3010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "max_tokens": 150
  }'
```

## Production Deployment

For production deployment, consider:

1. Using a proper ASGI server like Gunicorn with Uvicorn workers
2. Setting up the Redis rate limiter for distributed deployments
3. Adding authentication to the proxy
4. Setting up HTTPS
5. Using a process manager like Supervisor or systemd

Example with Gunicorn:

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3010
```

## API Endpoints

### OpenAI Compatible Endpoints

- `/v1/chat/completions` - Chat completions API
- `/v1/completions` - Text completions API
- `/v1/embeddings` - Embeddings API

### Monitoring Endpoints

- `/v1/instances/status` - Get detailed status of all Azure OpenAI instances, including:
  - Health status (healthy, rate-limited, error)
  - Current TPM usage and limits
  - Priority and weight settings
  - Error information
  - Rate limit expiration
  
  Example response:
  ```json
  {
    "status": "success",
    "timestamp": 1708995734,
    "instances": [
      {
        "name": "primary",
        "status": "healthy",
        "current_tpm": 12450,
        "max_tpm": 240000,
        "tpm_usage_percent": 5.19,
        "error_count": 0,
        "last_error": null,
        "rate_limited_until": null,
        "priority": 10,
        "weight": 100,
        "last_used": 1708995730.452
      },
      {
        "name": "backup",
        "status": "rate_limited",
        "current_tpm": 290450,
        "max_tpm": 300000,
        "tpm_usage_percent": 96.82,
        "error_count": 0,
        "last_error": null,
        "rate_limited_until": 1708995790.123,
        "priority": 20,
        "weight": 80,
        "last_used": 1708995680.892
      }
    ],
    "total_instances": 2,
    "healthy_instances": 1,
    "routing_strategy": "failover"
  }
  ```

- `/health` - Simple health check endpoint (returns 200 OK if server is running)

## License

[MIT License](LICENSE)
