# OpenAI API Proxy

A FastAPI-based proxy service that re-routes standard OpenAI API calls to various OpenAI-compatible services (including Azure OpenAI) with token rate limiting and multi-instance load balancing.

## Features

- Seamlessly redirects OpenAI API requests to various OpenAI-compatible services
- Supports both Azure OpenAI and generic OpenAI-compatible services (OpenAI, Claude, etc.)
- Maps OpenAI model names to service-specific deployments via configuration
- Supports multiple API instances with load balancing and failover
- Provides different routing strategies (priority, failover, weighted, round-robin, least-loaded, model-specific)
- Handles TPM (tokens per minute) limits for each instance with automatic rate limiting
- Provides automatic failover to healthy instances when rate limits are reached
- Supports both in-memory and Redis-based rate limiting for distributed deployments
- Handles streaming and non-streaming responses
- Supports chat completions, text completions, and embeddings endpoints
- Ensures compatibility with OpenAI clients like the official SDK, LangChain, and AI tools
- Intelligent routing based on model and provider type

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

4. Update the `.env` file with your credentials and model mappings for both Azure and generic OpenAI services.

## Configuration

The application is configured via environment variables. Here are the key settings:

### Multi-Instance Configuration

The proxy supports multiple API instances with load balancing and automatic failover:

- `API_INSTANCES`: Comma-separated list of instance names (e.g., `azure1,azure2,generic1`)
- `API_ROUTING_STRATEGY`: Strategy for routing requests across instances. Options:
  - `failover`: Try instances in priority order (default)
  - `priority`: Always use highest priority instance unless unavailable
  - `round_robin`: Distribute requests evenly among instances
  - `weighted`: Distribute requests according to weight values
  - `least_loaded`: Use instance with lowest TPM consumption
  - `model_specific`: Route requests based on model support (see Model-Specific Routing below)

For each instance, configure:
```
API_INSTANCE_[NAME]_API_KEY=your_api_key
API_INSTANCE_[NAME]_API_BASE=https://your-api-base.com
API_INSTANCE_[NAME]_API_VERSION=2023-07-01-preview
API_INSTANCE_[NAME]_PROVIDER_TYPE=azure     # Provider type: "azure" or "generic" (default: "azure")
API_INSTANCE_[NAME]_PRIORITY=10             # Lower number = higher priority
API_INSTANCE_[NAME]_WEIGHT=100              # Higher number = more traffic (for weighted strategy)
API_INSTANCE_[NAME]_MAX_TPM=240000          # Maximum tokens per minute for this instance
API_INSTANCE_[NAME]_MAX_INPUT_TOKENS=0      # Maximum input tokens allowed (0=unlimited)
API_INSTANCE_[NAME]_SUPPORTED_MODELS=model1,model2  # Models supported by this instance
```

Example configuration for three instances (two Azure and one generic OpenAI-compatible service):
```
API_INSTANCES=azure1,azure2,generic1
API_ROUTING_STRATEGY=failover

# Azure OpenAI instance (primary)
API_INSTANCE_AZURE1_API_KEY=your_azure_key_1
API_INSTANCE_AZURE1_API_BASE=https://your-resource-1.openai.azure.com
API_INSTANCE_AZURE1_PROVIDER_TYPE=azure
API_INSTANCE_AZURE1_PRIORITY=1
API_INSTANCE_AZURE1_MAX_TPM=30000
API_INSTANCE_AZURE1_SUPPORTED_MODELS=gpt-4o,gpt-3.5-turbo
API_INSTANCE_AZURE1_MODEL_MAP_GPT4O=gpt-4o
API_INSTANCE_AZURE1_MODEL_MAP_GPT35TURBO=gpt-35-turbo

# Azure OpenAI instance (secondary)
API_INSTANCE_AZURE2_API_KEY=your_azure_key_2
API_INSTANCE_AZURE2_API_BASE=https://your-resource-2.openai.azure.com
API_INSTANCE_AZURE2_PROVIDER_TYPE=azure
API_INSTANCE_AZURE2_PRIORITY=2
API_INSTANCE_AZURE2_MAX_TPM=30000
API_INSTANCE_AZURE2_SUPPORTED_MODELS=gpt-4o,gpt-3.5-turbo
API_INSTANCE_AZURE2_MODEL_MAP_GPT4O=gpt-4o
API_INSTANCE_AZURE2_MODEL_MAP_GPT35TURBO=gpt-35-turbo

# Generic OpenAI-compatible service
API_INSTANCE_GENERIC1_API_KEY=your_generic_key
API_INSTANCE_GENERIC1_API_BASE=https://api.openai.com
API_INSTANCE_GENERIC1_PROVIDER_TYPE=generic
API_INSTANCE_GENERIC1_PRIORITY=10
API_INSTANCE_GENERIC1_MAX_TPM=30000
API_INSTANCE_GENERIC1_SUPPORTED_MODELS=claude-3-opus,claude-3-sonnet,claude-3-haiku
```

### Legacy Single Instance Configuration

For backward compatibility, you can still use a single instance:

- `API_KEY`: Your API key
- `API_BASE`: Your API endpoint (e.g., `https://your-resource.openai.azure.com`)
- `API_VERSION`: API version to use (default: `2023-07-01-preview`)
- `API_PROVIDER_TYPE`: Provider type, either "azure" or "generic" (default: "azure")

### Provider Types

The proxy supports different provider types:

- `azure`: For Azure OpenAI services
  - Converts standard OpenAI format to Azure format 
  - Maps model names to deployment names
  - Removes the model parameter from requests
  - Adds the API version parameter to requests

- `generic`: For standard OpenAI-compatible services (like OpenAI API, Claude API, etc.)
  - Preserves the original OpenAI format
  - Keeps the model parameter in requests
  - Does not add any additional parameters to requests

Each instance can be configured with a different provider type, allowing you to mix and match different OpenAI-compatible services in the same proxy.

### Model Mappings

Map OpenAI model names to your service-specific deployments:

- `MODEL_MAP_GPT4O`: Your GPT-4o deployment name
- `MODEL_MAP_GPT35TURBO`: Your GPT-3.5 Turbo deployment name
- `MODEL_MAP_DEEPSEEK_R1`: Special mapping for DeepSeek R1 model

You can add additional mappings using the format `MODEL_MAP_<UPPERCASE_MODEL_NAME>`.

For each Azure instance, you can also specify instance-specific model mappings:
```
API_INSTANCE_[NAME]_MODEL_MAP_GPT4O=your-gpt4o-deployment
API_INSTANCE_[NAME]_MODEL_MAP_GPT35TURBO=your-gpt35-deployment
```

### Intelligent Routing

The proxy automatically routes requests to the appropriate service based on:

1. **Model Support**: If a request specifies a model, it will be routed to instances that support that model.
2. **Provider Type**: The proxy will prefer instances with the appropriate provider type for the request.
3. **Availability**: The proxy will automatically failover to healthy instances if an instance is unavailable or rate-limited.

You can influence the routing by configuring:

1. **Model Support**: Configure which models each instance supports
   ```
   API_INSTANCE_[NAME]_SUPPORTED_MODELS=gpt-4o,gpt-3.5-turbo,claude-3-opus
   ```

2. **Routing Strategy**: Set the routing strategy
   ```
   API_ROUTING_STRATEGY=model_specific
   ```

Example configuration for model-specific routing:
```
API_INSTANCES=azure_instance,generic_instance
API_ROUTING_STRATEGY=model_specific

# Azure OpenAI instance
API_INSTANCE_AZURE_INSTANCE_API_KEY=your_azure_key
API_INSTANCE_AZURE_INSTANCE_API_BASE=https://your-azure.openai.azure.com
API_INSTANCE_AZURE_INSTANCE_PROVIDER_TYPE=azure
API_INSTANCE_AZURE_INSTANCE_SUPPORTED_MODELS=gpt-4o,gpt-3.5-turbo
API_INSTANCE_AZURE_INSTANCE_PRIORITY=10
API_INSTANCE_AZURE_INSTANCE_MAX_TPM=240000
API_INSTANCE_AZURE_INSTANCE_MODEL_MAP_GPT4O=gpt-4o
API_INSTANCE_AZURE_INSTANCE_MODEL_MAP_GPT35TURBO=gpt-35-turbo

# Generic OpenAI-compatible instance
API_INSTANCE_GENERIC_INSTANCE_API_KEY=your_generic_key
API_INSTANCE_GENERIC_INSTANCE_API_BASE=https://api.openai.com
API_INSTANCE_GENERIC_INSTANCE_PROVIDER_TYPE=generic
API_INSTANCE_GENERIC_INSTANCE_SUPPORTED_MODELS=claude-3-opus,claude-3-sonnet,claude-3-haiku
API_INSTANCE_GENERIC_INSTANCE_PRIORITY=10
API_INSTANCE_GENERIC_INSTANCE_MAX_TPM=300000
```

With this configuration:
- Requests for `gpt-4o` and `gpt-3.5-turbo` will be routed to `azure_instance`
- Requests for `claude-3-opus`, `claude-3-sonnet`, and `claude-3-haiku` will be routed to `generic_instance`

### Special Model Support

#### DeepSeek R1

The proxy includes special handling for DeepSeek R1 model. When using this model:

1. Set `MODEL_MAP_DEEPSEEK_R1=DeepSeek-R1` in your configuration
2. The proxy will automatically route requests to the correct DeepSeek endpoints
3. The DeepSeek R1 model uses a different endpoint structure than standard Azure OpenAI models:
   - `/deepseek/chat/completions` for chat completions
   - `/deepseek/completions` for text completions
   - `/deepseek/embeddings` for embeddings

### Rate Limiting

- `RATE_LIMIT_TPM`: Token rate limit per minute (default: 30000)
- `RATE_LIMIT_ENABLED`: Enable/disable global rate limiting (default: true)
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

# Standard OpenAI API calls will be redirected appropriately
response = openai.ChatCompletion.create(
    model="gpt-4o",  # Will be routed to Azure OpenAI
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    max_tokens=150
)

print(response)

# Using Claude model (routed to generic provider type)
claude_response = openai.ChatCompletion.create(
    model="claude-3-opus",  # Will be routed to the generic provider
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=500
)

print(claude_response)

# Using DeepSeek R1 model
deepseek_response = openai.ChatCompletion.create(
    model="deepseek-r1",  # Will be routed to DeepSeek R1 endpoints
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=500
)

print(deepseek_response)
```

### cURL Examples

#### Azure OpenAI Model

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

#### Generic OpenAI-compatible Model (Claude)

```bash
curl -X POST http://localhost:3010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-opus",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
      }
    ],
    "max_tokens": 500
  }'
```

#### DeepSeek R1 Model

```bash
curl -X POST http://localhost:3010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
      }
    ],
    "max_tokens": 500
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

- `/v1/instances/status` - Get detailed status of all API instances, including:
  - Health status (healthy, rate-limited, error)
  - Current TPM usage and limits
  - Priority and weight settings
  - Error information
  - Rate limit expiration
  - Provider type (azure or generic)
  
  Example response:
  ```json
  {
    "status": "success",
    "timestamp": 1708995734,
    "instances": [
      {
        "name": "azure1",
        "provider_type": "azure",
        "status": "healthy",
        "current_tpm": 12450,
        "max_tpm": 240000,
        "tpm_usage_percent": 5.19,
        "error_count": 0,
        "last_error": null,
        "rate_limited_until": null,
        "priority": 1,
        "weight": 100,
        "last_used": 1708995730.452,
        "supported_models": ["gpt-4o", "gpt-3.5-turbo"]
      },
      {
        "name": "azure2",
        "provider_type": "azure",
        "status": "rate_limited",
        "current_tpm": 29450,
        "max_tpm": 30000,
        "tpm_usage_percent": 98.17,
        "error_count": 0,
        "last_error": null,
        "rate_limited_until": 1708995790.123,
        "priority": 2,
        "weight": 100,
        "last_used": 1708995680.892,
        "supported_models": ["gpt-4o", "gpt-3.5-turbo"]
      },
      {
        "name": "generic1",
        "provider_type": "generic",
        "status": "healthy",
        "current_tpm": 15000,
        "max_tpm": 30000,
        "tpm_usage_percent": 50.00,
        "error_count": 0,
        "last_error": null,
        "rate_limited_until": null,
        "priority": 10,
        "weight": 50,
        "last_used": 1708995720.123,
        "supported_models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
      }
    ],
    "total_instances": 3,
    "healthy_instances": 2,
    "routing_strategy": "failover"
  }
  ```

- `/health` - Simple health check endpoint (returns 200 OK if server is running)

## Architecture

The proxy is built around a modular architecture with specific services for different provider types:

1. **AzureOpenAIService**: Handles transforming and forwarding requests to Azure OpenAI instances
   - Converts standard OpenAI format to Azure format
   - Maps model names to deployment names
   - Removes the model parameter from requests
   - Adds the API version parameter to requests

2. **GenericOpenAIService**: Handles transforming and forwarding requests to generic OpenAI-compatible instances
   - Preserves the standard OpenAI format
   - Keeps the model parameter in requests
   - Does not add any additional parameters to requests

3. **InstanceManager**: Manages multiple API instances with load balancing and failover
   - Tracks instance status, health, and rate limits
   - Selects instances based on routing strategy, model support, and provider type
   - Handles automatic failover if an instance is unavailable or rate-limited

4. **RateLimiter**: Handles token rate limiting
   - Supports both in-memory and Redis-based rate limiting
   - Tracks token usage for each instance
   - Enforces rate limits based on TPM (tokens per minute)

5. **Router**: Directs requests to the appropriate service based on model and provider type
   - Determines which service to use based on the requested model
   - Transforms and forwards requests
   - Handles streaming and non-streaming responses

## License

[MIT License](LICENSE)
