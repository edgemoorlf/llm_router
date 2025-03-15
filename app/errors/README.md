# Error Handling System

This directory contains a centralized error handling system for the Azure OpenAI Proxy application. The system provides a consistent way to handle, format, and report errors throughout the application.

## Key Components

### 1. Exception Classes (`exceptions.py`)

The exception hierarchy provides domain-specific error types:

- `ProxyError` - Base class for all application errors
  - `ResourceNotFoundError` - For missing resources (404)
    - `InstanceNotFoundError` - Specifically for missing instances
  - `ValidationError` - For request validation errors (400)
  - `ConfigurationError` - For configuration issues (500)
  - `InstanceError` - Base for instance-related errors
    - `InstanceConnectionError` - For connection failures (503)
  - `ModelNotSupportedError` - When models aren't supported (400)
  - `ServiceUnavailableError` - For unavailable services (503)
  - `RateLimitError` - For rate limit errors (429)

Each exception type includes:
- Appropriate HTTP status code
- Standardized message format
- Additional context/details relevant to the error

### 2. Exception Handlers (`handlers.py`)

FastAPI exception handlers that convert exceptions to standardized HTTP responses:

- `proxy_error_handler` - Handles all `ProxyError` types
- `validation_error_handler` - Handles FastAPI validation errors
- `http_exception_handler` - Handles standard HTTP exceptions
- `generic_exception_handler` - Fallback for unexpected exceptions

Also includes:
- `format_error_response` - Utility for consistent error formatting
- `register_exception_handlers` - Function to register all handlers with FastAPI
- `handle_errors` - Decorator for service methods to simplify error handling

### 3. Error Utilities (`utils.py`)

Common error checking operations:

- `check_instance_exists` - Validates instance existence
- `check_model_supported` - Validates model support
- `validate_required_fields` - Validates request payloads
- `handle_router_errors` - Decorator for router endpoints to handle HTTP 500 errors consistently
- `create_500_error` - Creates standardized HTTP 500 error responses

## Usage Guidelines

### In Service Layer

1. Use the `@handle_errors` decorator on service methods:

```python
@handle_errors
async def my_service_method(self, param1, param2):
    # Method implementation
```

2. Raise specific exceptions when errors occur:

```python
if not instance:
    raise InstanceNotFoundError(instance_name=instance_name)
```

### In Router Layer

1. Use the `@handle_router_errors` decorator for endpoints:

```python
@router.get("/my-endpoint")
@handle_router_errors("retrieving data")
async def my_endpoint():
    # Focus on the business logic without worrying about try/except
    # Any uncaught exceptions will be properly formatted as HTTP 500 errors
    result = await service.get_data()
    return result
```

2. Import and use error utilities:

```python
from app.errors.utils import check_instance_exists

# Then in your endpoint:
instance = instance_manager.get_instance(instance_name)
check_instance_exists(instance, instance_name)
```

3. Document exceptions in docstrings:

```python
"""
Method description

Raises:
    InstanceNotFoundError: If the instance doesn't exist
    ValidationError: If the request is invalid
"""
```

4. When you need to raise a 500 error directly:

```python
from app.errors.utils import create_500_error

@router.get("/critical-operation")
async def critical_operation():
    # Some logic that requires manual error raising
    if critical_condition_failed:
        raise create_500_error(
            "Critical operation failed", 
            {"operation_id": operation_id, "status": "failed"}
        )
```

## Benefits

- **Consistency**: All errors follow the same format
- **Separation of Concerns**: Business logic is separate from error handling
- **Improved Debugging**: Better error context and logging
- **Cleaner Code**: Reduced duplication of error handling logic 