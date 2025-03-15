# Testing the Router Reorganization

This directory contains testing tools for the Azure OpenAI Proxy, particularly for validating the router reorganization implementation.

## Router Endpoint Tests

The `test_router_endpoints.py` script tests all the endpoints after the router reorganization to ensure they work correctly. It also tests the backward compatibility redirects that were put in place during the reorganization.

### Usage

To run the tests:

```bash
python -m tests.test_router_endpoints
```

This will test all endpoints against the default API URL (`http://localhost:3010`).

### Options

The script supports several command-line options:

- `--base-url`: Specify a different base URL for the API (default: `http://localhost:3010`)
- `--group`: Specify which group of tests to run:
  - `instance`: Test instance management endpoints
  - `stats`: Test statistics endpoints
  - `redirects`: Test backward compatibility redirects
  - `all`: Test all endpoint groups (default)

### Examples

Test against a different server:
```bash
python -m tests.test_router_endpoints --base-url http://api.example.com
```

Test only instance management endpoints:
```bash
python -m tests.test_router_endpoints --group instance
```

Test only redirect functionality:
```bash
python -m tests.test_router_endpoints --group redirects
```

## Main Test Features

1. **Instance Management Tests**:
   - GET `/instances` - List all instances
   - GET `/instances/{instance_name}` - Get instance details with health info
   - GET `/instances/config/{instance_name}` - Get instance configuration
   - GET `/instances/config/all` - Get all instance configurations
   - POST `/instances/verify/{instance_name}` - Verify an instance

2. **Statistics Tests**:
   - GET `/stats` - Get overall service metrics
   - GET `/stats/windows` - Get metrics for multiple time windows
   - GET `/stats/instances` - Get detailed stats and health info
   - GET `/stats/health` - Get overall health status summary

3. **Backward Compatibility Tests**:
   - GET `/health/instances/{instance_name}` → redirects to `/instances/{instance_name}`
   - POST `/verification/instances/{instance_name}` → redirects to `/instances/verify/{instance_name}`
   - GET `/health/instances` → redirects to `/stats/instances` 