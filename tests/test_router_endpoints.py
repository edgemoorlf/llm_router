"""
Test script for the new router endpoints.
This script tests the endpoints after the router reorganization.
"""
import requests
import json
import time
import argparse
from typing import Dict, Any, List

# Default settings
DEFAULT_BASE_URL = "http://localhost:3010"

def test_endpoint(base_url: str, url: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Test an endpoint and return the response.
    
    Args:
        base_url: The base URL of the API
        url: The URL path to test
        method: The HTTP method to use
        data: The request data (for POST/PUT/PATCH)
        
    Returns:
        The response as a dictionary
    """
    full_url = f"{base_url}{url}"
    print(f"\nTesting {method} {full_url}")
    
    headers = {"Content-Type": "application/json"}
    
    try:
        if method.upper() == "GET":
            response = requests.get(full_url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(full_url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = requests.put(full_url, headers=headers, json=data)
        elif method.upper() == "PATCH":
            response = requests.patch(full_url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(full_url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Handle redirects
        if response.status_code in (301, 302, 307, 308):
            print(f"    Redirected to: {response.headers['Location']}")
            
            # Follow the redirect
            if method.upper() == "GET":
                response = requests.get(response.headers['Location'], headers=headers)
            elif method.upper() == "POST":
                response = requests.post(response.headers['Location'], headers=headers, json=data)
        
        # Print response status
        print(f"    Status: {response.status_code}")
        
        # Try to parse as JSON
        try:
            json_response = response.json()
            print(f"    Response: {json.dumps(json_response, indent=2)[:200]}...")
            return json_response
        except:
            print(f"    Response: {response.text[:200]}...")
            return {"error": "Failed to parse response as JSON", "text": response.text}
            
    except Exception as e:
        print(f"    Error: {str(e)}")
        return {"error": str(e)}

def run_tests(base_url, test_group=None):
    """
    Run endpoint tests.
    
    Args:
        base_url: The base URL of the API
        test_group: Optional group of tests to run (instance, stats, redirects, or all)
    """
    print(f"=== Testing Router Endpoints against {base_url} ===")
    
    # Set default test_group to 'all' if not specified
    if not test_group or test_group == 'all':
        groups_to_run = ['instance', 'stats', 'redirects']
    else:
        groups_to_run = [test_group]
    
    first_instance = None
    
    # Test instance management endpoints
    if 'instance' in groups_to_run:
        print("\n=== Instance Management Endpoints ===")
        test_endpoint(base_url, "/instances")
        
        # Get first instance name to test other endpoints
        instances_response = test_endpoint(base_url, "/instances")
        
        if isinstance(instances_response, dict) and "instances" in instances_response:
            if instances_response["instances"]:
                first_instance = instances_response["instances"][0]["name"]
        
        if first_instance:
            print(f"\nUsing instance: {first_instance}")
            test_endpoint(base_url, f"/instances/{first_instance}")
            test_endpoint(base_url, f"/instances/config/{first_instance}")
            test_endpoint(base_url, "/instances/config/all")
            test_endpoint(base_url, f"/instances/verify/{first_instance}", "POST")
        else:
            print("No instances found to test instance-specific endpoints.")
    
    # Test statistics endpoints
    if 'stats' in groups_to_run:
        print("\n=== Statistics Endpoints ===")
        test_endpoint(base_url, "/stats")
        test_endpoint(base_url, "/stats/windows")
        test_endpoint(base_url, "/stats/instances")
        test_endpoint(base_url, "/stats/health")
    
    # Test backward compatibility redirects
    if 'redirects' in groups_to_run:
        print("\n=== Backward Compatibility Redirects ===")
        if first_instance:
            test_endpoint(base_url, f"/health/instances/{first_instance}")
            test_endpoint(base_url, f"/verification/instances/{first_instance}", "POST")
        test_endpoint(base_url, "/health/instances")

def main():
    """Parse command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test router endpoints after reorganization")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, 
                        help=f"Base URL of the API (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--group", choices=["instance", "stats", "redirects", "all"], default="all",
                        help="Test group to run (default: all)")
    args = parser.parse_args()
    
    run_tests(args.base_url, args.group)

if __name__ == "__main__":
    main() 