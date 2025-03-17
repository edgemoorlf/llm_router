"""
Comprehensive tests for the new implementation with separated config and state.

This module provides tests to validate that the new instance management system
works correctly under various conditions.
"""

import os
import sys
import json
import time
import random
import requests
import threading
import concurrent.futures
from typing import Dict, List, Any

# Add parent directory to path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.instance import InstanceConfig, InstanceState
from app.storage.config_store import ConfigStore
from app.storage.state_store import StateStore
from app.instance.new_manager import NewInstanceManager

# Configuration
API_BASE_URL = "http://localhost:3010"
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "youradminapikey")
TEST_INSTANCE_NAME = "test-instance"
CONFIG_FILE = "test_instance_configs.json"
STATE_FILE = "test_instance_states.json"

def setup():
    """Set up test environment."""
    # Clean up any existing test files
    for file in [CONFIG_FILE, STATE_FILE]:
        if os.path.exists(file):
            os.remove(file)
    
    # Create test instance manager with empty configs
    manager = NewInstanceManager(config_file=CONFIG_FILE, state_file=STATE_FILE)
    
    # Add a test instance
    config = InstanceConfig(
        name=TEST_INSTANCE_NAME,
        provider_type="azure",
        api_key="test-api-key",
        api_base="https://test.openai.azure.com",
        api_version="2024-05-01",
        priority=100,
        weight=100,
        max_tpm=10000,
        max_input_tokens=4000,
        supported_models=["gpt-4o-mini"],
        model_deployments={"gpt-4o-mini": "gpt-4o-mini"}
    )
    manager.add_instance(config)
    
    return manager

def teardown(manager: NewInstanceManager):
    """Clean up test environment."""
    manager.delete_instance(TEST_INSTANCE_NAME)
    
    # Clean up test files
    for file in [CONFIG_FILE, STATE_FILE]:
        if os.path.exists(file):
            os.remove(file)

def test_config_state_separation(manager: NewInstanceManager) -> bool:
    """Test that configuration and state are properly separated."""
    print("Testing configuration and state separation...")
    
    # Verify initial state
    config = manager.get_instance_config(TEST_INSTANCE_NAME)
    state = manager.get_instance_state(TEST_INSTANCE_NAME)
    
    if not config or not state:
        print("❌ Failed to retrieve config or state")
        return False
    
    # Update configuration
    manager.update_instance_config(
        TEST_INSTANCE_NAME,
        priority=50,
        max_tpm=20000
    )
    
    # Update state
    manager.update_instance_state(
        TEST_INSTANCE_NAME,
        status="healthy",
        total_requests=10,
        successful_requests=8
    )
    
    # Verify updates were applied separately
    new_config = manager.get_instance_config(TEST_INSTANCE_NAME)
    new_state = manager.get_instance_state(TEST_INSTANCE_NAME)
    
    if new_config.priority != 50 or new_config.max_tpm != 20000:
        print("❌ Configuration updates were not applied")
        return False
        
    if new_state.status != "healthy" or new_state.total_requests != 10:
        print("❌ State updates were not applied")
        return False
    
    print("✅ Configuration and state separation test passed")
    return True

def test_api_endpoints() -> bool:
    """Test the API endpoints for the new implementation."""
    print("Testing API endpoints...")
    
    # Helper function to make authenticated requests
    def api_request(method, endpoint, data=None):
        url = f"{API_BASE_URL}{endpoint}"
        headers = {"X-Admin-API-Key": ADMIN_API_KEY}
        
        if method.lower() == "get":
            return requests.get(url, headers=headers)
        elif method.lower() == "post":
            return requests.post(url, headers=headers, json=data)
        elif method.lower() == "put":
            return requests.put(url, headers=headers, json=data)
        elif method.lower() == "delete":
            return requests.delete(url, headers=headers)
    
    # Test implementation status endpoint
    try:
        response = api_request("get", "/admin/instances/implementation")
        if not response.ok or not response.json().get("using_new_implementation"):
            print("❌ Implementation status endpoint failed")
            return False
        print("✅ Implementation status endpoint passed")
    except Exception as e:
        print(f"❌ Implementation status endpoint error: {e}")
        return False
    
    # Test getting all instances
    try:
        response = api_request("get", "/admin/instances/new")
        if not response.ok or "instances" not in response.json():
            print("❌ Get all instances endpoint failed")
            return False
        print("✅ Get all instances endpoint passed")
    except Exception as e:
        print(f"❌ Get all instances endpoint error: {e}")
        return False
    
    # Test getting a specific instance config
    try:
        # Get a real instance name from the all instances response
        all_instances = api_request("get", "/admin/instances/new").json()
        if not all_instances.get("instances"):
            print("❌ No instances available for testing")
            return False
            
        instance_name = list(all_instances["instances"].keys())[0]
        response = api_request("get", f"/admin/instances/new/{instance_name}/config")
        
        if not response.ok or not response.json().get("name"):
            print("❌ Get instance config endpoint failed")
            return False
        print(f"✅ Get instance config endpoint passed for {instance_name}")
    except Exception as e:
        print(f"❌ Get instance config endpoint error: {e}")
        return False
    
    # Test updating a specific instance config
    try:
        # Use the same instance name
        current_priority = api_request("get", f"/admin/instances/new/{instance_name}/config").json().get("priority", 100)
        new_priority = 200 if current_priority != 200 else 100
        
        response = api_request("put", f"/admin/instances/new/{instance_name}/config", {"priority": new_priority})
        if not response.ok or "success" not in response.json().get("status", ""):
            print("❌ Update instance config endpoint failed")
            return False
            
        # Verify the update
        updated_config = api_request("get", f"/admin/instances/new/{instance_name}/config").json()
        if updated_config.get("priority") != new_priority:
            print("❌ Update was not applied correctly")
            return False
            
        print(f"✅ Update instance config endpoint passed for {instance_name}")
        
        # Reset the priority back
        api_request("put", f"/admin/instances/new/{instance_name}/config", {"priority": current_priority})
    except Exception as e:
        print(f"❌ Update instance config endpoint error: {e}")
        return False
    
    print("✅ All API endpoint tests passed")
    return True

def test_concurrent_updates(manager: NewInstanceManager) -> bool:
    """Test concurrent updates to configuration and state."""
    print("Testing concurrent updates...")
    
    # Define a function to update config in a thread
    def update_config(thread_id):
        for i in range(5):
            try:
                manager.update_instance_config(
                    TEST_INSTANCE_NAME,
                    priority=random.randint(1, 100),
                    weight=random.randint(1, 100)
                )
                time.sleep(0.01)  # Small sleep to yield
            except Exception as e:
                print(f"Config thread {thread_id} error: {e}")
                return False
        return True
    
    # Define a function to update state in a thread
    def update_state(thread_id):
        for i in range(5):
            try:
                manager.update_instance_state(
                    TEST_INSTANCE_NAME,
                    total_requests=random.randint(1, 100),
                    successful_requests=random.randint(1, 100)
                )
                time.sleep(0.01)  # Small sleep to yield
            except Exception as e:
                print(f"State thread {thread_id} error: {e}")
                return False
        return True
    
    # Create and start threads
    threads = []
    for i in range(5):
        t1 = threading.Thread(target=update_config, args=(i,))
        t2 = threading.Thread(target=update_state, args=(i,))
        threads.append(t1)
        threads.append(t2)
        t1.start()
        t2.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Verify final state
    config = manager.get_instance_config(TEST_INSTANCE_NAME)
    state = manager.get_instance_state(TEST_INSTANCE_NAME)
    
    if not config or not state:
        print("❌ Failed to retrieve config or state after concurrent updates")
        return False
    
    print("✅ Concurrent updates test passed")
    return True

def test_load(num_requests: int = 50) -> bool:
    """Test the system under load."""
    print(f"Testing system under load with {num_requests} requests...")
    
    # Helper function to make an API request
    def make_request(request_id):
        try:
            url = f"{API_BASE_URL}/admin/instances/implementation"
            headers = {"X-Admin-API-Key": ADMIN_API_KEY}
            response = requests.get(url, headers=headers)
            return response.ok
        except Exception as e:
            print(f"Request {request_id} error: {e}")
            return False
    
    # Use a thread pool to make concurrent requests
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1
    
    success_rate = (success_count / num_requests) * 100
    print(f"Load test completed: {success_count}/{num_requests} successful ({success_rate:.2f}%)")
    
    return success_rate >= 95  # At least 95% success rate

def main():
    """Run all tests."""
    print("\n🔍 Starting tests for the new implementation\n")
    
    # Set up test environment
    manager = setup()
    
    # Run tests
    config_state_test = test_config_state_separation(manager)
    api_test = test_api_endpoints()
    concurrent_test = test_concurrent_updates(manager)
    load_test = test_load()
    
    # Clean up
    teardown(manager)
    
    # Print summary
    print("\n📋 Test Results:")
    print(f"Configuration/State Separation: {'✅ PASSED' if config_state_test else '❌ FAILED'}")
    print(f"API Endpoints: {'✅ PASSED' if api_test else '❌ FAILED'}")
    print(f"Concurrent Updates: {'✅ PASSED' if concurrent_test else '❌ FAILED'}")
    print(f"Load Testing: {'✅ PASSED' if load_test else '❌ FAILED'}")
    
    overall_result = all([config_state_test, api_test, concurrent_test, load_test])
    print(f"\nOverall Result: {'✅ PASSED' if overall_result else '❌ FAILED'}")
    
    return 0 if overall_result else 1

if __name__ == "__main__":
    sys.exit(main()) 