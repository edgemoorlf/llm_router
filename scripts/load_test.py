#!/usr/bin/env python3
"""
Load testing script for the Azure OpenAI Proxy.

This script simulates multiple concurrent users making API requests to test:
1. System stability under load
2. Request routing functionality
3. Instance state tracking accuracy
4. Error handling and resilience
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import datetime
import concurrent.futures
import requests
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('load_test.log')
    ]
)

logger = logging.getLogger('load_test')

# Sample prompt templates
PROMPT_TEMPLATES = [
    "Write a brief summary of {topic}.",
    "Explain the concept of {topic} in simple terms.",
    "What are the key aspects of {topic}?",
    "Provide a short explanation about {topic}.",
    "Compare and contrast {topic} with another related concept."
]

# Sample topics
TOPICS = [
    "artificial intelligence",
    "machine learning",
    "cloud computing",
    "quantum computing",
    "blockchain technology",
    "Internet of Things",
    "cybersecurity",
    "data science",
    "virtual reality",
    "augmented reality"
]

# Sample models
MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview"
]

def generate_random_prompt() -> Dict[str, Any]:
    """Generate a random completion request."""
    template = random.choice(PROMPT_TEMPLATES)
    topic = random.choice(TOPICS)
    prompt = template.format(topic=topic)
    model = random.choice(MODELS)
    
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": random.randint(50, 200),
        "temperature": random.uniform(0.0, 1.0)
    }

def send_request(api_url: str, request_data: Dict[str, Any], api_key: str) -> Tuple[bool, Dict[str, Any], float]:
    """
    Send a request to the API and measure latency.
    
    Args:
        api_url: API endpoint URL
        request_data: Request payload
        api_key: API key for authentication
        
    Returns:
        Tuple of (success, response_data, latency)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=request_data,
            timeout=60
        )
        latency = time.time() - start_time
        
        if response.status_code == 200:
            return True, response.json(), latency
        else:
            return False, {"error": f"Status code: {response.status_code}, Response: {response.text}"}, latency
    except Exception as e:
        latency = time.time() - start_time
        return False, {"error": str(e)}, latency

def worker(api_url: str, api_key: str, worker_id: int, num_requests: int) -> Dict[str, Any]:
    """
    Worker function to send multiple requests.
    
    Args:
        api_url: API endpoint URL
        api_key: API key for authentication
        worker_id: ID of this worker
        num_requests: Number of requests to send
        
    Returns:
        Statistics for this worker
    """
    results = {
        "worker_id": worker_id,
        "requests_sent": 0,
        "requests_succeeded": 0,
        "requests_failed": 0,
        "total_latency": 0,
        "errors": [],
        "models_used": {}
    }
    
    for i in range(num_requests):
        logger.debug(f"Worker {worker_id}: Sending request {i+1}/{num_requests}")
        request_data = generate_random_prompt()
        model = request_data["model"]
        
        # Track models used
        if model not in results["models_used"]:
            results["models_used"][model] = 0
        results["models_used"][model] += 1
        
        # Send request
        success, response, latency = send_request(api_url, request_data, api_key)
        results["requests_sent"] += 1
        results["total_latency"] += latency
        
        if success:
            results["requests_succeeded"] += 1
        else:
            results["requests_failed"] += 1
            results["errors"].append(response["error"])
        
        # Add some randomness to request timing
        time.sleep(random.uniform(0.1, 1.0))
    
    return results

def check_instance_states(api_url: str, admin_key: str) -> Dict[str, Any]:
    """
    Check instance states after load test.
    
    Args:
        api_url: Base URL for the API
        admin_key: Admin API key
        
    Returns:
        Dictionary with instance states
    """
    try:
        # Get all instances
        url = f"{api_url}/admin/instances"
        headers = {"X-Admin-API-Key": admin_key}
        
        response = requests.get(url, headers=headers)
        if not response.ok:
            return {"error": f"Failed to get instances: {response.text}"}
        
        instances = response.json()
        
        # Get detailed information for each instance
        states = {}
        for instance_name in instances.get("instances", []):
            instance_url = f"{api_url}/admin/instances/{instance_name}"
            response = requests.get(instance_url, headers=headers)
            
            if response.ok:
                states[instance_name] = response.json()
            else:
                states[instance_name] = {"error": f"Failed to get instance details: {response.text}"}
        
        return states
    except Exception as e:
        return {"error": str(e)}

def run_load_test(api_url: str, api_key: str, admin_key: str, 
                  num_workers: int, requests_per_worker: int) -> Dict[str, Any]:
    """
    Run a load test with multiple concurrent workers.
    
    Args:
        api_url: Base URL for the API
        api_key: API key for authentication
        admin_key: Admin API key for checking instance states
        num_workers: Number of concurrent workers
        requests_per_worker: Number of requests per worker
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Starting load test with {num_workers} workers, {requests_per_worker} requests per worker")
    
    # Check implementation status
    try:
        url = f"{api_url}/admin/instances/implementation"
        headers = {"X-Admin-API-Key": admin_key}
        response = requests.get(url, headers=headers)
        
        if response.ok:
            impl_status = response.json()
            logger.info(f"Implementation status: {impl_status}")
        else:
            logger.warning(f"Failed to check implementation status: {response.text}")
    except Exception as e:
        logger.warning(f"Error checking implementation: {e}")
    
    start_time = time.time()
    completion_url = f"{api_url}/v1/chat/completions"
    
    # Create worker tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            future = executor.submit(
                worker, completion_url, api_key, i, requests_per_worker
            )
            futures.append(future)
        
        # Collect results
        worker_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                worker_results.append(result)
                logger.info(f"Worker {result['worker_id']} completed: {result['requests_succeeded']}/{result['requests_sent']} successful")
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Aggregate results
    total_requests = sum(w["requests_sent"] for w in worker_results)
    successful_requests = sum(w["requests_succeeded"] for w in worker_results)
    failed_requests = sum(w["requests_failed"] for w in worker_results)
    total_latency = sum(w["total_latency"] for w in worker_results)
    
    # Calculate statistics
    avg_latency = total_latency / total_requests if total_requests > 0 else 0
    success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
    requests_per_second = total_requests / total_duration if total_duration > 0 else 0
    
    # Aggregate model usage
    model_usage = {}
    for worker in worker_results:
        for model, count in worker["models_used"].items():
            if model not in model_usage:
                model_usage[model] = 0
            model_usage[model] += count
    
    # Collect all errors
    all_errors = []
    for worker in worker_results:
        all_errors.extend(worker["errors"])
    
    # Check instance states
    logger.info("Checking instance states after load test...")
    instance_states = check_instance_states(api_url, admin_key)
    
    # Prepare results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "test_duration": total_duration,
        "workers": num_workers,
        "requests_per_worker": requests_per_worker,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "requests_per_second": requests_per_second,
        "model_usage": model_usage,
        "errors": all_errors[:10] if len(all_errors) > 10 else all_errors,  # Limit errors
        "error_count": len(all_errors),
        "instance_states": instance_states
    }
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load test for Azure OpenAI Proxy")
    
    parser.add_argument("--api-url", default="http://localhost:3010", help="API base URL")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--admin-key", required=True, help="Admin API key")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=10, help="Requests per worker")
    parser.add_argument("--output", default="load_test_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Run the load test
    results = run_load_test(
        api_url=args.api_url,
        api_key=args.api_key,
        admin_key=args.admin_key,
        num_workers=args.workers,
        requests_per_worker=args.requests
    )
    
    # Print summary
    print("\n=== Load Test Results ===")
    print(f"Total requests: {results['total_requests']}")
    print(f"Success rate: {results['success_rate']:.2f}%")
    print(f"Average latency: {results['avg_latency']:.2f} seconds")
    print(f"Requests per second: {results['requests_per_second']:.2f}")
    print(f"Test duration: {results['test_duration']:.2f} seconds")
    print(f"Workers: {results['workers']}")
    
    # Print model usage
    print("\n== Model Usage ==")
    for model, count in results["model_usage"].items():
        print(f"{model}: {count} requests")
    
    # Print error summary if any
    if results["error_count"] > 0:
        print(f"\n== Errors ({results['error_count']} total) ==")
        for i, error in enumerate(results["errors"]):
            if i < 5:  # Show at most 5 errors
                print(f"- {error}")
        if results["error_count"] > 5:
            print(f"... and {results['error_count'] - 5} more errors")
    
    # Save detailed results to file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 