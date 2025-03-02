"""Test script for the Azure OpenAI Proxy."""
import os
import sys
import json
import logging
import argparse
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chat_completions(base_url, model="gpt-4"):
    """Test chat completions endpoint with the OpenAI client library."""
    try:
        import openai
        
        # Configure OpenAI client to use our proxy
        openai.api_base = base_url
        openai.api_key = "dummy-key"  # Can be any value as the proxy doesn't check it
        
        # Make a chat completion request
        logger.info(f"Testing chat completions with model: {model}")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hi and tell me what time it is in a single sentence."}
            ],
            max_tokens=150
        )
        
        # Print the response
        logger.info("Chat completion successful!")
        logger.info(f"Response: {response.choices[0].message.content}")
        logger.info(f"Usage: {json.dumps(response.usage.to_dict(), indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing chat completions: {str(e)}")
        return False

def test_completions(base_url, model="text-davinci-003"):
    """Test completions endpoint with the OpenAI client library."""
    try:
        import openai
        
        # Configure OpenAI client to use our proxy
        openai.api_base = base_url
        openai.api_key = "dummy-key"  # Can be any value as the proxy doesn't check it
        
        # Make a completion request
        logger.info(f"Testing completions with model: {model}")
        response = openai.Completion.create(
            model=model,
            prompt="Write a haiku about programming:",
            max_tokens=50
        )
        
        # Print the response
        logger.info("Completion successful!")
        logger.info(f"Response: {response.choices[0].text}")
        logger.info(f"Usage: {json.dumps(response.usage.to_dict(), indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing completions: {str(e)}")
        return False

def test_embeddings(base_url, model="text-embedding-ada-002"):
    """Test embeddings endpoint with the OpenAI client library."""
    try:
        import openai
        
        # Configure OpenAI client to use our proxy
        openai.api_base = base_url
        openai.api_key = "dummy-key"  # Can be any value as the proxy doesn't check it
        
        # Make an embeddings request
        logger.info(f"Testing embeddings with model: {model}")
        response = openai.Embedding.create(
            model=model,
            input="The food was delicious and the waiter was helpful."
        )
        
        # Print the response
        logger.info("Embeddings successful!")
        logger.info(f"Embedding dimensions: {len(response.data[0].embedding)}")
        logger.info(f"Usage: {json.dumps(response.usage.to_dict(), indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing embeddings: {str(e)}")
        return False

def test_streaming(base_url, model="gpt-4"):
    """Test streaming responses with the OpenAI client library."""
    try:
        import openai
        
        # Configure OpenAI client to use our proxy
        openai.api_base = base_url
        openai.api_key = "dummy-key"  # Can be any value as the proxy doesn't check it
        
        # Make a streaming chat completion request
        logger.info(f"Testing streaming chat completions with model: {model}")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 10, with each number on a new line."}
            ],
            max_tokens=100,
            stream=True
        )
        
        # Print the streaming responses
        logger.info("Streaming response:")
        for chunk in response:
            content = chunk.choices[0].delta.get("content", "")
            if content:
                print(content, end="", flush=True)
        print()  # Add a newline at the end
        
        logger.info("Streaming test completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error testing streaming: {str(e)}")
        return False

def test_rate_limiting(base_url, model="gpt-4"):
    """Test rate limiting by making multiple large requests quickly."""
    try:
        import openai
        import time
        
        # Configure OpenAI client to use our proxy
        openai.api_base = base_url
        openai.api_key = "dummy-key"  # Can be any value as the proxy doesn't check it
        
        # Make multiple large requests to test rate limiting
        logger.info("Testing rate limiting with multiple large requests")
        
        # Generate a large prompt that will use many tokens
        large_prompt = "Explain the theory of relativity in detail. " * 10
        
        # Make multiple requests
        for i in range(5):
            try:
                logger.info(f"Making request {i+1}/5")
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": large_prompt}
                    ],
                    max_tokens=500
                )
                logger.info(f"Request {i+1} successful. Tokens used: {response.usage.total_tokens}")
                
                # Don't wait between requests to trigger rate limiting
            except Exception as e:
                if "rate limit" in str(e).lower():
                    logger.info(f"Rate limit triggered as expected: {str(e)}")
                    return True
                else:
                    logger.error(f"Error during rate limit test: {str(e)}")
        
        logger.warning("Rate limit test completed without triggering limits. Check your configuration.")
        return True
    except Exception as e:
        logger.error(f"Error testing rate limiting: {str(e)}")
        return False

def test_instances_status(base_url):
    """Test the instances status endpoint."""
    try:
        import requests
        
        # Make a request to the instances status endpoint
        logger.info("Testing instances status endpoint")
        response = requests.get(f"{base_url}/instances/status")
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Failed to get instances status: {response.status_code} - {response.text}")
            return False
        
        # Parse the response
        status_data = response.json()
        
        # Log the instance status
        logger.info(f"Status: {status_data['status']}")
        logger.info(f"Total instances: {status_data['total_instances']}")
        logger.info(f"Healthy instances: {status_data['healthy_instances']}")
        logger.info(f"Routing strategy: {status_data['routing_strategy']}")
        
        # Log details for each instance
        for instance in status_data.get('instances', []):
            logger.info(f"Instance: {instance['name']}")
            logger.info(f"  Status: {instance['status']}")
            logger.info(f"  TPM usage: {instance['current_tpm']}/{instance['max_tpm']} ({instance['tpm_usage_percent']}%)")
            logger.info(f"  Priority: {instance['priority']}")
            logger.info(f"  Weight: {instance['weight']}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing instances status endpoint: {str(e)}")
        return False

def test_failover(base_url, model="gpt-4"):
    """Test failover functionality between instances."""
    try:
        import openai
        import requests
        import time
        
        # Configure OpenAI client to use our proxy
        openai.api_base = base_url
        openai.api_key = "dummy-key"  # Can be any value as the proxy doesn't check it
        
        # First, check the current status to see how many instances we have
        status_response = requests.get(f"{base_url}/instances/status")
        status_data = status_response.json()
        
        total_instances = status_data.get('total_instances', 0)
        if total_instances <= 1:
            logger.warning("Skipping failover test as there is only one Azure OpenAI instance configured")
            return True
        
        logger.info(f"Testing failover functionality with {total_instances} instances")
        
        # Make multiple requests to see if they succeed even if some fail
        for i in range(5):
            try:
                logger.info(f"Making request {i+1}/5")
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Tell me a very brief joke about number {i+1}"}
                    ],
                    max_tokens=50
                )
                logger.info(f"Request {i+1} successful: {response.choices[0].message.content.strip()}")
                
                # Add a short pause between requests
                time.sleep(1)
            except Exception as e:
                logger.error(f"Request {i+1} failed: {str(e)}")
        
        # Check the instance status again to see if any failovers occurred
        status_response = requests.get(f"{base_url}/instances/status")
        status_data = status_response.json()
        
        logger.info(f"Final status: {status_data['healthy_instances']}/{status_data['total_instances']} instances healthy")
        
        # Log details for each instance
        for instance in status_data.get('instances', []):
            logger.info(f"Instance: {instance['name']}")
            logger.info(f"  Status: {instance['status']}")
            logger.info(f"  TPM usage: {instance['current_tpm']}/{instance['max_tpm']} ({instance['tpm_usage_percent']}%)")
            
        return True
    except Exception as e:
        logger.error(f"Error testing failover functionality: {str(e)}")
        return False

def test_health_endpoint(base_url):
    """Test the health endpoint."""
    try:
        import requests
        base_url_root = base_url.rsplit('/v1', 1)[0]  # Remove '/v1' to get to the root URL
        
        # Make a request to the health endpoint
        logger.info("Testing health endpoint")
        response = requests.get(f"{base_url_root}/health")
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Failed to get health status: {response.status_code} - {response.text}")
            return False
        
        # Parse the response
        health_data = response.json()
        
        # Log the health status
        logger.info(f"Health status: {health_data['status']}")
        logger.info(f"Message: {health_data['message']}")
        if 'instance_summary' in health_data:
            logger.info(f"Instances: {health_data['instance_summary']['healthy']}/{health_data['instance_summary']['total']} healthy")
        
        return True
    except Exception as e:
        logger.error(f"Error testing health endpoint: {str(e)}")
        return False

def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description="Test the Azure OpenAI Proxy")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="Base URL of the proxy server")
    parser.add_argument("--chat-model", default="gpt-4", help="Model to use for chat completions test")
    parser.add_argument("--completion-model", default="text-davinci-003", help="Model to use for completions test")
    parser.add_argument("--embedding-model", default="text-embedding-ada-002", help="Model to use for embeddings test")
    parser.add_argument("--test", 
                       choices=["chat", "completions", "embeddings", "streaming", "rate-limiting", 
                                "instances-status", "failover", "health", "all"], 
                       default="all", 
                       help="Specific test to run")
    args = parser.parse_args()
    
    # Check if OpenAI package is installed
    try:
        import openai
    except ImportError:
        logger.error("OpenAI package not found. Please install it with: pip install openai")
        return
    
    # Check if requests package is installed for status and health tests
    if args.test in ["instances-status", "failover", "health", "all"]:
        try:
            import requests
        except ImportError:
            logger.error("Requests package not found. Please install it with: pip install requests")
            return
    
    # Run selected tests
    if args.test == "chat" or args.test == "all":
        test_chat_completions(args.url, args.chat_model)
    
    if args.test == "completions" or args.test == "all":
        test_completions(args.url, args.completion_model)
    
    if args.test == "embeddings" or args.test == "all":
        test_embeddings(args.url, args.embedding_model)
    
    if args.test == "streaming" or args.test == "all":
        test_streaming(args.url, args.chat_model)
    
    if args.test == "rate-limiting" or args.test == "all":
        test_rate_limiting(args.url, args.chat_model)
    
    if args.test == "instances-status" or args.test == "all":
        test_instances_status(args.url)
    
    if args.test == "failover" or args.test == "all":
        test_failover(args.url, args.chat_model)
    
    if args.test == "health" or args.test == "all":
        test_health_endpoint(args.url)

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    main()
