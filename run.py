#!/usr/bin/env python
"""
Entry point script for Azure OpenAI Proxy.

This script provides a convenient way to start the proxy server.
"""
import os
import argparse
import uvicorn
from dotenv import load_dotenv

def main():
    """Run the Azure OpenAI Proxy server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Azure OpenAI Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind the server to (overrides .env)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--env-file", default="app/.env", help="Path to .env file")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(args.env_file)
    
    # Get port from arguments or environment variables
    port = args.port or int(os.getenv("PORT", 8000))
    
    # Print startup message
    print(f"Starting Azure OpenAI Proxy server on {args.host}:{port}")
    print(f"API documentation will be available at http://{args.host}:{port}/docs")
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
