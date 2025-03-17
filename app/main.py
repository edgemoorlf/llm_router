import os
import uuid
import time
import logging
import requests
import json
from threading import Thread
from fastapi import FastAPI, Depends, Request
from dotenv import load_dotenv
import uvicorn
import argparse


# Load environment variables
load_dotenv()

# Import configuration system
from app.config import config_loader
from app.instance.manager import InstanceManager

# Load configuration
config = config_loader.load_config()

# Initialize the instance manager
instance_manager = InstanceManager()

# Configure logging
log_level = getattr(logging, config.logging.level)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(config.logging.file)
if not os.path.isabs(log_dir):
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), log_dir))
os.makedirs(log_dir, exist_ok=True)

# Configure logging FIRST
logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            config.logging.file if os.path.isabs(config.logging.file) else os.path.join(log_dir, os.path.basename(config.logging.file)),
            maxBytes=config.logging.max_size,
            backupCount=config.logging.backup_count
        )
    ]
)

# THEN get logger instance
logger = logging.getLogger(__name__)

# Get root logger and configure propagation
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

# Add Feishu webhook handler if configured
class FeishuHandler(logging.Handler):
    def __init__(self, webhook_url: str):
        super().__init__()
        self.setFormatter(logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.webhook_url = webhook_url
        self.setLevel(logging.ERROR)

    def emit(self, record):
        try:
            if not self.webhook_url:
                return
                
            # Format the log message according to Feishu's requirements
            log_data = {
                "msg_type": "text",
                "content": {"text": self.format(record)}
            }

            # Send in background thread to avoid blocking
            Thread(target=self._send_alert, args=(log_data,)).start()
            
        except Exception as e:
            print(f"Failed to send Feishu alert: {str(e)}")

    def _send_alert(self, data: dict):
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.webhook_url,
                data=json.dumps(data),
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Feishu API error: {str(e)}")

# Add Feishu handler if configured
if config.logging.feishu_webhook:
    feishu_handler = FeishuHandler(config.logging.feishu_webhook)
    feishu_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    # Add to root logger to ensure propagation to all child loggers
    root_logger.addHandler(feishu_handler)
    # Also add to file handler for consistency
    logging.getLogger().addHandler(feishu_handler)

# Import instance manager and router from the context module
from app.instance.instance_context import instance_manager, instance_router, check_for_updates

# Create FastAPI app
app = FastAPI(
    title=config.name,
    description="A proxy service to convert OpenAI API calls to Azure OpenAI API calls with rate limiting",
    version=config.version,
)

# Import and register exception handlers
from app.errors.handlers import register_exception_handlers
register_exception_handlers(app)

# Add middleware for request/response logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests and responses for debugging."""
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    
    # Log request details
    path = request.url.path
    method = request.method
    logger.info(f"Request {request_id} - {method} {path}")
    
    # Process the request
    try:
        response = await call_next(request)
        
        # Log response status
        logger.info(f"Response {request_id} - Status: {response.status_code}")
        
        return response
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        raise

@app.middleware("http")
async def check_instance_updates_middleware(request: Request, call_next):
    """Middleware to check for instance updates before processing each request."""
    # Check for updates from the shared state file
    check_for_updates()
    
    # Process the request
    response = await call_next(request)
    return response

# Import router modules
from app.routers import openai_proxy
from app.routers import stats
from app.routers import config as config_router
from app.routers import instance_management
from app.routers import admin

# Include routers
app.include_router(openai_proxy.router)
app.include_router(stats.router)
app.include_router(config_router.router)
app.include_router(instance_management.router)
app.include_router(admin.router)

@app.get("/")
async def root():
    return {
        "message": f"{config.name} API is running",
        "docs": "/docs",
        "version": config.version,
        "config_endpoints": {
            "configuration": "/config",
            "instances": "/config/instances",
            "instance_details": "/config/instances?detailed=true",
            "reload": "/config/reload"
        }
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Azure OpenAI Proxy Server")
    parser.add_argument("--port", type=int, default=config.port, help="Port to bind the server to")
    args = parser.parse_args()
    
    # Use the port from command line arguments or config
    port = args.port
    
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
