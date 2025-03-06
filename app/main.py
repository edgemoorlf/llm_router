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

# Load environment variables
load_dotenv()

# Configure logging
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create logs directory if it doesn't exist
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
os.makedirs(log_dir, exist_ok=True)

# Configure logging FIRST
logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            maxBytes=1024*1024*5,
            backupCount=3
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
                
            # # Format thez log message according to Feishu's requirements
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
feishu_url = os.getenv("FEISHU_WEBHOOK_URL")
if feishu_url:
    feishu_handler = FeishuHandler(feishu_url)
    feishu_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    # Add to root logger to ensure propagation to all child loggers
    root_logger.addHandler(feishu_handler)
    # Also add to file handler for consistency
    logging.getLogger().addHandler(feishu_handler)

# Create FastAPI app
app = FastAPI(
    title="Azure OpenAI Proxy",
    description="A proxy service to convert OpenAI API calls to Azure OpenAI API calls with rate limiting",
    version="0.1.0",
)

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

# Import routers after app is created to avoid circular imports
from app.routers import openai_proxy

# Include routers
app.include_router(openai_proxy.router)

@app.get("/")
async def root():
    return {
        "message": "Azure OpenAI Proxy API is running",
        "docs": "/docs",
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint that returns basic status information.
    Also checks the status of all Azure OpenAI instances.
    """
    from app.utils.instance_manager import instance_manager
    
    try:
        # Get instance stats
        instance_stats = instance_manager.get_instance_stats()
        total_instances = len(instance_stats)
        healthy_instances = len([i for i in instance_stats if i["status"] == "healthy"])
        
        # Check if we have enough healthy instances
        if total_instances == 0:
            status = "warning"
            message = "No Azure OpenAI instances configured"
        elif healthy_instances == 0:
            status = "unhealthy"
            message = "No healthy Azure OpenAI instances available"
        elif healthy_instances < total_instances:
            status = "degraded"
            message = f"{healthy_instances}/{total_instances} instances healthy"
        else:
            status = "healthy"
            message = "All instances operational"
        
        return {
            "status": status,
            "message": message,
            "timestamp": int(time.time()),
            "instance_summary": {
                "total": total_instances,
                "healthy": healthy_instances,
                "routing_strategy": os.getenv("API_ROUTING_STRATEGY", "failover")
            },
            "version": app.version
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error performing health check: {str(e)}",
            "timestamp": int(time.time()),
            "version": app.version
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3010))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
