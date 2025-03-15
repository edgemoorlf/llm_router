"""Authentication for administrative API endpoints."""
import os
import logging
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# Use environment variable or a default for admin token
ADMIN_TOKEN = os.environ.get("ADMIN_API_KEY", "admin-default-key-change-me")

# Define API key header
api_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)

async def verify_admin_token(api_key: str = Depends(api_key_header)):
    """
    Verify that the request includes a valid admin API key.
    
    Args:
        api_key: Admin API key from the X-Admin-API-Key header
        
    Raises:
        HTTPException: If the API key is invalid or missing
    """
    if not api_key:
        logger.warning("Admin API request missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin API key"
        )
        
    if api_key != ADMIN_TOKEN:
        logger.warning("Invalid admin API key used in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key"
        )
        
    # API key is valid
    return True 