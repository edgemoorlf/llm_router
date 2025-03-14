"""Instance verification API router."""
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Body
import time

from app.instance.manager import instance_manager
from app.services.instance_service import instance_service
from app.instance.forwarder import instance_forwarder

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/verification",
    tags=["Instance Verification"],
    responses={404: {"description": "Not found"}},
)

@router.post("/instances/{instance_name}")
async def verify_instance(
    instance_name: str,
    request: Optional[Dict[str, Any]] = Body(None, description="Optional request payload with model and message to use for testing")
) -> Dict[str, Any]:
    """
    Verify an instance by running a comprehensive set of checks.
    
    This endpoint performs a full verification of an instance including:
    - Configuration check
    - Connectivity test
    - Model support check
    - Actual model testing
    
    For a quicker, basic health check, use the 
    `/health/instances/{instance_name}` endpoint instead.
    
    Args:
        instance_name: Name of the instance to verify
        request: Optional request payload with model and message to use for testing
        
    Returns:
        Standardized response with detailed verification results
    """
    try:
        # Check if instance exists
        instance = instance_manager.get_instance(instance_name)
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "error",
                    "message": f"Instance '{instance_name}' not found",
                    "timestamp": int(time.time()),
                    "data": None
                }
            )
            
        # Set up verification steps
        verification_steps = {}
        
        # Step 1: Configuration check
        verification_steps["configuration_check"] = {
            "status": "success",
            "message": "Instance configuration is valid",
            "details": {
                "provider_type": instance.provider_type,
                "api_base": instance.api_base,
                "supported_models": instance.supported_models,
                "model_deployments": instance.model_deployments
            }
        }
        
        # Step 2: Connectivity test
        try:
            connectivity_test = await instance_service.test_instance(instance_name)
            verification_steps["connectivity_test"] = {
                "status": connectivity_test["status"],
                "message": connectivity_test["message"],
                "details": connectivity_test
            }
        except Exception as e:
            verification_steps["connectivity_test"] = {
                "status": "error",
                "message": f"Error testing connectivity: {str(e)}",
                "details": {"error": str(e)}
            }
            
        # Skip further steps if connectivity test failed
        if verification_steps["connectivity_test"]["status"] != "success":
            # Return early with standardized format
            return {
                "status": "error",
                "message": f"Instance verification failed at connectivity test",
                "timestamp": int(time.time()),
                "data": {
                    "name": instance_name,
                    "provider_type": instance.provider_type,
                    "api_base": instance.api_base,
                    "verification_timestamp": int(time.time()),
                    "verification_steps": verification_steps
                }
            }
        
        # Step 3: Model support check
        request_model = request.get("model") if request else None
        if not request_model and instance.supported_models:
            request_model = instance.supported_models[0]
            
        if request_model:
            # Check if the model is supported by the instance
            model_supported = False
            
            # Check in supported_models (exact match)
            if instance.supported_models:
                model_supported = request_model in instance.supported_models
                
            # Check in model_deployments (key match)
            if not model_supported and instance.model_deployments:
                model_supported = request_model in instance.model_deployments
                
            verification_steps["model_support_check"] = {
                "status": "success" if model_supported else "error",
                "message": f"Model '{request_model}' is {'supported' if model_supported else 'not supported'} by this instance",
                "details": {
                    "model": request_model,
                    "supported_models": instance.supported_models,
                    "model_deployments": instance.model_deployments
                }
            }
            
            # Skip model test if model is not supported
            if not model_supported:
                # Return early with standardized format
                return {
                    "status": "error",
                    "message": f"Instance verification failed: model '{request_model}' not supported",
                    "timestamp": int(time.time()),
                    "data": {
                        "name": instance_name,
                        "provider_type": instance.provider_type,
                        "api_base": instance.api_base,
                        "verification_timestamp": int(time.time()),
                        "verification_steps": verification_steps
                    }
                }
                
        # Step 4: Model test with a simple message
        if request_model:
            test_message = request.get("message", "Hello, this is a test")
            
            try:
                # Construct a minimal chat payload
                test_payload = {
                    "model": request_model,
                    "messages": [{"role": "user", "content": test_message}],
                    "max_tokens": 50
                }
                
                # Determine the endpoint and deployment
                endpoint = "/v1/chat/completions"
                deployment = ""
                if instance.provider_type == "azure" and instance.model_deployments and request_model in instance.model_deployments:
                    deployment = instance.model_deployments[request_model]
                
                # Make the test request
                response = await instance_forwarder.try_specific_instance(
                    instance_name, 
                    endpoint, 
                    deployment, 
                    test_payload
                )
                
                # Process the response
                if "error" in response:
                    verification_steps["model_test"] = {
                        "status": "error",
                        "message": f"Error from model: {response['error'].get('message', 'Unknown error')}",
                        "details": response["error"]
                    }
                else:
                    # Extract the response content
                    content = "No content returned"
                    if response.get("choices") and len(response["choices"]) > 0:
                        first_choice = response["choices"][0]
                        if "message" in first_choice and "content" in first_choice["message"]:
                            content = first_choice["message"]["content"]
                            
                    verification_steps["model_test"] = {
                        "status": "success",
                        "message": "Successfully received response from model",
                        "details": {
                            "model": request_model,
                            "content": content,
                            "usage": response.get("usage", {})
                        }
                    }
            except Exception as e:
                verification_steps["model_test"] = {
                    "status": "error",
                    "message": f"Error testing model: {str(e)}",
                    "details": {"error": str(e)}
                }
        
        # Determine overall status
        overall_status = "success"
        for step in verification_steps.values():
            if step["status"] != "success":
                overall_status = "error"
                break
        
        # Return standardized response format
        return {
            "status": overall_status,
            "message": f"Instance '{instance_name}' verification completed with status: {overall_status}",
            "timestamp": int(time.time()),
            "data": {
                "name": instance_name,
                "provider_type": instance.provider_type,
                "api_base": instance.api_base,
                "verification_timestamp": int(time.time()),
                "verification_steps": verification_steps
            }
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions but use standardized format
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail if isinstance(e.detail, dict) else {
                "status": "error",
                "message": str(e.detail),
                "timestamp": int(time.time()),
                "data": None
            }
        )
    except Exception as e:
        logger.error(f"Error verifying instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Error verifying instance: {str(e)}",
                "timestamp": int(time.time()),
                "data": None
            }
        )

@router.post("/instances/test")
async def test_instance_config(instance_config: Any) -> Dict[str, Any]:
    """
    Test a new instance configuration before adding it to the proxy service.
    
    This endpoint is used for testing new instance configurations that are not yet
    added to the system. It creates a temporary instance and tests the connection.
    
    NOTE: For testing existing instances, use:
    - `/health/instances/{instance_name}?test_connection=true` for a quick connection test
    - `/verification/instances/{instance_name}` for comprehensive verification
    
    Args:
        instance_config: Instance configuration to test
        
    Returns:
        Test results with status
    """
    try:
        return await instance_service.test_instance(instance_config)
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error testing instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing instance: {str(e)}"
        ) 