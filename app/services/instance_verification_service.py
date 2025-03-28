"""Service for verifying and testing instance functionality."""
import logging
import time
from typing import Dict, Any, Optional

from app.instance.instance_context import instance_manager
from app.services.instance_service import instance_service
from app.errors.internal_errors import InstanceNotFoundError
from app.errors.handlers import handle_errors

logger = logging.getLogger(__name__)

class InstanceVerificationService:
    """Service for verifying and testing instance functionality."""
    
    @handle_errors
    async def verify_instance(self, 
                             instance_name: str, 
                             model_to_test: Optional[str] = None,
                             message_to_test: str = "This is a test message to verify instance functionality."
                             ) -> Dict[str, Any]:
        """
        Verify an instance by running a comprehensive set of checks.
        
        This performs a full verification of an instance including:
        - Configuration check
        - Connectivity test
        - Model support check
        - Actual model testing
        
        Args:
            instance_name: Name of the instance to verify
            model_to_test: Model to test, if None will use first supported model
            message_to_test: Test message to send to the model
            
        Returns:
            Dictionary with detailed verification results
            
        Raises:
            InstanceNotFoundError: If the instance doesn't exist
        """
        # Get instance config and state
        config = instance_manager.get_instance_config(instance_name)
        state = instance_manager.get_instance_state(instance_name)
        
        if not config:
            raise InstanceNotFoundError(instance_name)
        
        start_time = time.time()
        
        # If no model specified, use the first supported model
        if not model_to_test and config.supported_models:
            model_to_test = config.supported_models[0]
        
        # Initialize test results
        result = {
            "status": "success",
            "instance_name": instance_name,
            "provider_type": config.provider_type,
            "timestamp": int(time.time()),
            "tests": {
                "configuration": {
                    "status": "pending",
                    "details": {}
                },
                "connectivity": {
                    "status": "pending",
                    "details": {}
                },
                "model_support": {
                    "status": "pending",
                    "details": {}
                },
                "model_test": {
                    "status": "pending",
                    "details": {}
                }
            },
            "overall_result": "pending"
        }
        
        # Test 1: Configuration check
        try:
            # Check if configuration is complete
            config_issues = []
            
            if not config.api_key:
                config_issues.append("API key is missing")
            
            if not config.api_base:
                config_issues.append("API base URL is missing")
            
            if config.provider_type == "azure" and not config.api_version:
                config_issues.append("API version is missing for Azure provider")
                
            # For Azure provider, check deployment mappings
            if config.provider_type == "azure" and not config.model_deployments:
                config_issues.append("No model deployment mappings defined for Azure provider")
            
            # Check if any supported models are defined
            if not config.supported_models:
                config_issues.append("No supported models defined")
            
            if config_issues:
                result["tests"]["configuration"]["status"] = "failed"
                result["tests"]["configuration"]["details"] = {
                    "issues": config_issues
                }
            else:
                result["tests"]["configuration"]["status"] = "passed"
                result["tests"]["configuration"]["details"] = {
                    "api_base": config.api_base,
                    "provider_type": config.provider_type,
                    "api_version": config.api_version if config.provider_type == "azure" else "N/A",
                    "supported_models_count": len(config.supported_models),
                    "deployment_mappings_count": len(config.model_deployments) if config.model_deployments else 0
                }
        except Exception as e:
            logger.error(f"Error during configuration check: {str(e)}")
            result["tests"]["configuration"]["status"] = "error"
            result["tests"]["configuration"]["details"] = {"error": str(e)}
        
        # Test 2: Connectivity test
        try:
            connectivity_result = await instance_service.test_connectivity(instance_name)
            result["tests"]["connectivity"] = connectivity_result
        except Exception as e:
            logger.error(f"Error during connectivity test: {str(e)}")
            result["tests"]["connectivity"]["status"] = "error"
            result["tests"]["connectivity"]["details"] = {"error": str(e)}
        
        # Test 3: Model support check
        try:
            if model_to_test:
                can_handle = self._can_handle_model(instance_name, model_to_test)
                result["tests"]["model_support"]["status"] = "passed" if can_handle else "failed"
                result["tests"]["model_support"]["details"] = {
                    "model": model_to_test,
                    "supported": can_handle,
                    "all_supported_models": config.supported_models
                }
                
                # For Azure, include deployment mapping
                if config.provider_type == "azure" and can_handle:
                    deployment = self._get_deployment_for_model(instance_name, model_to_test)
                    result["tests"]["model_support"]["details"]["deployment"] = deployment
            else:
                result["tests"]["model_support"]["status"] = "skipped"
                result["tests"]["model_support"]["details"] = {
                    "reason": "No model specified for testing",
                    "all_supported_models": config.supported_models
                }
        except Exception as e:
            logger.error(f"Error during model support check: {str(e)}")
            result["tests"]["model_support"]["status"] = "error"
            result["tests"]["model_support"]["details"] = {"error": str(e)}
        
        # Test 4: Model test (only if model is specified and supported)
        try:
            if (model_to_test and 
                result["tests"]["connectivity"]["status"] == "passed" and
                result["tests"]["model_support"]["status"] == "passed"):
                
                model_test_result = await self._test_model_call(
                    instance_name,
                    model_to_test,
                    message_to_test
                )
                
                result["tests"]["model_test"] = model_test_result
            else:
                result["tests"]["model_test"]["status"] = "skipped"
                result["tests"]["model_test"]["details"] = {
                    "reason": "Prerequisites not met for model testing"
                }
        except Exception as e:
            logger.error(f"Error during model test: {str(e)}")
            result["tests"]["model_test"]["status"] = "error"
            result["tests"]["model_test"]["details"] = {"error": str(e)}
        
        # Calculate overall result
        test_statuses = [test["status"] for test in result["tests"].values()]
        
        if "error" in test_statuses:
            result["overall_result"] = "error"
        elif "failed" in test_statuses:
            result["overall_result"] = "failed"
        elif "skipped" in test_statuses:
            if "passed" in test_statuses:
                result["overall_result"] = "partial_success"
            else:
                result["overall_result"] = "skipped"
        else:
            result["overall_result"] = "success"
        
        # Add execution time
        result["execution_time"] = round(time.time() - start_time, 2)
        
        return result
    
    def _can_handle_model(self, instance_name: str, model_name: str) -> bool:
        """Check if an instance can handle a specific model."""
        config = instance_manager.get_instance_config(instance_name)
        if not config:
            return False
            
        # If no supported models list, assume it can handle any model
        if not config.supported_models:
            return True
            
        # Check for exact match
        if model_name in config.supported_models:
            return True
            
        # Check for case-insensitive match
        return model_name.lower() in [m.lower() for m in config.supported_models]
    
    def _get_deployment_for_model(self, instance_name: str, model_name: str) -> Optional[str]:
        """Get the deployment name for a specific model."""
        config = instance_manager.get_instance_config(instance_name)
        if not config or not config.model_deployments:
            return None
            
        # Check for exact match
        if model_name in config.model_deployments:
            return config.model_deployments[model_name]
            
        # Check for case-insensitive match
        model_name_lower = model_name.lower()
        for k, v in config.model_deployments.items():
            if k.lower() == model_name_lower:
                return v
                
        return None
    
    @handle_errors
    async def _test_model_call(self, instance_name: str, model_name: str, message: str) -> Dict[str, Any]:
        """Test a model call with a simple message."""
        # Get the instance config
        config = instance_manager.get_instance_config(instance_name)
        if not config:
            raise InstanceNotFoundError(instance_name)
        
        # Build a simple test payload
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 50
        }
        
        # Start timing
        start_time = time.time()
        
        try:
            # Use the instance service to make the test call
            response = await instance_service.test_model(instance_name, payload)
            
            # Calculate latency
            latency = round((time.time() - start_time) * 1000, 2)  # Convert to ms
            
            return {
                "status": "passed",
                "details": {
                    "latency_ms": latency,
                    "response": response
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "details": {
                    "error": str(e)
                }
            }

# Create a singleton instance
instance_verification_service = InstanceVerificationService() 