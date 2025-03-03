from typing import Dict, Optional

class EndpointMapper:
    """Maps standard OpenAI endpoints to provider-specific endpoints"""
    
    def __init__(self):
        self.templates = {
            'azure': {
                'chat': '/openai/deployments/{deployment}/chat/completions',
                'completions': '/openai/deployments/{deployment}/completions',
                'embeddings': '/openai/deployments/{deployment}/embeddings'
            },
            'generic': {
                'chat': '/v1/chat/completions',
                'completions': '/v1/completions', 
                'embeddings': '/v1/embeddings'
            }
        }
        
    def get_endpoint(self, 
                    provider: str,
                    endpoint_type: str,
                    deployment: Optional[str] = None) -> str:
        """Get endpoint path for given provider and endpoint type"""
        if provider not in self.templates:
            raise ValueError(f"Unsupported provider: {provider}")
            
        template = self.templates[provider][endpoint_type]
        
        if provider == 'azure' and not deployment:
            raise ValueError("Azure endpoints require deployment name")
            
        return template.format(deployment=deployment) if deployment else template
