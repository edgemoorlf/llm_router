import tiktoken
import logging

logger = logging.getLogger(__name__)

def get_encoder_for_model(model: str, provider: str = "azure") -> tiktoken.Encoding:
    """Get the appropriate token encoder for the model and provider."""
    try:
        # Handle model name mappings for different providers
        if provider == "azure":
            model_mapping = {
                "gpt-4": "gpt-4-0613",
                "gpt-4o": "gpt-4o-2024-05-13",  # Updated to official GPT-4o encoder
                "gpt-35-turbo": "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo": "gpt-3.5-turbo-0613" 
            }
            model = model_mapping.get(model.lower(), model)
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def estimate_chat_tokens(messages, functions, model: str, provider: str = "azure") -> int:
    """Estimate token count for chat completion requests."""
    encoder = get_encoder_for_model(model, provider)
    # Adjust tokens per message based on model
    if "gpt-4" in model.lower():
        tokens_per_message = 4  # GPT-4 uses 4 tokens per message
    else:
        tokens_per_message = 3  # GPT-3.5 uses 3 tokens per message
    tokens_per_function = 6  # function_call + name + arguments
    
    token_count = 0
    for message in messages:
        token_count += tokens_per_message
        for key, value in message.items():
            if key == "content" and isinstance(value, str):
                # Special handling for long text content
                token_count += len(encoder.encode(value, disallowed_special=()))
            elif isinstance(value, list):  # Handle multimodal content
                for item in value:
                    token_count += len(encoder.encode(str(item), disallowed_special=()))
            else:
                token_count += len(encoder.encode(str(value), disallowed_special=()))
    
    if functions:
        for function in functions:
            token_count += tokens_per_function
            for key, value in function.items():
                token_count += len(encoder.encode(str(value), disallowed_special=()))
    
    # Add per-request overhead
    token_count += 3  # every reply is primed with <|start|>assistant<|message|>

    logger.debug(f"token count: {token_count}")
    
    return max(1, token_count)  # Ensure at least 1 token

def estimate_completion_tokens(prompt, model: str, provider: str = "azure") -> int:
    """Estimate token count for text completion requests."""
    encoder = get_encoder_for_model(model, provider)
    return len(encoder.encode(prompt, disallowed_special=()))
