from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion

class ModelProvider:
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"

def get_model_client(provider:str, model_name:str, end_point:str="that was not neccessary", api_key:str="that was not neccessary", max_tokens=10):
    if provider == ModelProvider.OLLAMA:
        return OllamaChatCompletionClient(model=model_name, max_tokens=max_tokens)
    elif provider == ModelProvider.OPENAI:
        if not api_key: 
            raise ValueError("you did'nt add correct end_point or api_key")
        return OpenAIChatCompletionClient(
            model=model_name,
            api_key = api_key,
            max_tokens=max_tokens
        )
    elif provider == ModelProvider.AZURE_OPENAI:
        if not end_point or api_key: 
            raise ValueError("you did'nt add correct end_point or api_key")
        return AzureOpenAIChatCompletionClient(
                model=model_name, 
                azure_endpoint=end_point,
                api_key=api_key,
                max_tokens=max_tokens
            )
    elif provider == ModelProvider.ANTHROPIC:
        if not api_key: 
            raise ValueError("you did'nt add correct end_point or api_key")
        return AnthropicChatCompletion(
            model=model_name,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
