import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

async def main() -> None:
ollama_client = OllamaChatCompletionClient(model="llama3.2")

response = await ollama_client.create([UserMessage(content="What is the capital of France?", source="user")])

print(response)
