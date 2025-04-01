import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

class Autogen:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OllamaChatCompletionClient(model=self.model_name)

    async def ask(self, message: str):
        response = await self.client.create([
            UserMessage(content=message, source="user")
        ])
        return response


if __name__ == "__main__":
    async def main():
        bot = Autogen("llama3")
        question = input("Enter your question: ")
        response = await bot.ask(question)
        print(response.strip())
    asyncio.run(main())
