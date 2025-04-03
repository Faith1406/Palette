import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

class OpenAI:

class AzureOpenAI:

class AzureAIFoundry:

class Anthropic:

class Ollama:

class Gemnini:


class MainTeam:
    model_client = OllamaChatCompletionClient(
        model = "llama3",
    )
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are not helpfull.",
    )
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedback are addressed.", 
    )
    text_termination = TextMentionTermination("APPROVE")

    team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

if __name__ == "__main__":
    async def main():
        result = await team.run(task="Who is the president of India?")
        print(result)
    asyncio.run(main())
