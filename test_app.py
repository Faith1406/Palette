import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient


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

async def main():
    await Console(team.run_stream(task="Current IPL Leaderboard"))

asyncio.run(main())
