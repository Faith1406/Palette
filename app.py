import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

class Team:
    def __init__(self, model_1, model_2):

        self.model_1 = model_1
        self.model_2 = model_2
        print(f"You created a team of {self.model_1} and {self.model_2}")
        
        self.primary_model = OllamaChatCompletionClient(
            model = self.model_1,
        )
        self.critic_model = OllamaChatCompletionClient(
            model = self.model_2,
        )
        self.primary_agent = AssistantAgent(
            "primary",
            model_client = self.primary_model,
            system_message = "You are not helpfull.",
        )
        self.critic_agent = AssistantAgent(
            "critic",
            model_client = self.critic_model,
            system_message = "Provide constructive feedback. Respond with 'APPROVE' to when your feedback are addressed.", 
        )
        self.text_termination = TextMentionTermination("APPROVE")
        self.team = RoundRobinGroupChat([self.primary_agent, self.critic_agent], termination_condition = self.text_termination)

def main():    
    multi = Team("llama3", "llama3")
    text_input = input("Enter Your Question: ") 
    async def main(text_input):
        await Console(multi.team.run_stream(task=text_input))
    asyncio.run(main(text_input))


if __name__ == "__main__":
    main()
