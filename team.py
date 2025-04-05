import asyncio
import os
from dotenv import load_dotenv
#this is the model factory 
from model_factory import get_model_client as gt 
#other team chats imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

class Team:
    def __init__(self,provider_1, provider_2, model_1, model_2, system_message_1, system_message_2, description_1, description_2, api_key_1=None, api_key_2=None, behaviour_1="primary", behaviour_2="critic",):

        self.model_1 = model_1
        self.model_2 = model_2
        self.provider_1 = provider_1
        self.provider_2 = provider_2
        self.api_key_1 = api_key_1
        self.api_key_2 = api_key_2
        self.behaviour_1 = behaviour_1 
        self.behaviour_2 = behaviour_2
        self.description_1 = description_1
        self.description_2 = description_2 
        self.system_message_1= system_message_1 
        self.system_message_2= system_message_2 
        print(f"You created a team of {self.model_1} and {self.model_2}")
        
        self.primary_model = gt(
            provider = self.provider_1,
            model_name = self.model_1,
            api_key = self.api_key_1,
        )
        self.secondary_model = gt(
            provider = self.provider_2,
            model_name = self.model_2,
            api_key = self.api_key_2,
        )
        self.primary_agent = AssistantAgent(
            self.behaviour_1,
            model_client = self.primary_model,
            description = self.description_1,
            system_message = self.system_message_1,
        )
        self.critic_agent = AssistantAgent(
            self.behaviour_2,
            model_client = self.secondary_model,
            description = self.description_2,
            system_message = self.system_message_2, 
        )
        self.text_termination = TextMentionTermination("APPROVE")
        self.team = RoundRobinGroupChat([self.primary_agent, self.critic_agent], termination_condition = self.text_termination)

def main():    
    load_dotenv()
    print(os.getenv("API_KEY"))
    api_key = os.getenv("API_KEY")
    multi = Team(
                "openai", 
                "ollama",  
                "gemini-1.5-flash-8b", 
                "llama3", 
                system_message_1="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
                system_message_2="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
                description_1="A helpful assistant that can plan trips.",
                description_2="A helpful assistant that can provide language tips for a given destination.",
                api_key_1=api_key, 
                behaviour_1="Planner", 
                behaviour_2="language_agent",
            )
    text_input = input("Enter Your Question: ") 
    async def run_team(text_input):
        await Console(multi.team.run_stream(task=text_input))
    asyncio.run(run_team(text_input))


if __name__ == "__main__":
    main()
