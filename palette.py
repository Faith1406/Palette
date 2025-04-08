import asyncio
#this is the model factory 
from model_factory import get_model_client 
#other team chats imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

class Palette:
    def __init__(self,provider_1=None, provider_2=None, agent_1=None, agent_2=None, system_message_1=None, system_message_2=None, description_1=None, description_2=None, external_termination=None, termination_text="Approve", api_key_1=None, api_key_2=None, behaviour_1="primary", behaviour_2="critic",config=None):
        if config:
            provider_1 = provider_1 or config.get("provider_1")
            provider_2 = provider_2 or config.get("provider_2")
            agent_1 = agent_1 or config.get("agent_1")
            agent_2 = agent_2 or config.get("agent_2")
            api_key_1 = api_key_1 or config.get("api_key_1")
            api_key_2 = api_key_2 or config.get("api_key_2")
            behaviour_1 = behaviour_1 or config.get("behaviour_1", "primary")
            behaviour_2 = behaviour_2 or config.get("behaviour_2", "critic")
            description_1 = description_1 or config.get("description_1")
            description_2 = description_2 or config.get("description_2")
            system_message_1 = system_message_1 or config.get("system_message_1")
            system_message_2 = system_message_2 or config.get("system_message_2")
            termination_text = termination_text or config.get("termination_text", "Approve")
            external_termination = external_termination or config.get("external_termination")

        required_fields = {
            "provider_1": provider_1,
            "provider_2": provider_2,
            "agent_1": agent_1,
            "agent_2": agent_2,
            "description_1": description_1,
            "description_2": description_2,
            "system_message_1": system_message_1,
            "system_message_2": system_message_2,
        }

        missing = [k for k, v in required_fields.items() if not v]
        if missing:
            raise ValueError(f"Missing required config fields: {', '.join(missing)}")

        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.provider_1 = provider_1
        self.provider_2 = provider_2
        self.termination_text = termination_text
        self.external_termination = ExternalTermination()
        self.api_key_1 = api_key_1
        self.api_key_2 = api_key_2
        self.behaviour_1 = behaviour_1 
        self.behaviour_2 = behaviour_2
        self.description_1 = description_1
        self.description_2 = description_2 
        self.system_message_1= system_message_1 
        self.system_message_2= system_message_2 
        
        self.primary_model = get_model_client(
            provider = self.provider_1,
            model_name = self.agent_1,
            api_key = self.api_key_1,
        )
        self.secondary_model = get_model_client(
            provider = self.provider_2,
            model_name = self.agent_2,
            api_key = self.api_key_2,
        )
        self.primary_agent = AssistantAgent(
            self.behaviour_1,
            model_client = self.primary_model,
            description = self.description_1,
            system_message = self.system_message_1,
        )
        self.secondary_agent = AssistantAgent(
            self.behaviour_2,
            model_client = self.secondary_model,
            description = self.description_2,
            system_message = self.system_message_2, 
        )
        self.text_termination = TextMentionTermination(self.termination_text)
        self.team = RoundRobinGroupChat([self.primary_agent, self.secondary_agent], termination_condition = self.external_termination | self.text_termination)


    async def print_convo(self, text_input:str):
        await Console(self.team.run_stream(task=text_input))

    def run_team(self, text_input:str):
        asyncio.run(self.print_convo(text_input))

    def display_team_members(self):
        print(f"You created the team of {self.agent_1} and {self.agent_2}")

    async def resetting_team(self):
        await self.team.reset()

    def stopping_team(self):
        self.external_termination.set()
        
    def token_used(self):
        pass

    async def resume_team(self, new_text_input):
        await Console(self.team.run_stream(new_text_input))

    
