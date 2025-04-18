import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from model_factory import get_model_client


class Palette:
    def __init__(
        self,
        provider_1=None,
        provider_2=None,
        provider_3=None,
        provider_4=None,
        agent_1=None,
        agent_2=None,
        agent_3=None,
        agent_4=None,
        system_message_1=None,
        system_message_2=None,
        system_message_3=None,
        system_message_4=None,
        max_tokens_1=None,
        max_tokens_2=None,
        max_tokens_3=None,
        max_tokens_4=None,
        description_1=None,
        description_2=None,
        description_3=None,
        description_4=None,
        external_termination=None,
        termination_text="Approve",
        api_key_1=None,
        api_key_2=None,
        api_key_3=None,
        api_key_4=None,
        behaviour_1="primary",
        behaviour_2="critic",
        behaviour_3="advisor",
        behaviour_4="executor",
        token_threshold=150,
        config=None,
    ):
        if config:
            provider_1 = provider_1 or config.get("provider_1")
            provider_2 = provider_2 or config.get("provider_2")
            provider_3 = provider_3 or config.get("provider_3")
            provider_4 = provider_4 or config.get("provider_4")
            agent_1 = agent_1 or config.get("agent_1")
            agent_2 = agent_2 or config.get("agent_2")
            agent_3 = agent_3 or config.get("agent_3")
            agent_4 = agent_4 or config.get("agent_4")
            api_key_1 = api_key_1 or config.get("api_key_1")
            api_key_2 = api_key_2 or config.get("api_key_2")
            api_key_3 = api_key_3 or config.get("api_key_3")
            api_key_4 = api_key_4 or config.get("api_key_4")
            behaviour_1 = behaviour_1 or config.get("behaviour_1", "primary")
            behaviour_2 = behaviour_2 or config.get("behaviour_2", "critic")
            behaviour_3 = behaviour_3 or config.get("behaviour_3", "advisor")
            behaviour_4 = behaviour_4 or config.get("behaviour_4", "executor")
            description_1 = description_1 or config.get("description_1")
            description_2 = description_2 or config.get("description_2")
            description_3 = description_3 or config.get("description_3")
            description_4 = description_4 or config.get("description_4")
            system_message_1 = system_message_1 or config.get("system_message_1")
            system_message_2 = system_message_2 or config.get("system_message_2")
            system_message_3 = system_message_3 or config.get("system_message_3")
            system_message_4 = system_message_4 or config.get("system_message_4")
            termination_text = termination_text or config.get(
                "termination_text", "Approve"
            )
            external_termination = external_termination or config.get(
                "external_termination"
            )
            max_tokens_1 = max_tokens_1 or config.get("max_tokens_1")
            max_tokens_2 = max_tokens_2 or config.get("max_tokens_2")
            max_tokens_3 = max_tokens_3 or config.get("max_tokens_3")
            max_tokens_4 = max_tokens_4 or config.get("max_tokens_4")

        if not (agent_1 and provider_1 and agent_2 and provider_2):
            raise ValueError("At least two agents with their providers must be defined")

        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.agent_3 = agent_3
        self.agent_4 = agent_4
        self.provider_1 = provider_1
        self.provider_2 = provider_2
        self.provider_3 = provider_3
        self.provider_4 = provider_4
        self.termination_text = termination_text
        self.external_termination = ExternalTermination()
        self.api_key_1 = api_key_1
        self.api_key_2 = api_key_2
        self.api_key_3 = api_key_3
        self.api_key_4 = api_key_4
        self.behaviour_1 = behaviour_1
        self.behaviour_2 = behaviour_2
        self.behaviour_3 = behaviour_3
        self.behaviour_4 = behaviour_4
        self.description_1 = description_1
        self.description_2 = description_2
        self.description_3 = description_3
        self.description_4 = description_4
        self.system_message_1 = system_message_1
        self.system_message_2 = system_message_2
        self.system_message_3 = system_message_3
        self.system_message_4 = system_message_4
        self.max_tokens_1 = max_tokens_1
        self.max_tokens_2 = max_tokens_2
        self.max_tokens_3 = max_tokens_3
        self.max_tokens_4 = max_tokens_4
        self.token_threshold = token_threshold

        self.agents = []

        self.primary_model = get_model_client(
            provider=self.provider_1,
            model_name=self.agent_1,
            api_key=self.api_key_1,
            max_tokens=self.max_tokens_1,
        )
        self.secondary_model = get_model_client(
            provider=self.provider_2,
            model_name=self.agent_2,
            api_key=self.api_key_2,
            max_tokens=self.max_tokens_2,
        )

        self.primary_agent = AssistantAgent(
            self.behaviour_1,
            model_client=self.primary_model,
            description=self.description_1,
            system_message=self.system_message_1,
        )
        self.secondary_agent = AssistantAgent(
            self.behaviour_2,
            model_client=self.secondary_model,
            description=self.description_2,
            system_message=self.system_message_2,
        )

        self.agents = [self.primary_agent, self.secondary_agent]

        if agent_3 and provider_3:
            self.third_model = get_model_client(
                provider=self.provider_3,
                model_name=self.agent_3,
                api_key=self.api_key_3,
                max_tokens=self.max_tokens_3,
            )
            self.third_agent = AssistantAgent(
                self.behaviour_3,
                model_client=self.third_model,
                description=self.description_3,
                system_message=self.system_message_3,
            )
            self.agents.append(self.third_agent)

        if agent_4 and provider_4:
            self.fourth_model = get_model_client(
                provider=self.provider_4,
                model_name=self.agent_4,
                api_key=self.api_key_4,
                max_tokens=self.max_tokens_4,
            )
            self.fourth_agent = AssistantAgent(
                self.behaviour_4,
                model_client=self.fourth_model,
                description=self.description_4,
                system_message=self.system_message_4,
            )
            self.agents.append(self.fourth_agent)

        self.text_termination = TextMentionTermination(self.termination_text)
        self.team = RoundRobinGroupChat(
            self.agents,
            termination_condition=self.text_termination,
        )

    async def print_convo_and_count_tokens(self, text_input: str):
        full_output = ""
        retries = 0

        while retries < 3:
            async for message in self.team.run_stream(task=text_input):
                if hasattr(message, "source") and hasattr(message, "content"):
                    print(f"[{message.source}]: {message.content}")
                    full_output += message.content + " "
                else:
                    print(message)

            word_count = len(full_output.split())
            estimated_tokens = int(word_count * 1.3)
            print(f"Estimated Total Tokens Used: {estimated_tokens}")

            if estimated_tokens >= self.token_threshold:
                print("Token limit exceeded, expanding team...")
                await self.create_new_team()
                retries += 1
                full_output = ""
            else:
                break

    def run_team(self, text_input: str):
        asyncio.run(self.print_convo_and_count_tokens(text_input))

    def display_team_members(self):
        team_members = [self.agent_1, self.agent_2]
        if hasattr(self, "agent_3") and self.agent_3:
            team_members.append(self.agent_3)
        if hasattr(self, "agent_4") and self.agent_4:
            team_members.append(self.agent_4)

        print(f"You created a team of {len(team_members)} agents:")
        for i, agent in enumerate(team_members, 1):
            print(f"  {i}. {agent}")

    async def resetting_team(self):
        await self.team.reset()

    def stopping_team(self):
        self.external_termination.set()

    async def create_new_team(self):
        print("Resetting the team with existing agents...")
        await self.team.reset()  # clears chat history if needed
        self.team = RoundRobinGroupChat(
            self.agents, termination_condition=self.text_termination
        )
        print("New team created from existing agents.")
