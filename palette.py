import asyncio
import os
import threading
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import tiktoken
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from model_factory import get_model_client

load_dotenv()


class SurveillanceAgent:
    def __init__(self, palette, gemini_api_key=os.getenv("API_KEY")):
        self.palette = palette
        self.monitoring_active = False
        self.background_thread = None

        self.status_history = []
        self.log_messages = []  # <-- Move this up here

        # Error patterns and solutions
        self.error_patterns = {
            "token_limit": [
                "token limit exceeded",
                "context length",
                "too many tokens",
            ],
            "api_failure": ["API error", "rate limit", "connection error", "timeout"],
            "model_error": [
                "content policy violation",
                "model error",
                "failed to generate",
            ],
            "team_deadlock": ["agents not making progress", "circular conversation"],
        }

        self.max_input_tokens = 4000
        self.token_warning_threshold = 0.9

        # Initialize Gemini
        self.gemini_client = None
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai.GenerativeModel("gemini-pro")
                self._log("Gemini AI agent initialized successfully")
            except Exception as e:
                self._log(f"Failed to initialize Gemini: {str(e)}")

            # Error patterns and solutions
            self.error_patterns = {
                "token_limit": [
                    "token limit exceeded",
                    "context length",
                    "too many tokens",
                ],
                "api_failure": [
                    "API error",
                    "rate limit",
                    "connection error",
                    "timeout",
                ],
                "model_error": [
                    "content policy violation",
                    "model error",
                    "failed to generate",
                ],
                "team_deadlock": [
                    "agents not making progress",
                    "circular conversation",
                ],
            }

            self.status_history = []
            self.log_messages = []
            self.max_input_tokens = 4000
            self.token_warning_threshold = 0.9

    async def get_ai_suggestion(self, context: str) -> str:
        """Get AI-powered suggestion from Gemini."""
        if not self.gemini_client:
            return "No AI agent available for suggestions"

        try:
            response = await self.gemini_client.generate_content_async(
                f"Analyze this system monitoring context and provide a concise solution:\n{context}",
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            return response.text
        except Exception as e:
            self._log(f"Gemini suggestion error: {str(e)}")
            return f"Standard solution: {self._get_standard_solution(context)}"

    def _get_standard_solution(self, error_type: str) -> str:
        """Fallback standard solutions when Gemini isn't available."""
        standard_solutions = {
            "token_limit": "Try reducing input size or using models with larger context windows.",
            "api_failure": "Check API keys, connection status, or try again later.",
            "model_error": "Consider changing the model or rephrasing your input.",
            "team_deadlock": "Try modifying agent system prompts or adding termination conditions.",
            "input_too_long": "Please shorten your input or increase the token limit.",
        }
        return standard_solutions.get(error_type, "No specific solution available")

    async def check_input_tokens(self, text: str) -> Dict[str, Any]:
        """Enhanced token check with AI suggestions."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            token_count = len(tokens)

            if token_count > self.max_input_tokens:
                context = (
                    f"Input exceeds token limit ({token_count}/{self.max_input_tokens})"
                )
                suggestion = await self.get_ai_suggestion("token_limit")

                self._log(f"Token limit exceeded: {context}")
                self._log(f"AI Suggestion: {suggestion}")

                return {
                    "status": "error",
                    "type": "input_too_long",
                    "token_count": token_count,
                    "max_limit": self.max_input_tokens,
                    "message": self._generate_token_limit_message(token_count),
                    "suggestion": suggestion,
                }

            elif token_count > self.max_input_tokens * self.token_warning_threshold:
                context = f"Input approaching token limit ({token_count}/{self.max_input_tokens})"
                suggestion = await self.get_ai_suggestion("token_warning")

                self._log(context)
                self._log(f"AI Suggestion: {suggestion}")

                return {
                    "status": "warning",
                    "type": "input_approaching_limit",
                    "token_count": token_count,
                    "max_limit": self.max_input_tokens,
                    "message": f"Warning: {context}",
                    "suggestion": suggestion,
                }

            return {
                "status": "ok",
                "token_count": token_count,
                "max_limit": self.max_input_tokens,
            }

        except Exception as e:
            error_msg = f"Token counting error: {str(e)}"
            self._log(error_msg)
            return {
                "status": "error",
                "type": "token_count_error",
                "message": error_msg,
            }

    async def _check_team_health(self) -> Dict[str, Any]:
        """Enhanced health check with AI analysis."""
        # Check input tokens
        if hasattr(self.palette, "text_input") and self.palette.text_input:
            input_check = await self.check_input_tokens(self.palette.text_input)
            if input_check["status"] != "ok":
                return {
                    "status": input_check["status"],
                    "type": input_check["type"],
                    "solution": input_check.get("suggestion"),
                    "message": input_check["message"],
                    "auto_recoverable": False,
                }

        # Check conversation activity
        if not hasattr(self.palette.team, "messages") or not self.palette.team.messages:
            return {"status": "ok", "message": "No active conversation to monitor"}

        messages = self.palette.team.messages

        # AI-powered deadlock detection
        if len(messages) > 6 and await self._ai_detect_deadlock(messages):
            context = "Agents appear stuck in circular conversation"
            suggestion = await self.get_ai_suggestion("team_deadlock")

            self._log(f"Deadlock detected: {context}")
            self._log(f"AI Suggestion: {suggestion}")

            return {
                "status": "warning",
                "type": "team_deadlock",
                "solution": suggestion,
                "auto_recoverable": True,
            }

        # AI-powered error detection
        for msg in messages[-3:]:
            if hasattr(msg, "content"):
                error_type = self._detect_error(msg.content)
                if error_type:
                    context = (
                        f"Detected {error_type} in message: {msg.content[:100]}..."
                    )
                    suggestion = await self.get_ai_suggestion(error_type)

                    self._log(context)
                    self._log(f"AI Suggestion: {suggestion}")

                    return {
                        "status": "error",
                        "type": error_type,
                        "solution": suggestion,
                        "auto_recoverable": error_type in ["api_failure"],
                    }

        return {"status": "ok"}

    async def _ai_detect_deadlock(self, messages: List) -> bool:
        """Use AI to detect conversation deadlocks."""
        if len(messages) < 6:
            return False

        conversation_snippet = "\n".join(
            f"{msg.source}: {msg.content[:200]}"
            for msg in messages[-6:]
            if hasattr(msg, "content") and hasattr(msg, "source")
        )

        try:
            prompt = f"""Analyze this conversation snippet and determine if the agents are stuck in a loop or deadlock:
            {conversation_snippet}
            
            Respond only with 'yes' or 'no'."""

            response = await self.gemini_client.generate_content_async(prompt)
            return response.text.strip().lower() == "yes"
        except Exception:
            # Fallback to standard detection if AI fails
            return self._detect_deadlock(messages)

    def _generate_token_limit_message(self, token_count: int) -> str:
        """Generate user-friendly message about token limits."""
        over_by = token_count - self.max_input_tokens
        reduction_pct = int((over_by / token_count) * 100) + 10  # Add buffer

        return (
            f"Your input exceeds the maximum token limit of {self.max_input_tokens}.\n"
            f"Current token count: {token_count} (over by {over_by})\n"
            f"Suggested actions:\n"
            f"1. Shorten your input by about {reduction_pct}%\n"
            f"2. Upgrade your plan to increase token limits\n"
            f"3. Split your query into multiple smaller requests"
        )

    def start_background_monitoring(self):
        """Start the surveillance agent in the background."""
        if self.monitoring_active:
            self._log("Surveillance monitoring is already active.")
            return

        self.monitoring_active = True
        self.background_thread = threading.Thread(target=self._background_monitor_loop)
        self.background_thread.daemon = True
        self.background_thread.start()
        self._log("Surveillance Agent activated and monitoring in background.")

    def stop_background_monitoring(self):
        """Stop the background monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.background_thread:
            self.background_thread.join(timeout=2.0)
            self._log("Surveillance Agent deactivated.")

    def _log(self, message):
        """Log a message to the internal log instead of printing it."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")
        # Limit log size
        if len(self.log_messages) > 100:
            self.log_messages = self.log_messages[-100:]

    def _background_monitor_loop(self):
        """Main background monitoring loop."""
        while self.monitoring_active:
            try:
                # Check team health
                health_status = self._check_team_health()

                if health_status["status"] != "ok":
                    self._log(f"SURVEILLANCE ALERT: {health_status['type']}")
                    self._log(f"SUGGESTED SOLUTION: {health_status['solution']}")

                    # Attempt auto-recovery if enabled
                    if health_status.get("auto_recoverable", False):
                        self._log("Attempting auto-recovery...")
                        self._attempt_recovery(health_status["type"])

                # Store health status history
                self.status_history.append(
                    {"timestamp": time.time(), "status": health_status}
                )

                # Keep history manageable
                if len(self.status_history) > 100:
                    self.status_history = self.status_history[-100:]

                # Sleep before next check
                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self._log(f"Surveillance monitoring error: {str(e)}")
                time.sleep(10)  # Back off on errors

    def _check_team_health(self) -> Dict[str, Any]:
        """Check the health status of the team."""
        # First check input tokens if there's text input
        if hasattr(self.palette, "text_input") and self.palette.text_input:
            input_check = asyncio.run(self.check_input_tokens(self.palette.text_input))
            if input_check["status"] != "ok":
                return {
                    "status": input_check["status"],
                    "type": input_check["type"],
                    "solution": input_check.get(
                        "suggestion", self.solutions.get(input_check["type"], "")
                    ),
                    "message": input_check["message"],
                    "auto_recoverable": False,
                }
        # Check if a conversation is active
        if not hasattr(self.palette.team, "messages") or not self.palette.team.messages:
            return {"status": "ok", "message": "No active conversation to monitor"}

        messages = self.palette.team.messages

        # Check for team deadlock
        if len(messages) > 6:
            if self._detect_deadlock(messages):
                return {
                    "status": "warning",
                    "type": "team_deadlock",
                    "solution": self.solutions["team_deadlock"],
                    "auto_recoverable": True,
                }

        # Check last few messages for errors
        for msg in messages[-3:]:
            if hasattr(msg, "content"):
                error_type = self._detect_error(msg.content)
                if error_type:
                    return {
                        "status": "error",
                        "type": error_type,
                        "solution": self.solutions[error_type],
                        "auto_recoverable": error_type in ["api_failure"],
                    }

        return {"status": "ok"}

    def _detect_error(self, message_text: str) -> Optional[str]:
        """Check message for known error patterns."""
        if not message_text:
            return None

        lower_text = message_text.lower()
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in lower_text for pattern in patterns):
                return error_type
        return None

    def _detect_deadlock(self, messages: List) -> bool:
        """Check if conversation is stuck in a loop."""
        if len(messages) < 6:
            return False

        # Extract relevant information from messages
        recent_contents = []
        recent_sources = []

        for msg in messages[-6:]:
            if hasattr(msg, "content") and hasattr(msg, "source"):
                recent_contents.append(msg.content)
                recent_sources.append(msg.source)

        # Check for repeated patterns
        for i in range(len(recent_contents) - 2):
            if (
                recent_sources[i] == recent_sources[i + 2]
            ):  # Same agent talking every other turn
                similarity = self._text_similarity(
                    recent_contents[i], recent_contents[i + 2]
                )
                if similarity > 0.7:  # High similarity threshold
                    return True

        return False

    def get_logs(self):
        """Get the surveillance agent logs."""
        return self.log_messages

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity measure."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0
        intersection = words1.intersection(words2)
        return len(intersection) / max(len(words1), len(words2))

    def _attempt_recovery(self, error_type: str):
        """Attempt to automatically recover from specific errors."""
        if error_type == "team_deadlock":
            # Create a new asyncio event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Reset the team
                loop.run_until_complete(self.palette.create_new_team())
                self._log("Team reset successfully after deadlock detection.")
            except Exception as e:
                self._log(f"Error during auto-recovery: {str(e)}")
            finally:
                loop.close()

        elif error_type == "api_failure":
            # Maybe implement retry logic or API key rotation
            pass

    def get_status_report(self) -> Dict[str, Any]:
        """Generate a status report of the monitoring."""
        return {
            "active": self.monitoring_active,
            "uptime": time.time() - self.status_history[0]["timestamp"]
            if self.status_history
            else 0,
            "total_checks": len(self.status_history),
            "recent_issues": [
                h for h in self.status_history if h["status"]["status"] != "ok"
            ][-5:],
            "current_status": self._check_team_health(),
        }


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
        token_threshold=10,
        config=None,
        *args,
        **kwargs,
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
        self.surveillance = SurveillanceAgent(self)
        self.text_input = ""

        self.agents = []

        auto_monitor = kwargs.get("auto_monitor", True)
        if auto_monitor:
            self.surveillance.start_background_monitoring()

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
        conversation_list = []
        retries = 0

        while retries < 3:
            async for message in self.team.run_stream(task=text_input):
                if hasattr(message, "source") and hasattr(message, "content"):
                    conversation_list.append(
                        {"source": message.source, "content": message.content}
                    )

                    full_output += message.content + " "
                else:
                    if isinstance(message, str):
                        conversation_list.append(
                            {"source": "system", "content": message}
                        )

            word_count = len(full_output.split())
            estimated_tokens = int(word_count * 1.3)
            print(f"Estimated Total Tokens Used: {estimated_tokens}")

            if estimated_tokens >= self.token_threshold:
                print("Token limit exceeded, expanding team...")
                await self.create_new_team()
                retries += 1
                full_output = ""
                conversation_list = []
            else:
                break

        return conversation_list, estimated_tokens

    def run_team(self, text: str):
        self.text_input = text

        # Check input tokens before running
        token_check = asyncio.run(self.surveillance.check_input_tokens(text))
        if token_check["status"] == "error":
            print(token_check["message"])
            print("\nSuggested solutions:")
            print(token_check["suggestion"])
            return [], 0

        try:
            result = asyncio.run(self.print_convo_and_count_tokens(text))
            return result
        except Exception as e:
            print(f"Error running team: {str(e)}")
            return [], 0

    def stop_surveillance(self):
        """Stop the background surveillance."""
        if hasattr(self, "surveillance"):
            self.surveillance.stop_background_monitoring()

    def surveillance_status(self):
        """Get current surveillance status."""
        if hasattr(self, "surveillance"):
            return self.surveillance.get_status_report()
        return {"active": False, "message": "Surveillance not initialized"}

    def display_team_members(self):
        team_members = [self.agent_1, self.agent_2]
        if hasattr(self, "agent_3") and self.agent_3:
            team_members.append(self.agent_3)
        if hasattr(self, "agent_4") and self.agent_4:
            team_members.append(self.agent_4)

        print(f"You created a team of {len(team_members)} agents:")
        for i, agent in enumerate(team_members, 1):
            print(f"  {i}. {agent}")

    def set_token_limit(self, max_tokens: int):
        """Set the maximum allowed input tokens."""
        self.surveillance.max_input_tokens = max_tokens
        print(f"Input token limit set to {max_tokens}")

    async def resetting_team(self):
        await self.team.reset()

    def stopping_team(self):
        self.external_termination.set()

    async def create_new_team(self):
        print("Resetting the team with existing agents...")
        await self.team.reset()
        self.team = RoundRobinGroupChat(
            self.agents, termination_condition=self.text_termination
        )
        print("New team created from existing agents.")

    def count_token_input(self, text):
        text = self.text_input

        encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        token_count = len(tokens)

        return token_count

    def check_token_limit(self, text=None, max_token_limit=1000):
        input_text = text if text is not None else self.text_input

        token_count = self.count_token_input(input_text)

        if token_count > max_token_limit:
            suggested_reduction = int(token_count * 0.2)

            warning_message = (
                f"Warning: Your input exceeds the maximum token limit of {max_token_limit}.\n"
                f"Current token count: {token_count}\n"
                f"Please reduce your input by approximately {suggested_reduction} tokens "
                f"(about {suggested_reduction * 4} characters)."
            )

            return {
                "exceeds_limit": True,
                "token_count": token_count,
                "max_limit": max_token_limit,
                "over_by": token_count - max_token_limit,
                "warning": warning_message,
            }

        return {
            "exceeds_limit": False,
            "token_count": token_count,
            "max_limit": max_token_limit,
            "remaining": max_token_limit - token_count,
        }

    def get_surveillance_logs(self):
        """Get logs from the surveillance agent."""
        if hasattr(self, "surveillance"):
            return self.surveillance.log_messages
        return []
