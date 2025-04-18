import os

from dotenv import load_dotenv

from palette import Palette

load_dotenv()

team = Palette(
    "openai",
    "ollama",
    "gemini-1.5-flash-8b",
    "llama3",
    system_message_1="You are going to answer the coding problem in c++ that are from leetcode with no explanation.",
    system_message_2="You area going to review the generated code if the answer is correct then answer as 'APPROVE'",
    description_1="A helpfull assistant that will give the answer to the coding problem in c++, with no explanation",
    description_2="A helpfull assistant that review the answer of the coding problem in c++.",
    termination_text="APPROVE",
    api_key_1=os.getenv("API_KEY"),
    behaviour_1="Coding_Assistant",
    behaviour_2="Code_Tester",
)


def main():
    team.display_team_members()
    question = input("Enter Your Question: ")
    team.run_team(question)


if __name__ == "__main__":
    main()
