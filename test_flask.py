import contextlib
import io
import os

from dotenv import load_dotenv

from palette import Palette

load_dotenv()

team = Palette(
    "ollama",
    "openai",
    "openai",
    "openai",
    "llama3",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b",
    system_message_1="You are going to answer the coding problem in c++ that are from leetcode with no explanation.",
    system_message_2="You area going to review the generated code if the answer is correct then answer as 'APPROVE'",
    system_message_3="You area going to review the generated code if the answer is correct then answer as 'APPROVE'",
    system_message_4="You area going to review the generated code if the answer is correct then answer as 'APPROVE'",
    description_1="A helpfull assistant that will give the answer to the coding problem in c++, with no explanation",
    description_2="A helpfull assistant that review the answer of the coding problem in c++.",
    description_3="A helpfull assistant that review the answer of the coding problem in c++.",
    description_4="A helpfull assistant that review the answer of the coding problem in c++.",
    termination_text="APPROVE",
    api_key_2=os.getenv("API_KEY"),
    api_key_3=os.getenv("API_KEY"),
    api_key_4=os.getenv("API_KEY"),
    behaviour_1="Coding_Assistant",
    behaviour_2="Code_Tester1",
    behaviour_3="Code_Tester2",
    behaviour_4="Code_Tester3",
)


def main(question):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        team.display_team_members()
        team.run_team(question)
    output = f.getvalue()
    return output


if __name__ == "__main__":
    main()
