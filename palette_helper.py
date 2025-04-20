
import asyncio
import os
import pickle
import sys
from dotenv import load_dotenv
from palette import Palette

load_dotenv()

def main():
    # Get input from stdin
    input_data = sys.stdin.buffer.read()
    question = pickle.loads(input_data)
    
    # Initialize Palette with the same parameters as in your app
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
        system_message_2="You are going to review the generated code if the answer is correct then answer as 'APPROVE'",
        system_message_3="You are going to review the generated code if the answer is correct then answer as 'APPROVE'",
        system_message_4="You are going to review the generated code if the answer is correct then answer as 'APPROVE'",
        description_1="A helpful assistant that will give the answer to the coding problem in c++, with no explanation",
        description_2="A helpful assistant that review the answer of the coding problem in c++.",
        description_3="A helpful assistant that review the answer of the coding problem in c++.",
        description_4="A helpful assistant that review the answer of the coding problem in c++.",
        termination_text="APPROVE",
        api_key_2=os.getenv("API_KEY"),
        api_key_3=os.getenv("API_KEY"),
        api_key_4=os.getenv("API_KEY"),
        behaviour_1="Coding_Assistant",
        behaviour_2="Code_Tester1",
        behaviour_3="Code_Tester2",
        behaviour_4="Code_Tester3",
        token_threshold=150,
    )
    
    # Run the team and get the result
    result = team.run_team(question)
    
    # Write result to stdout
    sys.stdout.buffer.write(pickle.dumps(result))

if __name__ == "__main__":
    main()
