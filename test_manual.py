import os
from neuronnet import NeuronNet 
from dotenv import load_dotenv

load_dotenv()

team = NeuronNet(
        "openai", 
        "ollama",  
        "gemini-1.5-flash-8b", 
        "llama3", 
        system_message_1="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
        system_message_2="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
        description_1="A helpful assistant that can plan trips.",
        description_2="A helpful assistant that can provide language tips for a given destination.",
        api_key_1=os.getenv("API_KEY"), 
        behaviour_1="Planner", 
        behaviour_2="language_agent",
    )
def main():
    team.display_team_members() 
    question = input("Enter Your Question: ")
    team.run_team(question)
    
if __name__ == "__main__":
    main()
