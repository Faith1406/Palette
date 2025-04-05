from team import Team
from autogen_agentchat.ui import Console

def main():
    theteam = Team("openai", "gemini-1.5-flash-8b", "gemini-1.5-flash-8b", api_key="AIzaSyB2MNAiUGIfaTDVF9xCdKJ0wLh05hZ54VY")
    question = input("Ask me a question damn it: ")
    answer = Console(theteam.team.run_stream(task=question))
    print(answer)

if __name__ == "__main__":
    main()
