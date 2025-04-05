from team import Team 
from config_loader import load_config

def main():
    config = load_config("config.yaml")

    team = Team(config = config)

    team.display_team_members()
    question = input("Enter something: ")
    team.run_team(question)

if __name__ == "__main__":
    main()
