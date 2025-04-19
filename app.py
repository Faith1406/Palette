import asyncio
import os
import pickle
import sys

from dotenv import load_dotenv
from quart import Quart, make_response, redirect, render_template, request, url_for

load_dotenv()
app = Quart(__name__)

# Store conversations in memory
conversations = {}

# Helper script that will run in a separate process
HELPER_SCRIPT = """
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
"""

# Create the helper script file if it doesn't exist
with open("palette_helper.py", "w") as f:
    f.write(HELPER_SCRIPT)


async def run_palette_in_subprocess(question):
    """Run the Palette team in a separate process"""
    # Create subprocess
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "palette_helper.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Send the question to the subprocess
    stdout, stderr = await process.communicate(pickle.dumps(question))

    if process.returncode != 0:
        error_message = stderr.decode() if stderr else "Unknown error"
        raise Exception(f"Palette helper script failed: {error_message}")

    # Parse the result from the subprocess
    return pickle.loads(stdout)


@app.route("/", methods=["GET", "POST"])
async def chat():
    conversation_id = request.cookies.get("conversation_id")
    if not conversation_id or conversation_id not in conversations:
        conversation_id = os.urandom(16).hex()
        conversations[conversation_id] = []

    if request.method == "POST":
        form = await request.form
        question = form.get("question")

        if question:
            # Add user question to conversation
            conversations[conversation_id].append(
                {"source": "user", "content": question}
            )

            try:
                # Use the subprocess to run the Palette team
                result = await run_palette_in_subprocess(question)

                # Process the result
                if isinstance(result, list):
                    for response in result:
                        conversations[conversation_id].append(response)
                else:
                    conversations[conversation_id].append(
                        {"source": "system", "content": str(result)}
                    )
            except Exception as e:
                conversations[conversation_id].append(
                    {"source": "system", "content": f"Error: {str(e)}"}
                )

        return redirect(url_for("chat"))

    response = await make_response(
        await render_template(
            "index.html", conversation=conversations.get(conversation_id, [])
        )
    )
    response.set_cookie("conversation_id", conversation_id)
    return response


# Add API route for JSON responses
@app.route("/api/chat", methods=["POST"])
async def api_chat():
    conversation_id = request.cookies.get("conversation_id", os.urandom(16).hex())

    if conversation_id not in conversations:
        conversations[conversation_id] = []

    json_data = await request.get_json()
    question = json_data.get("question", "")

    if question:
        conversations[conversation_id].append({"source": "user", "content": question})

        try:
            result = await run_palette_in_subprocess(question)

            if isinstance(result, list):
                for response in result:
                    conversations[conversation_id].append(response)
            else:
                conversations[conversation_id].append(
                    {"source": "system", "content": str(result)}
                )
        except Exception as e:
            conversations[conversation_id].append(
                {"source": "system", "content": f"Error: {str(e)}"}
            )

    response = await make_response(
        {
            "conversation": conversations[conversation_id],
            "conversation_id": conversation_id,
        }
    )
    response.set_cookie("conversation_id", conversation_id)
    return response


if __name__ == "__main__":
    app.run(debug=True)
