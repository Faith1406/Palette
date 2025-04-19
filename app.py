from flask import Flask, render_template, request

from test_flask import main

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        user_input = request.form["search"]
        answer = main(user_input)
    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
