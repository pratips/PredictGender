from flask import Flask
from gender_predictor_main import *
app = Flask(__name__)


@app.route("/")
def main():
    return "Welcome!"

@app.route("/getGender/<name>")
def predict_gender_app(name):
    return str(predict_gender(name))

if __name__ == "__main__":
    app.run(host= '0.0.0.0')
