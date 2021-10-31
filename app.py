from flask import Flask, jsonify, request
from classifier import getPrediction

app = Flask(__name__)

@app.route("/predictalphabet", methods = ["POST"])
def predictdata():
    image = request.files.get("alphabet")
    prediction = getPrediction(image)
    return jsonify({
        "prediction": prediction
    }), 200

if __name__ = "__main__":
    app.run(debug = True)