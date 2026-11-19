import os

from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, render_template, request
from flask_cors import CORS, cross_origin

from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.pipeline.prediction2 import PredictionPipeline2
from cnnClassifier.utils.common import decodeImage

load_dotenv()

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app)


def get_predict_route() -> str:
    """
    PREDICT_ROUTE=one -> /predict  (local trained model)
    PREDICT_ROUTE=two -> /predict2 (Hugging Face model)
    Default (missing/empty): /predict2
    """
    route = os.getenv("PREDICT_ROUTE", "two").strip().lower()
    if route in {"one", "1", "local"}:
        return "/predict"
    return "/predict2"


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)
        self.classifier2 = PredictionPipeline2(self.filename)


# Module-level init so gunicorn `app:app` can serve prediction routes.
clApp = ClientApp()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    response = make_response(
        render_template("index.html", predict_url=get_predict_route())
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    # Training is not intended inside the Render inference container.
    return jsonify(
        {
            "message": "Training is disabled in the deployed inference service. Run training locally or via DVC."
        }
    ), 400


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    image = request.json["image"]
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


@app.route("/predict2", methods=["POST"])
@cross_origin()
def predictRoute2():
    image = request.json["image"]
    decodeImage(image, clApp.filename)
    result = clApp.classifier2.predict()
    return jsonify(result)


@app.route("/health", methods=["GET"])
@cross_origin()
def health():
    return jsonify({"status": "ok", "predict_route": get_predict_route()})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
