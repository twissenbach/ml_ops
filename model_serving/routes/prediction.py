import logging

from flask import Blueprint, request

from model_serving.models.prediction import Prediction
from model_serving.controllers.prediction import prediction_controller

prediction = Blueprint('prediction', __name__)


logger = logging.getLogger(__name__)


@prediction.route('/<model>/version/<version>/predict', methods=['POST'])
def create_prediction(model, version):
    data = request.json

    prediction = Prediction(
        inputs=data['features']
    )

    prediction: Prediction = prediction_controller.create_prediction(model_name=model, model_version=version, prediction=prediction)

    logger.info(f'Created prediction {prediction.id} for {model} version {version}')

    return prediction.to_json(), 200


@prediction.get('/prediction/<id>')
def get_prediction(id):
    prediction: Prediction = prediction_controller.get_prediction(id)

    logger.info(f'Retrieved prediction {prediction.id}')

    return prediction.to_json(), 200
