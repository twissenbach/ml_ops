import logging
from flask import Blueprint, request, jsonify

from model_serving.models.prediction import Prediction
from model_serving.controllers.prediction import prediction_controller
from model_serving.domain.exceptions import (
    ModelNotFoundException,
    InvalidInputException,
    ModelTypeException,
    InferenceException
)

prediction = Blueprint('prediction', __name__)

logger = logging.getLogger(__name__)


@prediction.route('/<model>/version/<version>/predict', methods=['POST'])
def create_prediction(model, version):
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid Input", "message": "No JSON data provided"}), 400
        
        if 'features' not in data:
            return jsonify({"error": "Invalid Input", "message": "Missing 'features' in request"}), 400
            
        prediction = Prediction(inputs=data['features'])
        prediction = prediction_controller.create_prediction(
            model_name=model, 
            model_version=version, 
            prediction=prediction
        )
        
        logger.info(f'Created prediction {prediction.id} for {model} version {version}')
        return prediction.to_json(), 200

    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        return jsonify({"error": "Invalid Input", "message": str(e)}), 400
        
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {str(e)}")
        return jsonify({"error": "Model Not Found", "message": str(e)}), 404
        
    except (InvalidInputException, ModelTypeException) as e:
        logger.error(f"Invalid input: {str(e)}")
        return jsonify({"error": "Invalid Input", "message": str(e)}), 400
        
    except InferenceException as e:
        logger.error(f"Inference error: {str(e)}")
        return jsonify({"error": "Prediction Error", "message": str(e)}), 500
        
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500


@prediction.get('/prediction/<id>')
def get_prediction(id):
    prediction: Prediction = prediction_controller.get_prediction(id)

    logger.info(f'Retrieved prediction {prediction.id}')

    return prediction.to_json(), 200
