import logging

from model_serving.models.prediction import Prediction, Model
from model_serving.services.inference_service import inference_service
from model_serving.services.umap_service import umap_service
from model_serving.gateways.mlflow_gateway import mlflow_gateway
from model_serving.services.database.database_client import db
from model_serving.services.database.prediction import ModelSQL, PredictionSQL
from model_serving.services.validators import input_validator
from flask import abort
from model_serving.domain.exceptions import (
    ModelNotFoundException,
    InvalidInputException,
    ModelTypeException,
    InferenceException
)


logger = logging.getLogger(__name__)


class PredictionController:

    def create_prediction(self, model_name: str, model_version: int, prediction: Prediction) -> Prediction:
        try:
            # Get model
            model = mlflow_gateway.get_model(model_name, model_version)
            if not model:
                raise ModelNotFoundException(model_name, str(model_version))

            # Validate input
            validation_errors = input_validator.validate_prediction_input(model, prediction.inputs)
            if validation_errors:
                raise InvalidInputException(validation_errors)

            # Create inference
            try:
                prediction = inference_service.create_inference(model, prediction)
                
                # Add UMAP embeddings
                try:
                    embeddings = umap_service.create_embedding(model, prediction)
                    prediction.embeddings = embeddings
                except Exception as e:
                    logger.error(f"Failed to create UMAP embeddings: {str(e)}") # continue prediction if embeddings fail
                
            except Exception as e:
                raise InferenceException(f"Failed to create prediction: {str(e)}")

            # Save prediction
            prediction_sql = PredictionSQL.from_prediction(prediction, model)
            db.session.add(prediction_sql)
            db.session.commit()

            return prediction

        except Exception as e:
            db.session.rollback()
            raise

    def get_prediction(self, id) -> Prediction:
        prediction = db.session.query(PredictionSQL).filter(PredictionSQL.id == id).first()

        return prediction.to_prediction()


prediction_controller = PredictionController()
