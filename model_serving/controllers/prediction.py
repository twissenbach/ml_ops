import logging

from model_serving.models.prediction import Prediction
from model_serving.services.inference_service import inference_service
from model_serving.gateways.mlflow_gateway import mlflow_gateway
from model_serving.services.database.prediction import ModelSQL, PredictionSQL
from model_serving.services.database.database_client import db
logger = logging.getLogger(__name__)

class PredictionController:

    def create_prediction(self, model_name: str, model_version: int, prediction: Prediction) -> Prediction:

        model = mlflow_gateway.get_model(model_name, model_version)

        prediction = inference_service.create_inference(None, prediction)

        model_sql = ModelSQL.from_model(model)
        prediction_sql = PredictionSQL.from_prediction(prediction, model_sql)

        db.session.add(prediction_sql)
        db.session.commit()

        return prediction_sql.to_prediction()

prediction_controller = PredictionController()