import logging

from model_serving.models.prediction import Prediction, Model
from model_serving.services.inference_service import inference_service
from model_serving.gateways.mlflow_gateway import mlflow_gateway
from model_serving.services.database.database_client import db
from model_serving.services.database.prediction import ModelSQL, PredictionSQL


logger = logging.getLogger(__name__)


class PredictionController:

    def create_prediction(self, model_name: str, model_version: int, prediction: Prediction) -> Prediction:

        model: Model = mlflow_gateway.get_model(model_name, model_version)
        prediction.model = model


        prediction = inference_service.create_inference(model, prediction)

        model_sql = ModelSQL.from_model(model)

        prediction_sql = PredictionSQL.from_prediction(prediction, model_sql)

        db.session.add(prediction_sql)
        db.session.commit()

        return self.get_prediction(prediction_sql.id)

    def get_prediction(self, id) -> Prediction:
        prediction = db.session.query(PredictionSQL).filter(PredictionSQL.id == id).first()

        return prediction.to_prediction()


prediction_controller = PredictionController()
