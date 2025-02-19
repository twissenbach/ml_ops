import logging

from model_serving.models.prediction import Model, Prediction
from model_serving.monitoring import FUNCTION_DURATION


logger = logging.getLogger(__name__)


class InferenceService:

    @staticmethod
    @FUNCTION_DURATION.labels('create_prediction').time()
    def create_inference(model: Model, prediction: Prediction) -> Prediction:

        if hasattr(model.model, 'predict_proba'):
            # This will change in the future
            probability = model.model.predict_proba(prediction.get_pandas_frame_of_inputs())[:,1][0]

            if model.threshold['value'] > probability:
                label = model.threshold['above']
            elif model.threshold['value'] == probability:
                label = model.threshold['equal']
            else:
                label = model.threshold['below']

        else:
            label = model._model.predict(prediction.inputs)

        prediction.probability = probability
        prediction.threshold = model.threshold['value']
        prediction.label = label

        return prediction

inference_service = InferenceService()