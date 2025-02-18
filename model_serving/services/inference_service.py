import logging

from model_serving.models.prediction import Model, Prediction
from model_serving.domain.common_enums import Labels

logger = logging.getLogger(__name__)

class InferenceService:

    @staticmethod
    def create_inference(model: Model, prediction: Prediction) -> Prediction:



        probability = None
        if hasattr(model, 'predict_proba'):
            # this will change in the future
            # probability = model._model.predict_proba(prediction.inputs) # default to predict_proba because we get more information from it
            probability = model.model.predict_proba(prediction.get_pandas_frame_of_inputs())[:,1][0]

            if model.threshold > probability:
                label = model.threshold['above']
            elif model.threshold == probability:
                label = model.threshold['equal']
            else:
                label = model.threshold['below']

        else:
            model._model.predict(prediction.inputs)

        prediction.probabilities = probability
        prediction.threshold = model.threshold['value']
        prediction.label = label

        return prediction
    
inference_service = InferenceService()