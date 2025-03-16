import logging
from typing import Optional, Dict, Any
import numpy as np
from model_serving.models.prediction import Model, Prediction
from model_serving.monitoring import FUNCTION_DURATION
from model_serving.domain.common_enums import ModelType


logger = logging.getLogger(__name__)


class InferenceService:

    @staticmethod
    @FUNCTION_DURATION.labels('create_prediction').time()
    def create_inference(model: Any, prediction: Prediction) -> Prediction:
        """Create inference based on model type"""
        if model.model_type == ModelType.CLASSIFICATION.value:
            return InferenceService._create_classification_inference(model, prediction)
        else:
            return InferenceService._create_regression_inference(model, prediction)

    @staticmethod
    def _create_classification_inference(model: Any, prediction: Prediction) -> Prediction:
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

    @staticmethod
    def _create_regression_inference(model: Any, prediction: Prediction) -> Prediction:
        # New regression logic
        result = model.model.predict(prediction.inputs)
        prediction.value = float(result[0])  # Direct numeric prediction
        prediction.probability = None  # Regression doesn't use probability
        return prediction

    def _prepare_inputs(self, inputs: Dict) -> np.ndarray:
        """Prepare inputs for model prediction"""
        # Convert inputs to the format expected by the model
        # This might need to be customized based on your model requirements
        return np.array([list(inputs.values())])

inference_service = InferenceService()