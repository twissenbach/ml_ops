import logging

from model_serving.models.prediction import Model, Prediction
from model_serving.monitoring import FUNCTION_DURATION
from model_serving.models.prediction import Shap


logger = logging.getLogger(__name__)


class ExplainerService:

    @staticmethod
    @FUNCTION_DURATION.labels('create_explanation').time()
    def create_explanation(model: Model, prediction: Prediction) -> Prediction:

        shap_values_ = model._explainer.predict(prediction.inputs) or []

        shap_values = []
        if model.labels:
            for this_label in model.labels:
                idx = model.labels.index(this_label)
                shap_values.append(
                    Shap(
                        label=this_label.value,
                        shap=shap_values_[:,:, idx]
                    )
                )

        else:
            shap_values.append(Shap(
                label=None,
                shap=shap_values_[:,:,0]
            ))

        prediction.shap_values = shap_values

        return prediction

explainer_service = ExplainerService()