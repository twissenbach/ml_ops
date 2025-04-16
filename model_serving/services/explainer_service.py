import logging

from model_serving.models.prediction import Model, Prediction
from model_serving.monitoring import FUNCTION_DURATION
from model_serving.models.prediction import Shap


logger = logging.getLogger(__name__)


class ExplainerService:

    @staticmethod
    @FUNCTION_DURATION.labels('create_explanation').time()
    def create_explanation(model: Model, prediction: Prediction) -> Prediction:
        try:
            # Get SHAP values from model explainer
            shap_values_ = model._explainer.predict(prediction.inputs) or []
            
            # Get feature names from the prediction inputs
            feature_names = list(prediction.inputs.keys())
            
            shap_values = []
            if model.labels:
                for this_label in model.labels:
                    idx = model.labels.index(this_label)
                    # Create a dictionary mapping feature names to SHAP values
                    shap_dict = {}
                    
                    # Extract the SHAP values for this label
                    label_shap_values = shap_values_[:,:,idx]
                    
                    # Map each feature name to its SHAP value
                    for i, feature in enumerate(feature_names):
                        if i < label_shap_values.shape[1]:  # Ensure we don't go out of bounds
                            shap_dict[feature] = float(label_shap_values[0, i])
                    
                    shap_values.append(
                        Shap(
                            label=this_label.value,
                            shap_values=shap_dict
                        )
                    )
            else:
                # For regression or single-output models
                shap_dict = {}
                shap_array = shap_values_[:,:,0]  # Use the first (and likely only) output
                
                # Map each feature name to its SHAP value
                for i, feature in enumerate(feature_names):
                    if i < shap_array.shape[1]:  # Ensure we don't go out of bounds
                        shap_dict[feature] = float(shap_array[0, i])
                
                shap_values.append(Shap(
                    label=None,
                    shap_values=shap_dict
                ))
            
            prediction.shap_values = shap_values
            return prediction
            
        except Exception as e:
            # Log the error but don't fail the prediction
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return prediction

explainer_service = ExplainerService()