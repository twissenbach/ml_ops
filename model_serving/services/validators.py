from typing import Dict, Any
from model_serving.models.prediction import Model
from model_serving.domain.common_enums import ModelType

class InputValidator:
    @staticmethod
    def validate_prediction_input(model: Model, features: Dict[str, Any]) -> Dict[str, str]:
        """
        Validates prediction input features against model requirements.
        Returns a dictionary of error messages if validation fails.
        """
        errors = {}

        # Check if features are provided
        if not features:
            errors["features"] = "Input features cannot be empty"
            return errors

        # Validate numeric types for all features
        for feature_name, value in features.items():
            if not isinstance(value, (int, float)):
                errors[feature_name] = f"Feature {feature_name} must be numeric"

        return errors

input_validator = InputValidator()