import unittest
from model_serving.models.prediction import Model, Prediction
from model_serving.domain.common_enums import ModelType, Labels

class TestModelUpdates(unittest.TestCase):
    def test_model_types(self):
        # Test that both model types can be created
        classification_model = Model(
            model_name="test_classifier",
            model_version="1",
            model_type=ModelType.CLASSIFICATION
        )
        regression_model = Model(
            model_name="test_regressor",
            model_version="1",
            model_type=ModelType.REGRESSION
        )
        
        self.assertEqual(classification_model.model_type, ModelType.CLASSIFICATION)
        self.assertEqual(regression_model.model_type, ModelType.REGRESSION)

class TestPrediction(unittest.TestCase):
    def test_classification_prediction(self):
        prediction = Prediction(
            inputs={"feature1": 1.0},
            value=Labels.BENIGN,  # Using existing Labels enum
            probability=0.8
        )
        self.assertEqual(prediction.value, Labels.BENIGN)
    
    def test_regression_prediction(self):
        prediction = Prediction(
            inputs={"feature1": 1.0},
            value=42.5,  # Numeric value for regression
            probability=None  # Regression might not have probability
        )
        self.assertEqual(prediction.value, 42.5)