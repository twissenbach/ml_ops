import unittest
from model_serving.services.validators import input_validator
from model_serving.models.prediction import Model
from model_serving.domain.common_enums import ModelType

class TestInputValidator(unittest.TestCase):
    def setUp(self):
        self.model = Model(
            model_name="test_model",
            model_version="1",
            model_type=ModelType.REGRESSION
        )

    def test_empty_features(self):
        # Test validation with empty features
        errors = input_validator.validate_prediction_input(
            model=self.model,
            features={}
        )
        self.assertIn("features", errors)
        self.assertEqual(errors["features"], "Input features cannot be empty")

    def test_non_numeric_features(self):
        # Test validation with non-numeric features
        errors = input_validator.validate_prediction_input(
            model=self.model,
            features={"feature1": "string_value", "feature2": 1.0}
        )
        self.assertIn("feature1", errors)
        self.assertTrue("must be numeric" in errors["feature1"])

    def test_valid_features(self):
        # Test validation with valid features
        errors = input_validator.validate_prediction_input(
            model=self.model,
            features={"feature1": 1.0, "feature2": 2}
        )
        self.assertEqual(errors, {})