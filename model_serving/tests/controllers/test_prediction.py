from sklearn.datasets import load_breast_cancer
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from typing import Dict

from ..base import BaseTestCase, create_metrics_dir

create_metrics_dir()

from model_serving.controllers.prediction import prediction_controller
from model_serving.models.prediction import Prediction
from model_serving.domain.common_enums import Labels, ModelType
from model_serving.services.database.prediction import ModelSQL
from model_serving.domain.exceptions import ModelNotFoundException, InvalidInputException, InferenceException


class TestPredictionController(BaseTestCase):

    @patch('model_serving.gateways.mlflow_gateway.mlflow')
    def setUp(self, mock_mlflow):
        super().setUp()
        # Mock MLflow behavior
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [1]
        mock_mlflow.sklearn.load_model.return_value = self.mock_model

        self.app = self.create_app()

        data = load_breast_cancer()

        X = pd.DataFrame(data.data, columns=data.feature_names)

        self.sample_row =  {'mean radius': 9.029, 'mean texture': 17.33, 'mean perimeter': 58.79, 'mean area': 250.5,
  'mean smoothness': 0.1066,
  'mean compactness': 0.1413,
  'mean concavity': 0.313,
  'mean concave points': 0.04375,
  'mean symmetry': 0.2111,
  'mean fractal dimension': 0.08046,
  'radius error': 0.3274,
  'texture error': 1.194,
  'perimeter error': 1.885,
  'area error': 17.67,
  'smoothness error': 0.009549,
  'compactness error': 0.08606,
  'concavity error': 0.3038,
  'concave points error': 0.03322,
  'symmetry error': 0.04197,
  'fractal dimension error': 0.009559,
  'worst radius': 10.31,
  'worst texture': 22.65,
  'worst perimeter': 65.5,
  'worst area': 324.7,
  'worst smoothness': 0.1482,
  'worst compactness': 0.4365,
  'worst concavity': 1.252,
  'worst concave points': 0.175,
  'worst symmetry': 0.4228,
  'worst fractal dimension': 0.1175}

    @patch('model_serving.controllers.prediction.mlflow_gateway')
    @patch('model_serving.services.database.prediction.ModelSQL.from_model')
    @patch('model_serving.services.inference_service.InferenceService.create_inference')
    def test_create_prediction(self, mock_inference, mock_model_sql, mock_gateway):
        # Create prediction without explicitly setting ID
        prediction = Prediction(inputs={"feature1": 1.0})
        
        # Set up mock model with proper predict_proba behavior
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # Create a mock model object with proper attributes
        mock_gateway_model = MagicMock()
        mock_gateway_model.model = mock_model
        mock_gateway_model.model_type = ModelType.CLASSIFICATION.value
        mock_gateway_model.model_name = "test_model"
        mock_gateway_model.model_version = 1
        mock_gateway_model.id = "test_model_id"
        mock_gateway_model.threshold = {
            'value': 0.5,
            'above': Labels.BENIGN.value,
            'below': Labels.MALIGNANT.value
        }
        mock_gateway.get_model.return_value = mock_gateway_model

        # Mock the database model with a string ID
        mock_model_sql.return_value = ModelSQL(
            id="test_model_id",
            model_type=ModelType.CLASSIFICATION.value,
            model_name="test_model",
            model_version=1
        )

        # Set the model on the prediction object
        prediction.model = mock_model_sql.return_value

        # Mock the inference service
        def mock_inference_implementation(model, prediction):
            prediction.value = Labels.MALIGNANT
            prediction.probability = 0.7
            return prediction
        
        mock_inference.side_effect = mock_inference_implementation

        with self.app.app_context():
            result = prediction_controller.create_prediction(
                model_name="test_model",
                model_version=1,
                prediction=prediction
            )
            self.assertIsNotNone(result.id)
            self.assertEqual(result.value, Labels.MALIGNANT)
            self.assertEqual(result.probability, 0.7)

    @patch('model_serving.controllers.prediction.mlflow_gateway')
    def test_create_prediction_invalid_input(self, mock_gateway):
        with self.app.app_context():
            with self.assertRaises(Exception):
                prediction_controller.create_prediction(
                    model_name="test_model",
                    model_version=1,
                    prediction=Prediction(inputs={"feature1": "invalid"})
                )

    @patch('model_serving.controllers.prediction.mlflow_gateway')
    def test_create_prediction_model_not_found(self, mock_gateway):
        mock_gateway.get_model.return_value = None
        
        with self.app.app_context():
            with self.assertRaises(ModelNotFoundException):
                prediction_controller.create_prediction(
                    model_name="nonexistent_model",
                    model_version=1,
                    prediction=Prediction(inputs={"feature1": 1.0})
                )

    @patch('model_serving.controllers.prediction.mlflow_gateway')
    @patch('model_serving.controllers.prediction.input_validator')
    def test_create_prediction_validation_error(self, mock_validator, mock_gateway):
        # Setup mock model
        mock_model = MagicMock()
        mock_gateway.get_model.return_value = mock_model
        
        # Setup validation error
        mock_validator.validate_prediction_input.return_value = ["Invalid input format"]
        
        with self.app.app_context():
            with self.assertRaises(InvalidInputException):
                prediction_controller.create_prediction(
                    model_name="test_model",
                    model_version=1,
                    prediction=Prediction(inputs={"feature1": 1.0})
                )

    @patch('model_serving.controllers.prediction.mlflow_gateway')
    @patch('model_serving.controllers.prediction.input_validator')
    @patch('model_serving.controllers.prediction.inference_service')
    def test_create_prediction_inference_error(self, mock_inference, mock_validator, mock_gateway):
        # Setup mocks
        mock_model = MagicMock()
        mock_gateway.get_model.return_value = mock_model
        mock_validator.validate_prediction_input.return_value = []
        mock_inference.create_inference.side_effect = Exception("Inference failed")
        
        with self.app.app_context():
            with self.assertRaises(InferenceException):
                prediction_controller.create_prediction(
                    model_name="test_model",
                    model_version=1,
                    prediction=Prediction(inputs={"feature1": 1.0})
                )

    @patch('model_serving.controllers.prediction.mlflow_gateway')
    @patch('model_serving.controllers.prediction.input_validator')
    @patch('model_serving.controllers.prediction.inference_service')
    @patch('model_serving.controllers.prediction.db.session')
    def test_create_prediction_database_error(self, mock_session, mock_inference, mock_validator, mock_gateway):
        # Setup mocks
        mock_model = MagicMock()
        mock_gateway.get_model.return_value = mock_model
        mock_validator.validate_prediction_input.return_value = []
        mock_inference.create_inference.return_value = Prediction(inputs={"feature1": 1.0})
        mock_session.commit.side_effect = Exception("Database error")
        
        with self.app.app_context():
            with self.assertRaises(Exception):
                prediction_controller.create_prediction(
                    model_name="test_model",
                    model_version=1,
                    prediction=Prediction(inputs={"feature1": 1.0})
                )
            mock_session.rollback.assert_called_once()

    def test_get_prediction_success(self):
        with self.app.app_context():
            # Create a mock PredictionSQL object
            prediction_sql = PredictionSQL()
            prediction_sql.id = "test_id"
            prediction_sql.inputs = {"feature1": 1.0}
            prediction_sql.value = Labels.MALIGNANT.value
            prediction_sql.probability = 0.9
            
            # Add to database
            self.db.session.add(prediction_sql)
            self.db.session.commit()
            
            # Test retrieval
            result = prediction_controller.get_prediction("test_id")
            self.assertEqual(result.inputs, {"feature1": 1.0})
            self.assertEqual(result.value, Labels.MALIGNANT.value)
            self.assertEqual(result.probability, 0.9)

    def test_get_prediction_not_found(self):
        with self.app.app_context():
            with self.assertRaises(Exception):
                prediction_controller.get_prediction("nonexistent_id")

class TestPredictionValidation(unittest.TestCase):
    """Test input validation and data processing"""
    
    def test_prediction_input_validation(self):
        """Test that prediction inputs are properly validated"""
        # Test valid input
        valid_prediction = Prediction(inputs={"feature1": 1.0})
        self.assertEqual(valid_prediction.inputs, {"feature1": 1.0})
        
        # Test invalid input
        with self.assertRaises(ValueError):
            Prediction(inputs={"feature1": "invalid"})
    
    def test_model_type_validation(self):
        """Test that model types are properly handled"""
        # Test valid model types
        self.assertTrue(ModelType.CLASSIFICATION.value in ["classification", "regression"])
        self.assertTrue(ModelType.REGRESSION.value in ["classification", "regression"])

class TestPredictionProcessing(unittest.TestCase):
    """Test prediction value processing"""
    
    def test_classification_prediction_processing(self):
        """Test processing of classification predictions"""
        prediction = Prediction(inputs={"feature1": 1.0})
        
        # Test setting classification values
        prediction.value = Labels.MALIGNANT
        prediction.probability = 0.7
        
        self.assertEqual(prediction.value, Labels.MALIGNANT)
        self.assertEqual(prediction.probability, 0.7)
    
    def test_regression_prediction_processing(self):
        """Test processing of regression predictions"""
        prediction = Prediction(inputs={"feature1": 1.0})
        
        # Test setting regression values
        prediction.value = 42.0
        prediction.probability = None  # Regression doesn't use probability
        
        self.assertEqual(prediction.value, 42.0)
        self.assertIsNone(prediction.probability)