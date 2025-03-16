from unittest.mock import patch, MagicMock
import json
from ..base import BaseTestCase
from model_serving.models.prediction import Prediction
from model_serving.domain.common_enums import Labels, ModelType
from model_serving.domain.exceptions import (
    ModelNotFoundException,
    InvalidInputException,
    ModelTypeException,
    InferenceException
)

class TestPredictionRoutes(BaseTestCase):
    @patch('model_serving.gateways.mlflow_gateway.MLFlowGateway._load_model')
    def setUp(self, mock_load_model):
        # Mock the model loading to avoid MLflow connection
        mock_load_model.return_value = MagicMock()
        super().setUp()
        self.app = self.create_app()
        self.client = self.app.test_client()
        
    def test_create_prediction_success(self):
        with patch('model_serving.controllers.prediction.prediction_controller.create_prediction') as mock_create:
            # Setup mock return value
            mock_prediction = Prediction(inputs={"feature1": 1.0})
            mock_prediction.value = Labels.MALIGNANT
            mock_prediction.probability = 0.9
            mock_prediction.shap_values = None
            mock_create.return_value = mock_prediction
            
            response = self.client.post(
                '/test_model/version/1/predict',
                json={'features': {"feature1": 1.0}}
            )
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['value'], Labels.MALIGNANT.value)
            self.assertEqual(data['probability'], 0.9)
    
    def test_create_prediction_model_not_found(self):
        with patch('model_serving.controllers.prediction.prediction_controller.create_prediction') as mock_create:
            mock_create.side_effect = ModelNotFoundException(model_name="test_model", model_version="1")
            
            response = self.client.post(
                '/test_model/version/1/predict',
                json={'features': {"feature1": 1.0}}
            )
            
            self.assertEqual(response.status_code, 404)
            self.assertIn("Model Not Found", response.json['error'])
    
    def test_create_prediction_invalid_input(self):
        with patch('model_serving.controllers.prediction.prediction_controller.create_prediction') as mock_create:
            mock_create.side_effect = InvalidInputException("Invalid input type")
            
            response = self.client.post(
                '/test_model/version/1/predict',
                json={'features': {"feature1": "invalid"}}
            )
            
            self.assertEqual(response.status_code, 400)
            self.assertIn("Invalid Input", response.json['error'])
    
    def test_create_prediction_missing_features(self):
        response = self.client.post(
            '/test_model/version/1/predict',
            json={}
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_create_prediction_invalid_json(self):
        response = self.client.post(
            '/test_model/version/1/predict',
            data="invalid json",
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_get_prediction_success(self):
        with patch('model_serving.controllers.prediction.prediction_controller.get_prediction') as mock_get:
            mock_prediction = Prediction(inputs={"feature1": 1.0})
            mock_prediction.value = Labels.MALIGNANT
            mock_prediction.probability = 0.9
            mock_prediction.shap_values = None
            mock_get.return_value = mock_prediction
            
            response = self.client.get('/prediction/test_id')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['value'], Labels.MALIGNANT.value)
            self.assertEqual(data['probability'], 0.9)

    def test_get_prediction_not_found(self):
        with patch('model_serving.controllers.prediction.prediction_controller.get_prediction') as mock_get:
            mock_get.side_effect = ModelNotFoundException(model_name="test_model", model_version="1")
            
            response = self.client.get('/prediction/nonexistent_id')
            
            self.assertEqual(response.status_code, 404)
            self.assertIn("Model Not Found", response.json['error'])
