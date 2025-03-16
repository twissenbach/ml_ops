import unittest
from unittest.mock import patch, MagicMock
from model_serving.gateways.mlflow_gateway import MLFlowGateway
from model_serving.domain.common_enums import ModelType

class TestMLFlowGateway(unittest.TestCase):
    def setUp(self):
        self.gateway = MLFlowGateway()
        self.test_config = {
            'TRACKING_URI': 'http://test-mlflow-server',
            'MODELS': {
                'test_model': {
                    '1': {
                        'model_type': ModelType.CLASSIFICATION.value,
                        'mlflow_flavor': 'sklearn',
                        'threshold': {'value': 0.5, 'above': 'BENIGN', 'below': 'MALIGNANT', 'equal': 'BENIGN'},
                        'labels': ['BENIGN', 'MALIGNANT']
                    },
                    '2': {
                        'model_type': ModelType.REGRESSION.value,
                        'mlflow_flavor': 'pytorch',
                        'explainer': True
                    }
                }
            }
        }

    @patch('model_serving.gateways.mlflow_gateway.mlflow')
    def test_init_app(self, mock_mlflow):
        # Create mock app
        mock_app = MagicMock()
        mock_app.config = self.test_config

        # Mock model loading
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        mock_mlflow.pytorch.load_model.return_value = mock_model

        # Mock MLflow client for explainer
        mock_client = MagicMock()
        mock_client.get_model_version.return_value = MagicMock(run_id='test_run_id')
        mock_mlflow.client.MlflowClient.return_value = mock_client

        # Initialize gateway
        self.gateway.init_app(mock_app)

        # Verify MLflow tracking URI was set
        mock_mlflow.set_tracking_uri.assert_called_once_with('http://test-mlflow-server/mlruns')

        # Verify models were loaded
        self.assertIn('test_model', self.gateway.models)
        self.assertIn('1', self.gateway.models['test_model'])
        self.assertIn('2', self.gateway.models['test_model'])

    def test_get_model_uri(self):
        uri = self.gateway._get_model_uri('test_model', '1')
        self.assertEqual(uri, 'models:/test_model/1')

    def test_get_explainer_uri(self):
        uri = self.gateway._get_explainer_uri('test_run_id')
        self.assertEqual(uri, 'runs:/test_run_id/model/explainer')

    @patch('model_serving.gateways.mlflow_gateway.mlflow')
    def test_load_model_sklearn(self, mock_mlflow):
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        result = self.gateway._load_model('test_uri', 'sklearn')
        
        mock_mlflow.sklearn.load_model.assert_called_once_with('test_uri')
        self.assertEqual(result, mock_model)

    @patch('model_serving.gateways.mlflow_gateway.mlflow')
    def test_load_model_tensorflow(self, mock_mlflow):
        mock_model = MagicMock()
        mock_mlflow.tensorflow.load_model.return_value = mock_model
        
        result = self.gateway._load_model('test_uri', 'tensorflow')
        
        mock_mlflow.tensorflow.load_model.assert_called_once_with('test_uri')
        self.assertEqual(result, mock_model)

    @patch('model_serving.gateways.mlflow_gateway.mlflow')
    def test_load_model_pytorch(self, mock_mlflow):
        mock_model = MagicMock()
        mock_mlflow.pytorch.load_model.return_value = mock_model
        
        result = self.gateway._load_model('test_uri', 'pytorch')
        
        mock_mlflow.pytorch.load_model.assert_called_once_with('test_uri')
        self.assertEqual(result, mock_model)

    @patch('model_serving.gateways.mlflow_gateway.mlflow')
    def test_load_model_xgboost(self, mock_mlflow):
        mock_model = MagicMock()
        mock_mlflow.xgboost.load_model.return_value = mock_model
        
        result = self.gateway._load_model('test_uri', 'xgboost')
        
        mock_mlflow.xgboost.load_model.assert_called_once_with('test_uri')
        self.assertEqual(result, mock_model)

    @patch('model_serving.gateways.mlflow_gateway.mlflow')
    def test_load_model_pyfunc(self, mock_mlflow):
        mock_model = MagicMock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        result = self.gateway._load_model('test_uri', 'pyfunc')
        
        mock_mlflow.pyfunc.load_model.assert_called_once_with('test_uri')
        self.assertEqual(result, mock_model)

    def test_load_model_unsupported_flavor(self):
        with self.assertRaises(ValueError) as context:
            self.gateway._load_model('test_uri', 'unsupported_flavor')
        self.assertIn('Unsupported model flavor', str(context.exception))

    def test_get_model(self):
        # Setup test data
        self.gateway.models = {
            'test_model': {
                '1': {
                    'model_type': ModelType.CLASSIFICATION.value,
                    'model': MagicMock(),
                    'threshold': {'value': 0.5, 'above': 'BENIGN', 'below': 'MALIGNANT', 'equal': 'BENIGN'},
                    'labels': ['BENIGN', 'MALIGNANT']
                }
            }
        }

        # Get model
        model = self.gateway.get_model('test_model', '1')

        # Verify model properties
        self.assertEqual(model.model_type, ModelType.CLASSIFICATION.value)
        self.assertEqual(model.model_name, 'test_model')
        self.assertEqual(model.model_version, '1')
        self.assertEqual(model.threshold['value'], 0.5)
        self.assertIsNotNone(model.model)

    def test_get_model_regression(self):
        # Setup test data
        self.gateway.models = {
            'test_model': {
                '1': {
                    'model_type': ModelType.REGRESSION.value,
                    'model': MagicMock(),
                }
            }
        }

        # Get model
        model = self.gateway.get_model('test_model', '1')

        # Verify model properties
        self.assertEqual(model.model_type, ModelType.REGRESSION.value)
        self.assertIsNone(model.threshold)
        self.assertIsNone(model.labels)
