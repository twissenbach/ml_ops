import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from model_serving.services.umap_service import umap_service, UMAPService
from model_serving.models.prediction import Prediction, Model
from model_serving.domain.common_enums import ModelType, Labels

class TestUMAPService(unittest.TestCase):
    def setUp(self):
        # Create sample inputs
        self.sample_inputs = {"feature1": 1.0, "feature2": 2.0}
        self.sample_prediction = Prediction(inputs=self.sample_inputs)
        self.sample_model = Model(
            model_name="test_model",
            model_version="1",
            model_type=ModelType.CLASSIFICATION,
            labels=[Labels.BENIGN, Labels.MALIGNANT]
        )
        
        # Clear any cached models before each test
        UMAPService._umap_models = {}

    @patch('umap.UMAP')
    def test_create_embedding_success(self, mock_umap):
        # Mock the UMAP transform method
        mock_umap_instance = MagicMock()
        mock_umap_instance.transform.return_value = np.array([[0.1, 0.2]])
        mock_umap.return_value = mock_umap_instance
        
        # Call the service
        embeddings = umap_service.create_embedding(self.sample_model, self.sample_prediction)
        
        # Verify correct embeddings are returned
        self.assertEqual(embeddings, [0.1, 0.2])
        
        # Verify UMAP was created with default parameters
        mock_umap.assert_called_once()
        default_params = UMAPService._get_default_umap_params()
        for key, value in default_params.items():
            self.assertEqual(mock_umap.call_args[1][key], value)
        
        # Verify transform was called with the inputs
        mock_umap_instance.transform.assert_called_once()
        self.assertTrue(isinstance(mock_umap_instance.transform.call_args[0][0], pd.DataFrame))

    @patch('umap.UMAP')
    def test_create_embedding_with_custom_params(self, mock_umap):
        # Mock the UMAP transform method
        mock_umap_instance = MagicMock()
        mock_umap_instance.transform.return_value = np.array([[0.1, 0.2]])
        mock_umap.return_value = mock_umap_instance
        
        # Custom parameters
        custom_params = {
            "n_neighbors": 10,
            "min_dist": 0.2,
            "n_components": 3,
            "random_state": 123
        }
        
        # Call the service with custom parameters
        embeddings = umap_service.create_embedding(
            self.sample_model, 
            self.sample_prediction, 
            umap_params=custom_params
        )
        
        # Verify UMAP was created with custom parameters
        mock_umap.assert_called_once()
        for key, value in custom_params.items():
            self.assertEqual(mock_umap.call_args[1][key], value)

    @patch('umap.UMAP')
    def test_model_persistence(self, mock_umap):
        # Mock the UMAP transform method
        mock_umap_instance = MagicMock()
        mock_umap_instance.transform.return_value = np.array([[0.1, 0.2]])
        mock_umap.return_value = mock_umap_instance
        
        # Call the service twice with the same model
        umap_service.create_embedding(self.sample_model, self.sample_prediction)
        umap_service.create_embedding(self.sample_model, self.sample_prediction)
        
        # Verify UMAP was only created once
        mock_umap.assert_called_once()
        
        # Verify transform was called twice
        self.assertEqual(mock_umap_instance.transform.call_count, 2)
        
        # Verify the model was cached
        model_key = f"{self.sample_model.model_name}/{self.sample_model.model_version}"
        self.assertIn(model_key, UMAPService._umap_models)

    @patch('umap.UMAP')
    def test_handle_different_models(self, mock_umap):
        # Mock the UMAP transform method
        mock_umap_instance1 = MagicMock()
        mock_umap_instance1.transform.return_value = np.array([[0.1, 0.2]])
        mock_umap_instance2 = MagicMock()
        mock_umap_instance2.transform.return_value = np.array([[0.3, 0.4]])
        
        mock_umap.side_effect = [mock_umap_instance1, mock_umap_instance2]
        
        # Create a second model
        second_model = Model(
            model_name="test_model",
            model_version="2",  # Different version
            model_type=ModelType.CLASSIFICATION,
            labels=[Labels.BENIGN, Labels.MALIGNANT]
        )
        
        # Call the service with different models
        embeddings1 = umap_service.create_embedding(self.sample_model, self.sample_prediction)
        embeddings2 = umap_service.create_embedding(second_model, self.sample_prediction)
        
        # Verify different embeddings
        self.assertEqual(embeddings1, [0.1, 0.2])
        self.assertEqual(embeddings2, [0.3, 0.4])
        
        # Verify UMAP was created twice for different models
        self.assertEqual(mock_umap.call_count, 2)
        
        # Verify both models were cached
        self.assertEqual(len(UMAPService._umap_models), 2)

    @patch('umap.UMAP')
    def test_error_handling(self, mock_umap):
        # Mock UMAP to raise an exception
        mock_umap_instance = MagicMock()
        mock_umap_instance.transform.side_effect = Exception("Test error")
        mock_umap.return_value = mock_umap_instance
        
        # Call the service
        with self.assertLogs(level='ERROR') as log:
            embeddings = umap_service.create_embedding(self.sample_model, self.sample_prediction)
            
            # Verify error is logged
            self.assertTrue(any("Error calculating UMAP embeddings" in message for message in log.output))
            
            # Verify None is returned on error
            self.assertIsNone(embeddings)

    @patch('model_serving.monitoring.FUNCTION_DURATION')
    @patch('umap.UMAP')
    def test_monitoring_metrics(self, mock_umap, mock_duration):
        # Mock the UMAP transform method
        mock_umap_instance = MagicMock()
        mock_umap_instance.transform.return_value = np.array([[0.1, 0.2]])
        mock_umap.return_value = mock_umap_instance
        
        # Mock timer
        mock_timer = MagicMock()
        mock_duration.labels.return_value = mock_timer
        
        # Call the service
        umap_service.create_embedding(self.sample_model, self.sample_prediction)
        
        # Verify monitoring metrics are recorded
        mock_duration.labels.assert_called_once_with('create_umap_embedding')
        mock_timer.time.assert_called_once()
