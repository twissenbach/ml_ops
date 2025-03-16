import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from model_serving.services.inference_service import inference_service
from model_serving.models.prediction import Prediction, Model
from model_serving.domain.common_enums import ModelType, Labels

class TestInferenceService(unittest.TestCase):
    def setUp(self):
        # Create sample inputs
        self.sample_inputs = {"feature1": 1.0, "feature2": 2.0}
        self.sample_prediction = Prediction(inputs=self.sample_inputs)

    def test_classification_with_probability(self):
        # Create mock model with predict_proba
        mock_model = MagicMock()
        mock_model.model_type = ModelType.CLASSIFICATION.value
        mock_model.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.threshold = {
            'value': 0.5,
            'above': Labels.BENIGN.value,
            'below': Labels.MALIGNANT.value,
            'equal': Labels.BENIGN.value
        }

        # Test prediction above threshold
        mock_model.model.predict_proba.return_value = np.array([[0.3, 0.8]])
        prediction = inference_service.create_inference(mock_model, self.sample_prediction)
        self.assertEqual(prediction.label, Labels.MALIGNANT.value)
        self.assertEqual(prediction.probability, 0.8)
        self.assertEqual(prediction.threshold, 0.5)

        # Test prediction below threshold
        mock_model.model.predict_proba.return_value = np.array([[0.3, 0.3]])
        prediction = inference_service.create_inference(mock_model, self.sample_prediction)
        self.assertEqual(prediction.label, Labels.BENIGN.value)
        self.assertEqual(prediction.probability, 0.3)

        # Test prediction equal to threshold
        mock_model.model.predict_proba.return_value = np.array([[0.3, 0.5]])
        prediction = inference_service.create_inference(mock_model, self.sample_prediction)
        self.assertEqual(prediction.label, Labels.BENIGN.value)
        self.assertEqual(prediction.probability, 0.5)

    def test_classification_without_probability(self):
        # Create mock model without predict_proba
        mock_model = MagicMock()
        mock_model.model_type = ModelType.CLASSIFICATION.value
        mock_model._model.predict.return_value = Labels.MALIGNANT.value
        
        # Remove predict_proba attribute
        del mock_model.model.predict_proba

        prediction = inference_service.create_inference(mock_model, self.sample_prediction)
        self.assertEqual(prediction.label, Labels.MALIGNANT.value)

    def test_regression_prediction(self):
        # Create mock regression model
        mock_model = MagicMock()
        mock_model.model_type = ModelType.REGRESSION.value
        mock_model.model.predict.return_value = np.array([42.0])

        prediction = inference_service.create_inference(mock_model, self.sample_prediction)
        self.assertEqual(prediction.value, 42.0)
        self.assertIsNone(prediction.probability)

    def test_prepare_inputs(self):
        # Test input preparation
        inputs = {"feature1": 1.0, "feature2": 2.0}
        expected = np.array([[1.0, 2.0]])
        
        result = inference_service._prepare_inputs(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_pandas_frame_conversion(self):
        # Test that inputs are correctly converted to pandas DataFrame
        inputs = {"feature1": 1.0, "feature2": 2.0}
        prediction = Prediction(inputs=inputs)
        
        df = prediction.get_pandas_frame_of_inputs()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (1, 2))
        self.assertEqual(df.iloc[0]["feature1"], 1.0)
        self.assertEqual(df.iloc[0]["feature2"], 2.0)

    def test_invalid_model_type(self):
        # Test handling of invalid model type
        mock_model = MagicMock()
        mock_model.model_type = "invalid_type"

        with self.assertRaises(ValueError):
            inference_service.create_inference(mock_model, self.sample_prediction)

    @patch('model_serving.monitoring.FUNCTION_DURATION')
    def test_monitoring_metrics(self, mock_duration):
        # Test that monitoring metrics are recorded
        mock_model = MagicMock()
        mock_model.model_type = ModelType.REGRESSION.value
        mock_model.model.predict.return_value = np.array([42.0])

        inference_service.create_inference(mock_model, self.sample_prediction)
        mock_duration.labels.assert_called_once_with('create_prediction')
