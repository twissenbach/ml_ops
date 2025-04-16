import pytest
import numpy as np
from unittest.mock import Mock, patch

from model_serving.models.prediction import Model, Prediction, Shap
from model_serving.services.explainer_service import ExplainerService
from model_serving.domain.common_enums import ModelType, Labels


class TestExplainerService:
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "feature_1": 0.5,
            "feature_2": -1.0,
            "feature_3": 2.0
        }

    @pytest.fixture
    def mock_model(self):
        model = Model(
            model_name="test_model",
            model_version="1.0",
            model_type=ModelType.CLASSIFICATION
        )
        # Create a mock explainer that returns SHAP values
        model._explainer = Mock()
        return model

    def test_create_explanation_empty_shap_values(self, sample_inputs, mock_model):
        """Test handling of empty SHAP values"""
        prediction = Prediction(inputs=sample_inputs)
        mock_model._explainer.predict.return_value = []
        
        result = ExplainerService.create_explanation(mock_model, prediction)
        assert len(result.shap_values) == 0

    def test_create_explanation_regression(self, sample_inputs, mock_model):
        """Test handling regression model SHAP values"""
        prediction = Prediction(inputs=sample_inputs)
        mock_model.model_type = ModelType.REGRESSION
        mock_model.labels = None
        
        # Shape: [1 sample, 3 features, 1 output]
        mock_shap_values = np.zeros((1, 3, 1))
        mock_shap_values[0, :, 0] = [0.1, 0.2, 0.3]
        mock_model._explainer.predict.return_value = mock_shap_values
        
        result = ExplainerService.create_explanation(mock_model, prediction)
        
        assert len(result.shap_values) == 1
        shap_dict = result.shap_values[0].shap_values
        assert shap_dict == {
            "feature_1": 0.1,
            "feature_2": 0.2,
            "feature_3": 0.3
        }

    def test_create_explanation_classification(self, sample_inputs, mock_model):
        """Test handling classification model SHAP values"""
        prediction = Prediction(inputs=sample_inputs)
        mock_model.labels = [Labels.MALIGNANT, Labels.BENIGN]
        
        # Shape: [1 sample, 3 features, 2 classes]
        mock_shap_values = np.zeros((1, 3, 2))
        mock_shap_values[0, :, 0] = [0.1, 0.2, 0.3]  # MALIGNANT class
        mock_shap_values[0, :, 1] = [-0.1, -0.2, -0.3]  # BENIGN class
        mock_model._explainer.predict.return_value = mock_shap_values
        
        result = ExplainerService.create_explanation(mock_model, prediction)
        
        assert len(result.shap_values) == 2
        malignant_shap = result.shap_values[0].shap_values
        benign_shap = result.shap_values[1].shap_values
        
        assert malignant_shap == {
            "feature_1": 0.1,
            "feature_2": 0.2,
            "feature_3": 0.3
        }
        assert benign_shap == {
            "feature_1": -0.1,
            "feature_2": -0.2,
            "feature_3": -0.3
        }

    def test_create_explanation_error_handling(self, sample_inputs, mock_model):
        """Test error handling during SHAP calculation"""
        prediction = Prediction(inputs=sample_inputs)
        mock_model._explainer.predict.side_effect = Exception("SHAP calculation failed")
        
        result = ExplainerService.create_explanation(mock_model, prediction)
        assert result == prediction
        assert len(result.shap_values) == 0

    def test_create_explanation_out_of_bounds(self, sample_inputs, mock_model):
        """Test handling when SHAP values array is smaller than feature count"""
        prediction = Prediction(inputs=sample_inputs)
        mock_model.labels = None
        
        # Only 2 SHAP values for 3 features
        mock_shap_values = np.zeros((1, 2, 1))
        mock_shap_values[0, :, 0] = [0.1, 0.2]
        mock_model._explainer.predict.return_value = mock_shap_values
        
        result = ExplainerService.create_explanation(mock_model, prediction)
        
        assert len(result.shap_values) == 1
        shap_dict = result.shap_values[0].shap_values
        assert shap_dict == {
            "feature_1": 0.1,
            "feature_2": 0.2
        }

    def test_prediction_to_json_with_shap(self, sample_inputs, mock_model):
        # Setup
        prediction = Prediction(inputs=sample_inputs)
        prediction.model = mock_model
        prediction.value = Labels.MALIGNANT
        
        # Add some SHAP values
        shap_dict = {
            "feature_1": 0.35,
            "feature_2": -0.12,
            "feature_3": 0.05
        }
        prediction.shap_values = [Shap(
            label=Labels.MALIGNANT.value,
            shap_values=shap_dict
        )]

        # Execute
        result = prediction.to_json()

        # Assert
        assert "label" in result
        assert "shap_values" in result
        assert result["label"] == Labels.MALIGNANT.value
        assert result["shap_values"] == shap_dict

    def test_create_explanation_empty_response(self, sample_inputs, mock_model):
        """Test handling of empty SHAP values from explainer"""
        prediction = Prediction(inputs=sample_inputs)
        mock_model._explainer.predict.return_value = []

        result = ExplainerService.create_explanation(mock_model, prediction)
        
        assert result == prediction
        assert len(result.shap_values) == 0

    def test_create_explanation_none_response(self, sample_inputs, mock_model):
        """Test handling of None response from explainer"""
        prediction = Prediction(inputs=sample_inputs)
        mock_model._explainer.predict.return_value = None

        result = ExplainerService.create_explanation(mock_model, prediction)
        
        assert result == prediction
        assert len(result.shap_values) == 0

    def test_create_explanation_different_shapes(self, sample_inputs, mock_model):
        """Test handling of different SHAP value array shapes"""
        prediction = Prediction(inputs=sample_inputs)
        
        # Test 2D array shape
        mock_shap_values_2d = np.array([[0.1, 0.2, 0.3]])
        mock_model._explainer.predict.return_value = mock_shap_values_2d
        
        result = ExplainerService.create_explanation(mock_model, prediction)
        assert len(result.shap_values) > 0
        assert isinstance(result.shap_values[0].shap_values, dict)

    def test_prediction_to_json_no_shap(self, sample_inputs, mock_model):
        """Test JSON response when no SHAP values are present"""
        prediction = Prediction(inputs=sample_inputs)
        prediction.model = mock_model
        prediction.value = Labels.MALIGNANT
        
        result = prediction.to_json()
        
        assert "label" in result
        assert result["label"] == Labels.MALIGNANT.value
        assert "shap_values" not in result

    def test_prediction_to_json_regression(self, sample_inputs, mock_model):
        """Test JSON response for regression predictions"""
        prediction = Prediction(inputs=sample_inputs)
        prediction.model = mock_model
        prediction.model.model_type = ModelType.REGRESSION
        prediction.value = 0.75
        
        shap_dict = {"feature_1": 0.35, "feature_2": -0.12, "feature_3": 0.05}
        prediction.shap_values = [Shap(label=None, shap_values=shap_dict)]
        
        result = prediction.to_json()
        
        assert "label" in result
        assert isinstance(result["label"], float)
        assert result["shap_values"] == shap_dict

    @pytest.mark.parametrize("invalid_shape", [
        np.array([0.1]),  # 1D array
        np.array([[[[0.1]]]]),  # 4D array
    ])
    def test_create_explanation_invalid_shapes(self, sample_inputs, mock_model, invalid_shape):
        """Test handling of invalid SHAP value array shapes"""
        prediction = Prediction(inputs=sample_inputs)
        mock_model._explainer.predict.return_value = invalid_shape
        
        result = ExplainerService.create_explanation(mock_model, prediction)
        assert result == prediction  # Should still return a prediction
