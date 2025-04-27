import logging
import numpy as np
import umap
from tempfile import TemporaryDirectory
import pickle
import os

from model_serving.models.prediction import Model, Prediction
from model_serving.monitoring import FUNCTION_DURATION

logger = logging.getLogger(__name__)

class UMAPService:
    _umap_models = {}  # Store UMAP models by model_name/version for persistence
    
    @staticmethod
    def _get_default_umap_params():
        return {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "random_state": 42
        }
    
    @staticmethod
    @FUNCTION_DURATION.labels('create_umap_embedding').time()
    def create_embedding(model: Model, prediction: Prediction, umap_params=None) -> np.ndarray:
        try:
            model_key = f"{model.model_name}/{model.model_version}"
            inputs_array = prediction.get_pandas_frame_of_inputs()
            
            # Use provided params or defaults
            params = umap_params or UMAPService._get_default_umap_params()
            
            # Use existing UMAP model or create a new one
            if model_key not in UMAPService._umap_models:
                UMAPService._umap_models[model_key] = umap.UMAP(**params)
                # Note: For actual training, you would need historical data
                # This is just a placeholder for the transform-only case
            
            # Calculate embeddings
            embeddings = UMAPService._umap_models[model_key].transform(inputs_array)
            
            return embeddings.tolist()[0]  # Return as list for JSON serialization
            
        except Exception as e:
            # Log error but don't fail the prediction
            logger.error(f"Error calculating UMAP embeddings: {str(e)}")
            return None

umap_service = UMAPService()
