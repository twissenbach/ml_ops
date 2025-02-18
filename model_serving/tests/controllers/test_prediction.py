from model_serving.controllers.prediction import prediction_controller
from model_serving.models.prediction import Prediction
from model_serving.domain.common_enums import Labels
from sklearn.datasets import load_breast_cancer
import pandas as pd

from model_serving.tests.base import BaseTestCase

class TestPredictionController(BaseTestCase):

    def setUp(self):
        super().setUp()  # Call parent's setUp
        self.create_app()

        # Load the breast cancer dataset
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        # Create a sample row for testing
        self.sample_row = pd.DataFrame([X[0]], columns=data.feature_names)

    def test_create_prediction(self, *args, **kwargs):
        model_name = "breast_cancer_random_forest_model"
        model_version = "1"
        
        prediction = Prediction(
            inputs=self.sample_row.to_dict(orient='records')[0]
        )

        prediction = prediction_controller.create_prediction(model_name, model_version, prediction)

        # Assert that we got a prediction back
        self.assertIsNotNone(prediction.label)
        # Assert that the label is one of the expected values
        self.assertIn(prediction.label, [Labels.BENIGN, Labels.MALIGNANT])
        # Assert that we got probabilities
        self.assertIsNotNone(prediction.probabilities)
        # Assert that we got a threshold
        self.assertIsNotNone(prediction.threshold)