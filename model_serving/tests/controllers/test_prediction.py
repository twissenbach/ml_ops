from sklearn.datasets import load_breast_cancer
import pandas as pd

from tests.base import BaseTestCase, create_metrics_dir

create_metrics_dir()

from model_serving.controllers.prediction import prediction_controller
from model_serving.models.prediction import Prediction
from model_serving.domain.common_enums import Labels


class TestPredictionController(BaseTestCase):

    def setUp(self):

        super().setUp()

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

    def test_create_prediction(self, *args, **kwargs):

        model_name = 'breast_cancer_random_forest_model'
        model_version = '1'
        prediction = Prediction(
            inputs=self.sample_row
        )

        with self.app.app_context():

            prediction = prediction_controller.create_prediction(model_name, model_version, prediction)

        self.assertEqual(Labels.BENIGN.value, prediction.label)