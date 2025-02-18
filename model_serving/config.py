from pathlib import Path
import os

from model_serving.domain.common_enums import Labels
class Config:
    ENV = os.environ.get('FLASK_ENV', 'development')
    APP_NAME = 'user_application'
    # WORKING_DIRECTORY = str(Path(__file__).parent / 'user_application')
    WORKING_DIRECTORY = current_path = str(Path(__name__).resolve().parent) + '\\model_serving\\user_application'

    # logger
    # LOG_LEVEL = 'INFO'
    LOG_LEVEL = 'DEBUG'

    # SQL Alchemy
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///users.db'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///user_application.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # MLFlow
    TRACKING_URI = "/Users/troywissenbach/Documents/spring_25_classes" # NEED TO DOUBLE CHECK THIS PATH
    MODELS = {
        "breast_cancer_random_forest_model": {
            "1": {
                "model": None,
                "model_type": "SCORE_CATEGORICAL", # Monitoring
                "mlflow_flavor": "sklearn",
                "labels": [Labels.BENIGN, Labels.MALIGNANT],
                "threshold": {
                    "value": 0.5,
                    "above": Labels.MALIGNANT,
                    "equal": Labels.BENIGN,
                    "below": Labels.BENIGN
                }
            }
        }
    }


class UnitTest(Config):
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

    
class Production(Config):
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

environments = {
    'development': Config,
    'unittest': UnitTest,
    'production': Production
}