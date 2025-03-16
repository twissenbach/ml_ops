from model_serving.domain.common_enums import ModelType, Labels

class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # Use in-memory database for testing
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TRACKING_URI = 'http://localhost:5000'  # Mock MLflow tracking URI for testing
    
    # Mock model configuration for testing
    MODELS = {
        'test_model': {
            '1': {
                'model_type': ModelType.REGRESSION.value,
                'threshold': {'value': 0.5, 'above': Labels.BENIGN.value, 'below': Labels.MALIGNANT.value, 'equal': Labels.BENIGN.value},
                'mlflow_flavor': 'sklearn',
                'model': None  # Mock model will be set during tests
            }
        }
    }
