from typing import Optional, Dict, Any

class ModelServingException(Exception):
    """Base exception class for model serving errors"""
    def __init__(self, message: str, status_code: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class ModelNotFoundException(ModelServingException):
    """Raised when requested model is not found"""
    def __init__(self, model_name: str, model_version: str):
        message = f"Model {model_name} version {model_version} not found"
        super().__init__(message=message, status_code=404)

class InvalidInputException(ModelServingException):
    """Raised when input validation fails"""
    def __init__(self, validation_errors: Dict[str, str]):
        message = "Invalid input data"
        super().__init__(message=message, status_code=400, details={"validation_errors": validation_errors})

class ModelTypeException(ModelServingException):
    """Raised when there's a model type mismatch"""
    def __init__(self, expected_type: str, received_type: str):
        message = f"Model type mismatch. Expected {expected_type}, got {received_type}"
        super().__init__(message=message, status_code=400)

class InferenceException(ModelServingException):
    """Raised when prediction fails"""
    def __init__(self, message: str):
        super().__init__(message=message, status_code=500) 