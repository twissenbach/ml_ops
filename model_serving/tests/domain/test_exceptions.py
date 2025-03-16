import unittest
from model_serving.domain.exceptions import (
    ModelNotFoundException,
    InvalidInputException,
    ModelTypeException,
    InferenceException
)

class TestExceptions(unittest.TestCase):
    def test_model_not_found_exception(self):
        exception = ModelNotFoundException("test_model", "1")
        self.assertEqual(exception.status_code, 404)
        self.assertEqual(str(exception), "Model test_model version 1 not found")

    def test_invalid_input_exception(self):
        validation_errors = {"feature1": "must be numeric"}
        exception = InvalidInputException(validation_errors)
        self.assertEqual(exception.status_code, 400)
        self.assertEqual(exception.details["validation_errors"], validation_errors)

    def test_model_type_exception(self):
        exception = ModelTypeException("regression", "classification")
        self.assertEqual(exception.status_code, 400)
        self.assertTrue("Expected regression, got classification" in str(exception))

    def test_inference_exception(self):
        error_message = "Failed to create prediction"
        exception = InferenceException(error_message)
        self.assertEqual(exception.status_code, 500)
        self.assertEqual(str(exception), error_message)