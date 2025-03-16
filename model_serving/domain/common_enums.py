import enum
from enum import Enum

class Labels(enum.StrEnum):
    FRAUD = 'FRAUD'
    NOT_FRAUD = 'NOT-FRAUD'
    CANCER = 'CANCER'

    MALIGNANT = "MALIGNANT"
    BENIGN = "BENIGN"

Actuals = Labels

class ModelType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"