import enum

class Labels(enum.StrEnum):
    FRAUD = 'FRAUD'
    NOT_FRAUD = 'NOT-FRAUD'
    CANCER = 'CANCER'

    MALIGNANT = "MALIGNANT"
    BENIGN = "BENIGN"

Actuals = Labels