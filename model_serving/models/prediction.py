from typing import Union, List, Optional, Dict
import pandas as pd
import hashlib
import uuid

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from model_serving.domain.common_enums import ModelType, Labels, Actuals


def get_id():
    return uuid.uuid4().hex


@dataclass_json
@dataclass
class Model:
    id: str = field(default_factory=get_id)
    model_name: str = None
    model_version: str = None
    model_type: ModelType = None
    threshold: Union[float, None] = None
    labels: Union[None, List[Labels]] = None
    _model = None
    _explainer = None

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5((self.model_name + str(self.model_version)).encode()).hexdigest()



@dataclass_json
@dataclass
class Shap:
    id: str = field(default_factory=get_id)
    label: Union[None, Labels] = None
    shap_values: dict = field(default_factory=dict)


@dataclass_json
@dataclass
class Prediction:
    id: str = field(default_factory=get_id)
    inputs: dict = field(default_factory=dict)
    value: Union[float, Labels, None] = None
    probability: Union[float, None] = None
    confidence_interval: Optional[tuple[float, float]] = None
    actual: Union[float, None] = None
    threshold: Union[float, None] = None
    shap_values: List[Shap] = field(default_factory=list)
    model: Model = None
    metadata: dict = field(default_factory=dict)

    _input_key: str = None

    def __init__(self, 
                 inputs: Dict[str, float], 
                 value: Union[Labels, float, None] = None,
                 probability: Optional[float] = None):
        self._validate_inputs(inputs)
        self.inputs = inputs
        self.id = uuid.uuid4().hex
        self._value = value
        self._probability = probability
        self.actual = None
        self.model = None

    def _validate_inputs(self, inputs: Dict[str, float]):
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary")
        
        for key, value in inputs.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Input '{key}' must be numeric, got {type(value)}")

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, val):
        self._probability = val

    def get_pandas_frame_of_inputs(self):

        return pd.DataFrame([self.inputs], index=[0])

    def __repr__(self):
        return f'Prediction ID: {self.id}'

    def to_json(self):
        if self.model.model_type == ModelType.REGRESSION:
            return {
                "label": float(self.value)
            }
        else:  # CLASSIFICATION
            return {
                "label": self.value.value if isinstance(self.value, Labels) else self.value
            }
