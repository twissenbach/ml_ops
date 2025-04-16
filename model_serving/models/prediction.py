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
    inputs: Dict[str, float] = field(default_factory=dict)
    _value: Union[float, Labels, None] = None
    _probability: Union[float, None] = None
    confidence_interval: Optional[tuple[float, float]] = None
    actual: Union[float, None] = None
    threshold: Union[float, None] = None
    shap_values: List[Shap] = field(default_factory=list)
    model: Model = None
    metadata: dict = field(default_factory=dict)
    _input_key: str = None

    def __post_init__(self):
        self._validate_inputs(self.inputs)

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
        result = {
            "inputs": self.inputs,
            "metadata": self.metadata
        }
        
        if self.model and self.model.model_type == ModelType.REGRESSION:
            result["label"] = float(self.value) if self.value is not None else None
        else:  # CLASSIFICATION
            result["label"] = self.value.value if isinstance(self.value, Labels) else self.value
        
        if self.probability is not None:
            result["probability"] = float(self.probability)
        
        # Add SHAP values to the response
        if self.shap_values and len(self.shap_values) > 0:
            shap_dict = {}
            for shap_obj in self.shap_values:
                if shap_obj.shap_values:  # Use shap_values instead of shap
                    if isinstance(shap_obj.shap_values, dict):
                        shap_dict = shap_obj.shap_values
                    else:
                        # Handle numpy array format
                        feature_names = list(self.inputs.keys())
                        shap_values_array = shap_obj.shap_values[0] if hasattr(shap_obj.shap_values, 'ndim') and shap_obj.shap_values.ndim > 1 else shap_obj.shap_values
                        for i, feature in enumerate(feature_names):
                            if i < len(shap_values_array):
                                shap_dict[feature] = float(shap_values_array[i])
            
            if shap_dict:
                result["shap_values"] = shap_dict
        
        return result
