from typing import Union, List
import pandas as pd
import hashlib
import uuid

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from model_serving.domain.common_enums import Labels, Actuals


def get_id():
    return uuid.uuid4().hex


@dataclass_json
@dataclass
class Model:
    id: str = field(default_factory=get_id)
    model_name: str = None
    model_version: str = None
    model_type: str = None
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
    label: Union[Labels, int, float, None] = None
    probability: Union[float, None] = None
    actual: Union[Actuals, int, float, None] = None
    threshold: Union[float, None] = None
    shap_values: List[Shap] = field(default_factory=dict)

    model: Model = None

    _input_key: str = None

    def get_pandas_frame_of_inputs(self):

        return pd.DataFrame([self.inputs], index=[0])

    def __repr__(self):
        return f'Prediction ID: {self.id}'
