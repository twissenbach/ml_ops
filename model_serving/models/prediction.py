from typing import Union
import hashlib
import pandas as pd

from dataclasses import dataclass, field
import uuid


@dataclass
class Model:
    id: str = uuid.uuid4().hex
    model_name: str = None
    model_version: str = None
    model_type: str = None
    threshold: Union[float, None] = None
    _model = None

    def __post_init__(self):
        self.id = hashlib.md5((self.model_name + self.model_version).encode()).hexdigest()



@dataclass
class Prediction:

    id: str = uuid.uuid4().hex
    inputs: dict = field(default_factory=dict)
    label: Union[str, int, float, None] = None
    probabilities: Union[float, None] = None
    actual: Union[str, int, float, None] = None
    threshold: Union[float, None] = None

    model: Model = None

    _input_key: str = None

    def get_pandas_frame_of_inputs(self):
        return pd.DataFrame([self.inputs], index=[0])

    def __repr__(self):
        return f'Prediction ID: {self.id}'