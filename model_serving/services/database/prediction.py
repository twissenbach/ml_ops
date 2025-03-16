from sqlalchemy import CLOB, Numeric, ForeignKey, DateTime
from datetime import datetime
import json
from typing import List

from sqlalchemy.orm import Mapped, mapped_column, relationship
from model_serving.services.database.database_client import db
from model_serving.models.prediction import Prediction, Model
from model_serving.domain.common_enums import ModelType
from model_serving.domain.common_enums import Labels

class PredictionSQL(db.Model):
    __tablename__ = 'predictions'

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)
    inputs: Mapped[str] = mapped_column(CLOB, nullable=False)
    value: Mapped[str] = mapped_column(db.String(120), nullable=True)
    probability: Mapped[float] = mapped_column(Numeric, nullable=True)
    actual: Mapped[float] = mapped_column(Numeric, nullable=True)
    updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    created: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    model_id: Mapped[str] = mapped_column(db.String(120), ForeignKey("models.id"), nullable=False)
    model: Mapped["ModelSQL"] = relationship("ModelSQL", back_populates="predictions")

    @classmethod
    def from_prediction(cls, prediction: Prediction, model: 'ModelSQL') -> 'PredictionSQL':
        return cls(
            id=prediction.id
            , inputs=json.dumps(prediction.inputs)
            , value=str(prediction.value.value) if prediction.model.model_type == ModelType.CLASSIFICATION.value else str(prediction.value)
            , probability=prediction.probability
            , actual=float(prediction.actual) if prediction.actual else None
            , model_id=model.id
        )

    def to_prediction(self) -> Prediction:
        return Prediction(
            id=self.id,
            inputs=json.loads(self.inputs),
            value=float(self.value) if self.model.model_type == ModelType.REGRESSION.value else Labels(self.value),
            probability=float(self.probability) if self.probability else None,
            actual=float(self.actual) if self.actual else None,
            model=Model(
                id=self.model_id,
                model_type=ModelType(self.model.model_type),
                model_name=self.model.model_name,
                model_version=self.model.model_version
            )
        )

class ModelSQL(db.Model):
    __tablename__ = 'models'

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)
    model_type: Mapped[str] = mapped_column(db.String(250), nullable=False)
    model_name: Mapped[str] = mapped_column(db.String(240), unique=True, nullable=False)
    model_version: Mapped[str] = mapped_column(db.INTEGER, unique=True, nullable=True)
    updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    created: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    predictions: Mapped[List["PredictionSQL"]] = relationship("PredictionSQL", back_populates="model")

    @classmethod
    def from_model(cls, model: Model) -> 'ModelSQL':

        model_ = db.session.query(ModelSQL).filter(
            (ModelSQL.model_name == model.model_name)
            , (ModelSQL.model_version == model.model_version)
        ).first()

        if not model_:
            model_ = cls(
                id=model.id
                , model_type=model.model_type
                , model_name=model.model_name
                , model_version=model.model_version
            )

            db.session.add(model_)

        return model_

    def __repr__(self):
        return f'Model Name {self.model_name}, Model Version {self.model_version}'
    
class ShapSQL(db.Model):

    __tablename__ = 'shaps'

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)
    type: Mapped[str] = mapped_column(db.String(120), primary_key=True)
    shap_values: Mapped[dict] = mapped_column(CLOB)


    updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    created: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    prediction_id: Mapped['PredictionSQL'] = mapped_column(db.String(120), ForeignKey("predictions.id"), nullable=False)
    