# schema/housing_input.py

from pydantic import BaseModel, Field, model_validator
from typing import List

class HousingInput(BaseModel):
    features: List[List[float]] = Field(..., description="List of samples, each with 8 float features")

    @model_validator(mode='before')
    def check_feature_shape(cls, values):
        features = values.get('features')
        if not all(len(feature) == 8 for feature in features):
            raise ValueError("Each feature list must contain exactly 8 float values.")
        return values
