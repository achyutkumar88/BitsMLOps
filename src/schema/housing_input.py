# schema/housing_input.py

from pydantic import BaseModel, Field, model_validator
from typing import List


class HousingInput(BaseModel):
    desc = "List of samples, each with 8 float features"
    features: List[List[float]] = Field(..., description=desc)

    @model_validator(mode='before')
    def check_feature_shape(cls, values):
        features = values.get('features')
        if not all(len(feature) == 8 for feature in features):
            failure_Reason = "Each feature list must contain 8 float values"
            raise ValueError(failure_Reason)
        return values
