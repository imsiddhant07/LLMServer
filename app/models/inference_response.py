"""
Module to hold Pydantic model for inference response body.
"""
from pydantic import BaseModel

class InferenceResponse(BaseModel):
    result: str
