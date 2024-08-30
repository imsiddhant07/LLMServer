"""
Module to hold Pydantic model for inference request body.
"""
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    prompt: str
    use_medusa: bool = False
