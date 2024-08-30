"""
Module holds class for inference engine.
"""
from app.core.model_compilation import load_model
from app.core.medusa_head import MedusaHead


class InferenceEngine:
    def __init__(self, use_medusa: bool = False):
        base_model = load_model()
        self.model = MedusaHead(base_model) if use_medusa else base_model
    
    def generate(self, text: str, max_tokens: int = 100):
        return self.model.generate(text, max_tokens=max_tokens)
