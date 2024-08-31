"""
Module holds class for inference engine.
"""
from app.core.model_compilation import load_model
from app.core.medusa_head import MedusaHeadForTransformers


class InferenceEngine:
    def __init__(self, use_medusa: bool = True):
        base_model = load_model()
        self.model = MedusaHeadForTransformers(base_model) if use_medusa else base_model
    
    def generate(self, text: str, max_tokens: int = 100):
        print('text in inference engine generate', text)
        model_generation = self.model.generate(text, max_tokens=max_tokens)
        decoded = self.model.tokenizer.decode(model_generation.cpu().detach().tolist())
        print('decoded in inference engine generate', decoded)
        return decoded
