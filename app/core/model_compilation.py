"""
Module holds logic for model compilation.
"""

# import ctypes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.core.settings import settings

class CompiledModel:
    def __init__(self):
        model_name = settings.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        self.device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear},
            dtype=torch.qint8
        ).to(self.device)

    def generate(self, text, max_tokens=100):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            output = self.quantized_model.generate(input_ids, max_length=max_tokens)
        
        return output.cpu().detach()
        # return self.tokenizer.decode(output[0], skip_special_tokens=True)

COMPILATION_MODEL_MAP = {
    'llama.cpp': CompiledModel()
}

def load_model():
    """
    """
    compilation = 'llama.cpp'
    compiled_model = COMPILATION_MODEL_MAP.get(compilation)
    return compiled_model
