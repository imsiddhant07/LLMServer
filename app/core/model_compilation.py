"""
Module holds logic for model compilation.
"""

import ctypes
from llama_cpp import Llama

class CompiledModel:
    def __init__(self):
        self.model = Llama(model_path="path/to/vicuna-7b.gguf", n_ctx=2048, n_threads=4)

    def generate(self, text, max_tokens=100):
        return self.model(text, max_tokens=max_tokens)['choices'][0]['text']

COMPILATION_MODEL_MAP = {
    'llama.cpp': CompiledModel()
}

def load_model():
    """
    """
    compilation = 'llama.cpp'
    compiled_model = COMPILATION_MODEL_MAP.get(compilation)
    return compiled_model
