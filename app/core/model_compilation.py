"""
Module holds logic for model compilation.
"""

import ctypes
# from llama_cpp import Llama
from app.dependency.llama_cpp.llama import Llama
from app.core.settings import settings

class CompiledModel:
    def __init__(self):
        self.model = Llama(
            model_path=settings.MODEL_PATH,
            n_ctx=settings.N_CTX,
            n_threads=settings.N_THREADS
        )

    def generate(self, text, max_tokens=100):
        return self.model.generate(text, max_tokens=max_tokens)

COMPILATION_MODEL_MAP = {
    'llama.cpp': CompiledModel()
}

def load_model():
    """
    """
    compilation = 'llama.cpp'
    compiled_model = COMPILATION_MODEL_MAP.get(compilation)
    return compiled_model
