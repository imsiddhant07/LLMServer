# Placeholder for README.md


To convert the model to gguf format.
 ```
 git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

python examples/convert_legacy_llama.py <path/to/vicuna-7b-v1.3> --outfile path/to/vicuna-7b-v1.318_0.gguf --outtype q8_0

python examples/convert_legacy_llama.py <path/to/vicuna-7b-v1.3> --outfile path/to/vicuna-7b-v1.3.gguf
 ```


 To run the app.
```
cd LLMServer
// make sure the model_path in app.core.settings is updated to the GGUF file.

uvicorn main:app --host 0.0.0.0 --port 8000
```

