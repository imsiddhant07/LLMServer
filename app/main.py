from fastapi import FastAPI
from app.api.endpoints import inference

app = FastAPI(
    title='LLMServer',
    description='FastAPI server to serve models for inference.',
    version='0.0.1'
)

app.include_router(inference.router, prefix='/api/v1')

@app.get('/')
def read_route():
    response = dict()
    response['message'] = 'Lets get started! FastAPI server to serve models for inference.'
    return response