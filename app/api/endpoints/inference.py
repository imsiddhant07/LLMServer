"""
File to hold Routes for model inference.
"""
import asyncio
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

from app.core.dynamic_batching import DynamicBatcher
from app.models.inference_request import InferenceRequest
from app.models.inference_response import InferenceResponse
from app.core.inference_engine import InferenceEngine

router = APIRouter()
inference_engine = InferenceEngine()
batcher = DynamicBatcher(inference_engine)


@router.post("/generate")
async def generate(request: InferenceRequest, background_tasks: BackgroundTasks):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await batcher.add_request({
        'input': request.prompt,
        'future': future,
        'use_medusa': request.use_medusa
    })
    
    output = await future
    
    return {"generated_text": output}