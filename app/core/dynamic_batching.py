import asyncio

class DynamicBatcher:
    def __init__(self, model, batch_size=4, max_wait_time=0.1):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = asyncio.Queue()
        self.processing = False

    async def add_request(self, request):
        await self.queue.put(request)
        if not self.processing:
            self.processing = True
            asyncio.create_task(self.process_batch())

    async def process_batch(self):
        while not self.queue.empty():
            batch = []
            start_time = asyncio.get_event_loop().time()
            while len(batch) < self.batch_size and (asyncio.get_event_loop().time() - start_time) < self.max_wait_time:
                try:
                    request = await asyncio.wait_for(self.queue.get(), timeout=self.max_wait_time)
                    batch.append(request)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                # Process the batch with the MedusaHead model
                inputs = [req['input'] for req in batch]
                # outputs = [self.model.generate(input, max_tokens=100) for input in inputs]
                outputs = [self.model.generate(text=input) for input in inputs]
                
                # Distribute results
                for req, output in zip(batch, outputs):
                    req['future'].set_result(output)

        self.processing = False
