import fastapi
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test(input: str):
    return {"message": f"get the input{input}"}

@app.get("/stream")
async def stream():
    async def event_generator():
        for i in range(5):
            yield f"data: Message {i}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
    