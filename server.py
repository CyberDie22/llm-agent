import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from agent import run_agent



# TODO: Implement images on the webui



app = FastAPI()

async def stream_generator(messages: list[dict], user_message: dict):
    """
    Streams responses as newline-delimited JSON objects.
    """
    async for chunk in run_agent(messages, user_message):
        yield json.dumps(chunk) + "\n"

@app.post("/api/v1/chat/completions")
async def chat_completion(request: Request):
    # Parse the incoming JSON request
    data = await request.json()

    # Extract messages and the latest user message
    messages = data.get("messages", [])
    user_message = messages[-1] if messages else {}
    messages = messages[:-1]

    # Return a streaming response with appropriate content type for NDJSON
    return StreamingResponse(
        stream_generator(messages, user_message),
        media_type="application/x-ndjson"
    )


def start_server():
    """Start the server on port 8282"""
    uvicorn.run(app, host="0.0.0.0", port=8282)


if __name__ == "__main__":
    start_server()