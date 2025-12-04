# NFL LM Server

This is a lightweight FastAPI server that exposes a `/generate` endpoint for text generation using a local language model (GPT-2 by default).

## How to run locally

Install dependencies:

pip install -r requirements.txt

Start the server:

uvicorn server:app --reload

This will run on:

http://127.0.0.1:8000

## API Endpoints

### GET /health
Health check endpoint. Returns:

{"status": "ok"}

### POST /generate
Send a JSON body like:

{
  "prompt": "Write a short summary about the NFL:",
  "max_new_tokens": 100,
  "temperature": 0.7
}

Response:

{
  "output": "The NFL is..."
}

## Deploying on Render

Use this start command in Render:

uvicorn server:app --host 0.0.0.0 --port $PORT

Render will install dependencies automatically using:

pip install -r requirements.txt
