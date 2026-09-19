from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os
import pytesseract
from PIL import Image
import PyPDF2
import io

# Load config from root of project
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

FIREWORKS_API_KEY = config.get("fireworks_api_key", "")
MODEL_ID = config.get("model_id", "")
BASE_URL = config.get("base_url", "https://api.fireworks.ai/inference/v1")

app = FastAPI(title="Berkelium Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/chat")
async def chat(payload: dict):
    prompt = payload.get("message", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="message is required")

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=body
        )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()
    return {"response": data["choices"][0]["message"]["content"]}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    text = pytesseract.image_to_string(image)
    return {"text": text}

@app.post("/pdf-ocr")
async def pdf_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    reader = PyPDF2.PdfReader(io.BytesIO(contents))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return {"text": text}