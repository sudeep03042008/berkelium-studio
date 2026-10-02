"""
Berkelium Studio Backend API

Handles AI chat via Fireworks AI, image OCR, and PDF text extraction.
Hardened for hackathon: rate limits, size caps, input validation, retries, logging, auth.
"""

import asyncio
import io
import json
import logging
import os
import sys
from typing import Optional

import httpx
import pytesseract
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from pypdf import PdfReader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "backend.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("berkelium")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    log.error("config.json not found at %s. Create it in the project root.", CONFIG_PATH)
    sys.exit(1)
except json.JSONDecodeError as e:
    log.error("config.json is not valid JSON: %s", e)
    sys.exit(1)

FIREWORKS_API_KEY = config.get("fireworks_api_key", "")
MODEL_ID = config.get("model_id", "")
BASE_URL = config.get("base_url", "https://api.fireworks.ai/inference/v1")
BACKEND_API_KEY = config.get("backend_api_key", "")
ALLOWED_ORIGINS = config.get("allowed_origins", ["http://localhost:5173"])

if not FIREWORKS_API_KEY or not MODEL_ID:
    log.warning("fireworks_api_key or model_id is empty; /chat will fail until you set them.")
if not BACKEND_API_KEY:
    log.warning("backend_api_key is empty; anyone with the URL can call the backend.")

MAX_PROMPT_CHARS = 4000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
FIREWORKS_TIMEOUT_SECONDS = 60.0
FIREWORKS_MAX_RETRIES = 3

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Berkelium Studio API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not BACKEND_API_KEY:
        return
    if x_api_key != BACKEND_API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing api key")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def chat(request: Request, payload: ChatRequest):
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": payload.message}],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    last_error: Optional[str] = None
    for attempt in range(1, FIREWORKS_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=FIREWORKS_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{BASE_URL}/chat/completions", headers=headers, json=body
                )
            if response.status_code == 200:
                return {"response": response.json()["choices"][0]["message"]["content"]}
            if 400 <= response.status_code < 500:
                log.error("Fireworks 4xx on attempt %s: %s", attempt, response.text)
                raise HTTPException(status_code=502, detail="upstream request rejected")
            last_error = f"status {response.status_code}"
            log.warning("Fireworks 5xx on attempt %s: %s", attempt, response.text)
        except httpx.RequestError as e:
            last_error = str(e)
            log.warning("Fireworks network error on attempt %s: %s", attempt, e)
        if attempt < FIREWORKS_MAX_RETRIES:
            await asyncio.sleep(2 ** (attempt - 1))

    log.error("Fireworks failed after %s attempts. Last: %s", FIREWORKS_MAX_RETRIES, last_error)
    raise HTTPException(status_code=502, detail="upstream service unavailable")


@app.post("/ocr", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def ocr(request: Request, file: UploadFile = File(...)):
    contents = await _read_capped(file)
    try:
        text = await asyncio.to_thread(_run_image_ocr, contents)
    except Exception as e:
        log.exception("OCR failed: %s", e)
        raise HTTPException(status_code=400, detail="could not process image")
    return {"text": text}


@app.post("/pdf-ocr", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def pdf_ocr(request: Request, file: UploadFile = File(...)):
    contents = await _read_capped(file)
    try:
        text = await asyncio.to_thread(_extract_pdf_text, contents)
    except Exception as e:
        log.exception("PDF extract failed: %s", e)
        raise HTTPException(status_code=400, detail="could not process pdf")
    return {"text": text}


async def _read_capped(file: UploadFile) -> bytes:
    buf = bytearray()
    while chunk := await file.read(64 * 1024):
        buf.extend(chunk)
        if len(buf) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes")
    return bytes(buf)


def _run_image_ocr(contents: bytes) -> str:
    image = Image.open(io.BytesIO(contents))
    return pytesseract.image_to_string(image)


def _extract_pdf_text(contents: bytes) -> str:
    reader = PdfReader(io.BytesIO(contents))
    return "".join((page.extract_text() or "") for page in reader.pages)