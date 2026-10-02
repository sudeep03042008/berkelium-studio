"""
Berkelium Studio Backend API

Handles AI chat via Fireworks AI, image OCR, and PDF text extraction.
Hardened for hackathon: rate limits, size caps, input validation, retries, logging, auth.
Tier A+B: log rotation, config shape check, request IDs, startup ping, upload type check.
"""

import asyncio
import contextvars
import io
import json
import logging
import logging.handlers
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import pytesseract
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Rotating log keeps backend.log under 10MB with 3 archives. Prevents a long
# running demo from quietly filling the disk.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


_log_path = os.path.join(os.path.dirname(__file__), "backend.log")
_file_handler = logging.handlers.RotatingFileHandler(
    _log_path, maxBytes=10 * 1024 * 1024, backupCount=3
)
_stream_handler = logging.StreamHandler(sys.stdout)
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s"
)
for h in (_file_handler, _stream_handler):
    h.setFormatter(_fmt)
    h.addFilter(RequestIdFilter())

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stream_handler])
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

# Shape check catches typos (modelid vs model_id) and wrong types before they
# turn into confusing errors at request time.
_SHAPES = {
    "fireworks_api_key": str,
    "model_id": str,
    "base_url": str,
    "backend_api_key": str,
    "allowed_origins": list,
}
for _key, _expected in _SHAPES.items():
    if _key in config and not isinstance(config[_key], _expected):
        log.error("config.json: '%s' must be %s", _key, _expected.__name__)
        sys.exit(1)
if isinstance(config.get("allowed_origins"), list):
    if not all(isinstance(o, str) for o in config["allowed_origins"]):
        log.error("config.json: 'allowed_origins' must be a list of strings")
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
STARTUP_PING_TIMEOUT = 10.0


# Startup ping to /models is free metadata, no tokens burned. Catches a wrong key
# before the demo starts instead of during it. Doesn't block startup if Fireworks
# is down so devs can keep working on other endpoints.
@asynccontextmanager
async def lifespan(app: FastAPI):
    if FIREWORKS_API_KEY and MODEL_ID:
        try:
            async with httpx.AsyncClient(timeout=STARTUP_PING_TIMEOUT) as client:
                r = await client.get(
                    f"{BASE_URL}/models",
                    headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
                )
            if r.status_code == 200:
                log.info("Fireworks reachable, API key accepted.")
            elif r.status_code == 401:
                log.error("Fireworks rejected API key. /chat will fail.")
            else:
                log.warning("Fireworks /models returned status %s.", r.status_code)
        except Exception as e:
            log.warning("Could not reach Fireworks at startup: %s", e)
    else:
        log.info("Skipping Fireworks startup ping (key or model_id empty).")
    yield


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Berkelium Studio API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Honor a client-provided X-Request-ID so the frontend can correlate its own
# logs with ours, otherwise mint a short one. Every log line in this request
# carries it via the contextvar.
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        request_id_var.reset(token)


# Reject oversized requests before any handler runs. The chunked read in
# _read_capped still catches streams that lie about Content-Length.
@app.middleware("http")
async def body_size_cap(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        log.warning("Rejected oversized request: %s bytes", cl)
        return JSONResponse(status_code=413, content={"detail": "file too large"})
    return await call_next(request)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not BACKEND_API_KEY:
        return
    if x_api_key != BACKEND_API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing api key")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)

    @field_validator("message")
    @classmethod
    def strip_and_recheck(cls, v: str) -> str:
        # All-whitespace passes min_length=1 but wastes tokens; strip and re-check.
        stripped = v.strip()
        if not stripped:
            raise ValueError("message cannot be whitespace only")
        return stripped


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
    _require_content_type(file, prefix="image/")
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
    _require_content_type(file, exact="application/pdf")
    contents = await _read_capped(file)
    try:
        text = await asyncio.to_thread(_extract_pdf_text, contents)
    except Exception as e:
        log.exception("PDF extract failed: %s", e)
        raise HTTPException(status_code=400, detail="could not process pdf")
    return {"text": text}


def _require_content_type(
    file: UploadFile,
    prefix: Optional[str] = None,
    exact: Optional[str] = None,
) -> None:
    ct = (file.content_type or "").lower()
    if prefix and ct.startswith(prefix):
        return
    if exact and ct == exact:
        return
    log.warning("Rejected upload with content-type %r", ct)
    raise HTTPException(status_code=415, detail="unsupported file type")


async def _read_capped(file: UploadFile) -> bytes:
    # Generic error to the client; exact bytes stay in the log where they help us.
    buf = bytearray()
    while chunk := await file.read(64 * 1024):
        buf.extend(chunk)
        if len(buf) > MAX_UPLOAD_BYTES:
            log.warning("Upload exceeded cap: %s bytes read", len(buf))
            raise HTTPException(status_code=413, detail="file too large")
    return bytes(buf)


def _run_image_ocr(contents: bytes) -> str:
    image = Image.open(io.BytesIO(contents))
    return pytesseract.image_to_string(image)


def _extract_pdf_text(contents: bytes) -> str:
    reader = PdfReader(io.BytesIO(contents))
    return "".join((page.extract_text() or "") for page in reader.pages)