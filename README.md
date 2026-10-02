# Berkelium Studio — Backend API

The FastAPI backend powering Berkelium Studio. Handles AI chat, image OCR, and PDF text extraction.

---

## Endpoints

- **GET /health** — Health check. Returns status of Tesseract and config. Add `?deep=1` to also ping Fireworks (slower, use before demos to confirm the Fireworks key works).
- **POST /chat** — Send a message to the AI model via Fireworks AI. Requires `X-API-Key` header.
- **POST /ocr** — Extract text from an image using Tesseract OCR. Requires `X-API-Key` header.
- **POST /pdf-ocr** — Extract text from a PDF file. Requires `X-API-Key` header.

---

## Tech Stack

- Python 3.10+
- FastAPI
- httpx (async HTTP calls to Fireworks AI)
- pytesseract + Pillow (image OCR)
- pypdf (PDF text extraction)
- slowapi (rate limiting)
- Uvicorn (ASGI server)

---

## Limits

- Chat message: max 4000 characters
- Uploads (OCR, PDF): max 10MB per file
- Rate limit: 30 requests per minute per IP on all endpoints
- Fireworks calls retry up to 3 times with backoff

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/sudeep03042008/berkelium-studio
cd berkelium-studio
```

### 2. Create config.json in the project root

```json
{
  "fireworks_api_key": "YOUR_API_KEY_HERE",
  "model_id": "accounts/YOUR_USERNAME/models/YOUR_MODEL_ID",
  "base_url": "https://api.fireworks.ai/inference/v1",
  "backend_api_key": "CHANGE_THIS_TO_A_LONG_RANDOM_STRING",
  "allowed_origins": ["http://localhost:5173"],
  "app_name": "Berkelium Studio",
  "version": "1.0.0"
}
```

> config.json is never committed to git. Keep your API keys private.

### 3. Install Tesseract OCR

The `/ocr` endpoint needs the Tesseract binary installed on your system (pytesseract is just a Python wrapper around it).

- **Mac**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

Verify it works with: `tesseract --version`

### 4. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 5. Run the backend

```bash
uvicorn main:app --reload
```

Backend runs at: http://127.0.0.1:8000

---

## Frontend Integration

Every request to /chat, /ocr, and /pdf-ocr must include the backend API key as a header:

Without this header, requests return 401 Unauthorized.

---

## Logs

Backend logs are written to `backend/backend.log` and also printed to the terminal.

---

## Team Berkelium — AMD Developer Hackathon ACT III

Built for the lablab.ai AMD hackathon.