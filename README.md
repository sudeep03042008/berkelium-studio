# Berkelium Studio — Backend API

The FastAPI backend powering Berkelium Studio. Handles AI chat, image OCR, and PDF text extraction.

---

## What This Does

- **/health** — Check if the backend is running
- **/chat** — Send a message to the AI model via Fireworks AI
- **/ocr** — Extract text from an image using Tesseract OCR
- **/pdf-ocr** — Extract text from a PDF file

---

## Tech Stack

- Python 3.10+
- FastAPI
- httpx (async HTTP calls to Fireworks AI)
- pytesseract + Pillow (image OCR)
- PyPDF2 (PDF text extraction)
- Uvicorn (ASGI server)

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
  "app_name": "Berkelium Studio",
  "version": "1.0.0"
}
```

> config.json is never committed to git. Keep your API key private.

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Run the backend

```bash
uvicorn main:app --reload
```

Backend runs at: http://127.0.0.1:8000

---

## Team Berkelium — AMD Developer Hackathon ACT III

Built for the lablab.ai AMD hackathon.