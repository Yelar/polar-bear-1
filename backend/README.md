# Polar Bear Backend

This FastAPI service performs the Tokenc compression in Python and is called by the macOS app.

## Setup
- Create `.env` in this folder with your API key.
- Install dependencies and run the server.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Environment variables
- `TOKENC_API_KEY` (required)
- `TOKENC_ENDPOINT` (optional, defaults to `https://api.tokenc.com/v1/compress`)

If you install the `tokenc` library, the backend will use it directly. Otherwise it falls back to an HTTP call.
