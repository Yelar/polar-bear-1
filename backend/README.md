# Polar Bear Backend

This FastAPI service performs the Tokenc compression in Python and is called by the macOS app.

## Setup
- Create `.env` in this folder with your API key.
- Install dependencies and run the server.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Environment variables
- `TOKENC_API_KEY` (required if you want TokenC routing)
- `TOKENC_ENDPOINT` (optional, defaults to `https://api.tokenc.com/v1/compress`)

If you install the `tokenc` library, the backend will use it directly. Otherwise it falls back to an HTTP call.

## API

### `GET /health`

Simple health check.

### `POST /compress`

The macOS app calls this endpoint and passes:
- provider selection (TokenC vs local)
- compression mode (local: LLMLingua vs Hybrid)
- aggressiveness (0–1)

The response includes token stats:
- `output`
- `original_tokens`, `compressed_tokens`, `tokens_saved`, `reduction_ratio`

Example:

```bash
curl -s http://127.0.0.1:8000/compress \
  -H "Content-Type: application/json" \
  -d '{"text":"hello world","provider":"local","mode":"ml","aggressiveness":0.5}' | python -m json.tool
```
