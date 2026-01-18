# Polar Bear (polar-bear-1)

Polar Bear is a macOS **menu bar app** that compresses your prompt **in-place** (in any focused text field) so you can paste it into ChatGPT / Claude / Claude Code / etc with **lower inference cost** and fewer context window issues.

Press **⌘ + ⌥ + C** to compress the currently focused text field.

## Inspiration
TheTokenCompany’s product inspired this project — it tackles a critical LLM pain point: **inference cost** and **context window limitations**.

## What it does
You type your prompt in any LLM provider (ChatGPT / Claude / Claude Code / etc). Polar Bear:
- reads the focused text field,
- compresses it using our pipeline (TokenC for short prompts + local LLMLingua for longer prompts),
- writes the compressed text back,
- tracks how many tokens you saved over time.

## How we built it
- A macOS menu bar app in Swift (Accessibility API + global hotkey)
- A local FastAPI backend that provides a `/compress` endpoint
- Multiple compression backends:
  - **Microsoft LLMLingua** (local ML)
  - **Hybrid pipeline** (local heuristic + retrieval-style selection)
  - **TokenC API** (for short prompts)

## Challenges we ran into
- LLMLingua models have a **512 token** context limit, so for long prompts we **chunk** inputs and compress chunk-by-chunk.
- Our deterministic/hybrid pipeline shines on **long prompts**, but short prompts are dense and often don’t compress well.
- We implemented **Auto mode**:
  - **short prompts** (< 500 tokens) → TokenC API
  - **long prompts** (≥ 500 tokens) → local LLMLingua (with hybrid fallback)

## Accomplishments that we’re proud of
- Building a local prompt-compression pipeline that’s practical and fast
- Getting near-comparable token saving results (≈50% at ~0.7 compression) on LongBench-style long-context inputs
- Shipping all of this in under 24 hours

## What we learned
- Swift/Xcode macOS menu bar app development
- Prompt/token compression strategies and tradeoffs
- Evaluation workflows for long-context benchmarks

## What’s next
- Improve LongBench benchmark results + quality/robustness
- Better chunk stitching and constraint preservation for extremely long prompts

---

## Project layout

- `polar-bear/` — macOS app (Xcode project)
  - `polar-bear/polar-bear.xcodeproj`
  - `polar-bear/polar-bear/` Swift sources
- `backend/` — FastAPI backend (local compression + TokenC proxy)
  - `backend/main.py` API entrypoint
  - `backend/.env` (user-created) API keys + config
- `evals/` — compression library + LongBench evaluation

---

## macOS app (Xcode) — how to run

1) Open Xcode project:

```bash
open polar-bear/polar-bear.xcodeproj
```

2) In Xcode:
- Select the app target
- Press **⌘R**

3) Grant Accessibility permissions:
- System Settings → Privacy & Security → **Accessibility** → enable Polar Bear

4) Use it:
- Click a text field (ChatGPT / Claude / TextEdit / browser input, etc)
- Press **⌘ + ⌥ + C**

5) Settings:
- Click the menu bar icon → **Settings…**
- In Settings:
  - **Compression slider (0–1)** controls how aggressive compression is
  - Provider mode:
    - **Auto** (recommended)
    - Local (LLMLingua)
    - Local (Hybrid)
    - TokenC

---

## Backend (FastAPI) — how to run

Create and activate a venv:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env` (do not commit secrets). At minimum for TokenC:

```bash
TOKENC_API_KEY=YOUR_TOKENC_KEY
```

Run server:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### `/compress` API
The macOS app always calls your local backend. The backend chooses:
- `provider="tokenc"` → calls TokenC API using `TOKENC_API_KEY`
- `provider="local"` / `mode="ml"` → local LLMLingua compression
- `provider="local"` / `mode="hybrid"` → local hybrid pipeline

The response includes token stats:
- `output`
- `original_tokens`, `compressed_tokens`, `tokens_saved`, `reduction_ratio`

---

## Evals / LongBench — run benchmarks

The evaluator lives in `evals/longbench_eval.py`.

Dry run:

```bash
python3 -m evals.longbench_eval --dry-run --compressor-mode ml
```

Full run:

```bash
python3 -m evals.longbench_eval --n 30 --cutoffs 0.3 0.9 --budget-usd 10 --compressor-mode ml
```

If you want to compare to hybrid:

```bash
python3 -m evals.longbench_eval --n 30 --cutoffs 0.3 0.9 --budget-usd 10 --compressor-mode hybrid
```

---

## Troubleshooting

### “Hotkey doesn’t work”
- Ensure Accessibility permission is enabled for the app
- Re-run the app from Xcode
- Try restarting the app once after granting permissions

### “TokenC doesn’t work for short prompts”
- Ensure `TOKENC_API_KEY` exists in `backend/.env`
- Restart backend after updating `.env`

### “LLMLingua model error: >512 tokens”
- This is handled by chunking in `evals/compressor.py` (long prompts are chunked automatically)
