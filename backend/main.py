from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

app = FastAPI(title="Polar Bear Backend")
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

DEFAULT_PROVIDER = os.getenv("COMPRESSION_PROVIDER", "custom").lower()
DEFAULT_MODEL = os.getenv("COMPRESSION_MODEL", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
DEFAULT_USE_EMBEDDINGS = os.getenv("COMPRESSION_USE_EMBEDDINGS", "true").lower() in {"1", "true", "yes"}
DEFAULT_CACHE_DIR = os.getenv("COMPRESSION_CACHE_DIR", ".cache/compressor")
DEFAULT_MAX_EMBED_CHUNKS = int(os.getenv("COMPRESSION_MAX_EMBED_CHUNKS", "120"))
DEFAULT_MODE = os.getenv("COMPRESSION_MODE", "ml")  # ml (LLMLingua) or hybrid


class CompressRequest(BaseModel):
    input: str
    aggressiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    provider: str | None = None
    model: str | None = None
    mode: str | None = None  # "ml" (LLMLingua) or "hybrid"
    use_embeddings: bool | None = None
    target_tokens: int | None = Field(default=None, ge=1)
    target_reduction: float | None = Field(default=None, ge=0.0, le=1.0)
    cache_dir: str | None = None
    max_embed_chunks: int | None = Field(default=None, ge=1)
    device: str | None = None  # "cpu" or "cuda"


class CompressResponse(BaseModel):
    output: str
    original_tokens: int | None = None
    compressed_tokens: int | None = None
    tokens_saved: int | None = None
    reduction_ratio: float | None = None
    method: str | None = None


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/compress", response_model=CompressResponse)
async def compress_text(payload: CompressRequest) -> CompressResponse:
    provider = (payload.provider or DEFAULT_PROVIDER).lower()
    logger.info(f"Compress request: provider={provider}, mode={payload.mode}, aggressiveness={payload.aggressiveness}, input_len={len(payload.input)}")

    if provider in {"custom", "local", "hybrid", "ml", "llmlingua"}:
        try:
            from evals.compressor import compress_text as local_compress  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Local compressor unavailable: {exc}") from exc

        # Determine compression mode: ml (LLMLingua) or hybrid
        mode = payload.mode or DEFAULT_MODE
        if provider in {"ml", "llmlingua"}:
            mode = "ml"
        elif provider == "hybrid":
            mode = "hybrid"

        use_embeddings = DEFAULT_USE_EMBEDDINGS if payload.use_embeddings is None else payload.use_embeddings
        model_name = payload.model or DEFAULT_MODEL
        cache_dir = payload.cache_dir or DEFAULT_CACHE_DIR
        device = payload.device or "cpu"

        output, stats = local_compress(
            payload.input,
            importance_cutoff=payload.aggressiveness,
            mode=mode,
            target_tokens=payload.target_tokens,
            target_reduction=payload.target_reduction,
            use_embeddings=use_embeddings,
            cache_dir=cache_dir,
            model_name=model_name,
            device=device,
        )
        original_tokens = stats.get("original_tokens", 0)
        compressed_tokens = stats.get("compressed_tokens", 0)
        tokens_saved = max(0, original_tokens - compressed_tokens)
        
        _log_token_savings(original_tokens, compressed_tokens, f"{provider}/{mode}")
        
        return CompressResponse(
            output=output,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            reduction_ratio=stats.get("reduction_ratio"),
            method=stats.get("method"),
        )

    if provider in {"tokenc", "api", "external"}:
        api_key = os.getenv("TOKENC_API_KEY") or os.getenv("TTC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="TOKENC_API_KEY is not set")

        try:
            import tokenc  # type: ignore
        except Exception:
            output = await compress_via_http(api_key, payload.input, payload.aggressiveness)
        else:
            client = tokenc.TokenClient(api_key=api_key)
            result = client.compress_input(input=payload.input, aggressiveness=payload.aggressiveness)
            output = result.output

        original_tokens = 0
        compressed_tokens = 0
        try:
            from evals.compressor import count_tokens  # type: ignore
            original_tokens = count_tokens(payload.input)
            compressed_tokens = count_tokens(output)
            _log_token_savings(original_tokens, compressed_tokens, provider)
        except Exception:
            pass

        tokens_saved = max(0, original_tokens - compressed_tokens)
        reduction_ratio = tokens_saved / original_tokens if original_tokens > 0 else 0.0

        return CompressResponse(
            output=output,
            original_tokens=original_tokens if original_tokens > 0 else None,
            compressed_tokens=compressed_tokens if compressed_tokens > 0 else None,
            tokens_saved=tokens_saved if tokens_saved > 0 else None,
            reduction_ratio=reduction_ratio if reduction_ratio > 0 else None,
            method="tokenc",
        )

    raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


def _log_token_savings(original_tokens: int, compressed_tokens: int, provider: str) -> None:
    if original_tokens <= 0:
        return
    saved = max(0, original_tokens - compressed_tokens)
    pct = (saved / original_tokens) * 100.0
    logger.info(
        "compress provider=%s original_tokens=%s compressed_tokens=%s saved=%s (%.1f%%)",
        provider,
        original_tokens,
        compressed_tokens,
        saved,
        pct,
    )


async def compress_via_http(api_key: str, text: str, aggressiveness: float) -> str:
    import httpx

    endpoint = os.getenv("TOKENC_ENDPOINT", "https://api.tokenc.com/v1/compress")
    payload = {"input": text, "aggressiveness": aggressiveness}
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(endpoint, json=payload, headers=headers)

    if response.status_code < 200 or response.status_code >= 300:
        detail = response.text or "Tokenc API error"
        raise HTTPException(status_code=response.status_code, detail=detail)

    data = response.json()
    output = data.get("output")
    if not isinstance(output, str):
        raise HTTPException(status_code=500, detail="Tokenc API response missing output")

    return output
