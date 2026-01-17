from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Polar Bear Backend")


class CompressRequest(BaseModel):
    input: str
    aggressiveness: float = Field(default=0.5, ge=0.0, le=1.0)


class CompressResponse(BaseModel):
    output: str


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/compress", response_model=CompressResponse)
async def compress_text(payload: CompressRequest) -> CompressResponse:
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

    return CompressResponse(output=output)


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
