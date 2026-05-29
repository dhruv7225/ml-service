"""
QuickSpares CLIP photo-search sidecar.

Single-purpose FastAPI service that wraps OpenAI's CLIP ViT-B/32 model
(via sentence-transformers) to produce 512-d image embeddings.

Endpoints (consumed by app/Services/Ml/HttpMlClient.php):

  GET  /healthz           - simple liveness probe
  GET  /version           - { "model_version": "clip-vitb32-v1" }
  POST /embed             - body: JSON { "image_url": "..." }
                                    OR multipart form-data with `image` file
                            response: { "embedding": [512 floats], "model_version": "..." }
  POST /switch-model      - dev no-op; accepts/echoes a model_version

Run:
  uvicorn main:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import io
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
from typing import Optional

import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("CLIP_MODEL", "clip-ViT-B-32")
MODEL_VERSION = os.getenv("MODEL_VERSION", "clip-vitb32-v1")
EMBEDDING_DIM = 512
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB hard cap
HTTP_FETCH_TIMEOUT = 10  # seconds, for image_url path

# Wikimedia / many CDNs reject Python's default UA. Pretend to be a real client
# but identify the project (Wikimedia's UA policy: include a contact / project name).
HTTP_USER_AGENT = (
    "QuickSpaces-PhotoSearch/1.0 (https://quickspares.in; contact: ops@quickspares.in) "
    "python-requests"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("clip-sidecar")

# ── Model load (eager) ─────────────────────────────────────────────────────────

log.info("Loading CLIP model: %s", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME, device="cpu")
log.info("Model loaded; serving as version=%s", MODEL_VERSION)

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="QuickSpares CLIP sidecar",
    description="Produces 512-d CLIP image embeddings for photo search.",
    version="1.0",
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "model_version": MODEL_VERSION}


@app.get("/version")
def version() -> dict:
    return {"model_version": MODEL_VERSION}


@app.post("/switch-model")
async def switch_model(request: Request) -> dict:
    """
    Single-model dev sidecar. Accepts a model_version, but only the version
    we already have loaded is honoured. Returns the current version either way
    so the PHP side can cache it without raising.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    requested = (body or {}).get("model_version", MODEL_VERSION)
    log.info("/switch-model: requested=%s actual=%s", requested, MODEL_VERSION)
    return {"model_version": MODEL_VERSION, "status": "loaded"}


def _embed_pil(img: Image.Image) -> list[float]:
    """Run the loaded CLIP model on a PIL image and return a 512-d float list."""
    img_rgb = img.convert("RGB")
    vec = model.encode(
        [img_rgb],
        convert_to_tensor=False,
        normalize_embeddings=False,
    )[0]

    floats = [float(x) for x in vec.tolist()]
    if len(floats) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected embedding dimension {len(floats)} (expected {EMBEDDING_DIM}). "
            f"Check CLIP_MODEL='{MODEL_NAME}'."
        )
    return floats


def _open_image_safely(blob: bytes) -> Image.Image:
    if len(blob) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image is too large ({len(blob)} bytes; max {MAX_IMAGE_BYTES}).",
        )
    try:
        img = Image.open(io.BytesIO(blob))
        img.verify()  # cheap validity check
        # `verify()` consumes the file handle, so re-open for actual use.
        return Image.open(io.BytesIO(blob))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.post("/embed")
async def embed(request: Request, image: Optional[UploadFile] = File(None)) -> dict:
    """
    Accept an image either as:
      - multipart upload field `image`  (preferred — robust)
      - JSON body `{ "image_url": "..." }`

    Returns the 512-d embedding + model_version.
    """
    img_bytes: bytes | None = None
    source = "unknown"

    if image is not None:
        img_bytes = await image.read()
        source = f"multipart name={image.filename!r} mime={image.content_type!r}"
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}

        url = (body or {}).get("image_url")
        if not url:
            raise HTTPException(
                status_code=400,
                detail="Provide either an `image` multipart field or `image_url` JSON.",
            )

        try:
            resp = requests.get(
                url,
                timeout=HTTP_FETCH_TIMEOUT,
                stream=True,
                headers={"User-Agent": HTTP_USER_AGENT, "Accept": "image/*,*/*;q=0.8"},
                allow_redirects=True,
            )
            resp.raise_for_status()
            buf = io.BytesIO()
            total = 0
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                buf.write(chunk)
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Remote image exceeds {MAX_IMAGE_BYTES} bytes.",
                    )
            img_bytes = buf.getvalue()
            source = f"url={url}"
        except requests.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not fetch image from URL: {e}",
            )

    img = _open_image_safely(img_bytes)

    try:
        embedding = _embed_pil(img)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Embed failed (%s)", source)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    log.info("/embed ok src=%s dim=%d", source, len(embedding))
    return {"embedding": embedding, "model_version": MODEL_VERSION}
