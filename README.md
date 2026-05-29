# QuickSpares CLIP photo-search sidecar

Tiny FastAPI service that the Laravel app talks to over HTTP for image
embeddings (photo search → "snap a part, find the SKU").

## What it does

Wraps `clip-ViT-B-32` via `sentence-transformers` and exposes:

| Verb | Path | Purpose |
| ---- | ---- | ------- |
| GET  | `/healthz` | Liveness probe |
| GET  | `/version` | Returns `{ model_version }` |
| POST | `/embed`   | Body: JSON `{ image_url }` **or** multipart `image` file. Returns `{ embedding: [512 floats], model_version }` |
| POST | `/switch-model` | Dev no-op (single-model build) |

Consumed by `app/Services/Ml/HttpMlClient.php`.

## First-time setup (one-time)

```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first `pip install` pulls torch + transformers + sentence-transformers;
expect ~1 GB of dependencies. Plan for 5-10 minutes the first time.

## Run

```bash
cd ml-service
source venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8001
```

On first launch, sentence-transformers downloads the CLIP ViT-B/32 weights
(~150 MB) from HuggingFace and caches them in `~/.cache/huggingface/`.
Subsequent restarts are instant.

You should see:

```
[INFO] Loading CLIP model: clip-ViT-B-32
[INFO] Model loaded; serving as version=clip-vitb32-v1
INFO:     Uvicorn running on http://127.0.0.1:8001
```

## Smoke test

```bash
curl http://127.0.0.1:8001/healthz
# { "status": "ok", "model": "clip-ViT-B-32", "model_version": "clip-vitb32-v1" }

curl -X POST -F "image=@/path/to/test.jpg" http://127.0.0.1:8001/embed | jq '.embedding | length'
# 512
```

## Wire it into Laravel

Add to your `.env` and `php artisan config:clear`:

```env
ML_DRIVER=http
ML_SERVICE_HOST=http://127.0.0.1:8001
ML_SERVICE_TIMEOUT=30
```

Then backfill embeddings for the existing catalog (one-time):

```bash
php artisan qs:rebuild-embeddings
```

This walks every `master_skus` row that has a `primary_image_url`, calls
the sidecar to compute its embedding, and persists it as JSON in
`master_skus.image_embedding`. Re-runs are safe and only embed missing rows
unless you pass `--force`.

After that, buyers can use **Catalog → Search by photo**.

## Production note

In prod, run this behind a reverse proxy and bump uvicorn workers:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 2
```

Workers > 1 means each worker keeps its own copy of the model in memory
(~600 MB each). For higher throughput, prefer a single worker with batched
embeddings rather than multiple model copies.
