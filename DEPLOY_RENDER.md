# Deploy on Render (Docker)

Yes — **Docker + Render works** for this app as an inference service (`PREDICT_ROUTE=two` / Hugging Face model).

## What was added
- `Dockerfile` — Python 3.12, prod deps, downloads `hf_model` at build time, runs gunicorn
- `requirements-prod.txt` — inference-only packages (no DVC/MLflow/notebook)
- `.dockerignore` — keeps build context small
- `render.yaml` — Docker web service blueprint
- `app.py` — gunicorn-safe (`clApp` at import), `/health`, training disabled in deploy

## Render steps
1. Push this repo to GitHub (including `Dockerfile` / `requirements-prod.txt` / `render.yaml`).
2. In [Render Dashboard](https://dashboard.render.com): **New → Web Service**.
3. Connect the repo.
4. Render should detect Docker. If not:
   - **Runtime:** Docker
   - **Dockerfile path:** `./Dockerfile`
5. Set env vars (or use `render.yaml`):
   - `PREDICT_ROUTE=two`
   - `KERAS_BACKEND=tensorflow`
6. Choose a plan with enough RAM (**Starter / 512MB+ may OOM** with TensorFlow; prefer **Standard 2GB+** if available).
7. Deploy. Open the Render URL when build finishes.

Build downloads ~82MB HF weights into the image, so the first build is slower. Later deploys reuse layers when possible.

## Local Docker test
```bash
docker build -t chest-ct-classifier .
docker run --rm -p 8080:8080 -e PREDICT_ROUTE=two chest-ct-classifier
```
Then open http://127.0.0.1:8080/

## Notes
- Default UI route is `/predict2` (HF). Local VGG `/predict` needs `model/model.keras` in the image (not baked by default).
- Do not put `.env` secrets into the image; set only needed vars in Render.
- Free/low-memory instances often fail with TensorFlow; upgrade plan if the service is killed / OOM.
