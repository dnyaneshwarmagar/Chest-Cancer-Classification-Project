# Chest CT Scan Classification (Deep Learning)

End-to-end CNN pipeline that classifies chest CT scan images as **Normal** or **Adenocarcinoma Cancer** using transfer learning (VGG16) with TensorFlow / Keras 3.

## Features

- Modular training pipeline (data ingestion → base model → training → evaluation)
- DVC for reproducible experiment stages
- MLflow experiment tracking (metrics / artifacts)
- Flask web UI for image upload and prediction
- Trained model saved as Keras `.keras` and copied for inference under `model/`

## Tech stack

| Area | Tools |
|------|--------|
| Language | Python 3.12 |
| Deep learning | TensorFlow 2.18, Keras 3 |
| Base model | VGG16 (ImageNet weights) |
| Pipeline / config | DVC, YAML (`config/`, `params.yaml`) |
| Tracking | MLflow |
| Web app | Flask, Flask-CORS |

## Project structure

```text
├── app.py                      # Flask inference app
├── main.py                     # Run all training stages sequentially
├── config/config.yaml          # Paths and pipeline config
├── params.yaml                 # Hyperparameters
├── dvc.yaml                    # DVC stage definitions
├── model/model.keras           # Deployment copy of trained model
├── templates/index.html        # Prediction UI
├── src/cnnClassifier/
│   ├── components/             # Data, prepare model, train, evaluate
│   ├── config/                 # Configuration manager
│   ├── pipeline/               # Stage runners + PredictionPipeline
│   └── utils/                  # Shared helpers
└── artifacts/                  # Generated data, models, scores (local)
```

## Classes

Training folder order maps to:

| Index | Label |
|-------|--------|
| `0` | Adenocarcinoma Cancer |
| `1` | Normal |

## Setup

```bash
# Create / activate conda env (Python 3.12)
conda create -n venv2 python=3.12 -y
conda activate venv2

# Install dependencies
pip install -r requirements.txt
```

Optional: copy `.env` values as needed (e.g. MLflow / DagsHub credentials). Do not commit secrets.

## Training

### Option A — DVC (recommended)

```bash
dvc repro
```

Stages: `data_ingestion` → `prepare_base_model` → `training` → `evaluation`.

### Option B — Full script

```bash
python main.py
```

### Outputs

| Artifact | Path |
|----------|------|
| Trained model | `artifacts/training/model.keras` |
| Deployment model | `model/model.keras` |
| Evaluation scores | `scores.json` (via DVC metrics) |

### Key hyperparameters (`params.yaml`)

- Image size: `224 × 224 × 3`
- Batch size: `16`
- Epochs: `2` (increase for better results)
- Learning rate: `0.01`
- Augmentation: enabled
- Classes: `2`

## Inference (trained model)

Ensure `model/model.keras` exists (produced after training).

```bash
# Use the locally trained model in the UI
# In .env: PREDICT_ROUTE=one

python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) (or the `PORT` you set). Upload a CT image to get **Normal** or **Adenocarcinoma Cancer**.

API endpoint for the trained model:

```http
POST /predict
Content-Type: application/json
```

Body includes a base64-encoded image (same format the UI sends).

Health check:

```http
GET /health
```

## Configuration

- Paths: `config/config.yaml`
- Hyperparameters: `params.yaml`
- Predict route switch: `PREDICT_ROUTE=one` → trained model at `/predict`

## Notes

- Training is intended to run locally (or via DVC), not inside a thin inference container.
- Evaluation can log metrics/artifacts to MLflow when tracking URI / credentials are configured.
- Keep large artifacts (`artifacts/`, local caches) out of git; use DVC remotes if you need to share data/models.
