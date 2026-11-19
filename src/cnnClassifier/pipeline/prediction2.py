import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
import keras

from cnnClassifier import logger

HF_REPO_ID = "AreejAlotaibi12/cancer_image_classifier"
HF_MODEL_DIR = Path("hf_model")
_hf_model = None


def _model_files_present(model_dir: Path) -> bool:
    """SavedModel needs a variables/ directory (and usually keras_metadata.pb)."""
    return (model_dir / "variables").exists() and (
        (model_dir / "saved_model.pb").exists() or (model_dir / "keras_metadata.pb").exists()
    )


def ensure_hf_model_dir(model_dir: Path = HF_MODEL_DIR) -> Path:
    """
    Download the HF model once into ./hf_model and reuse that folder afterwards.
    """
    model_dir = Path(model_dir)
    if _model_files_present(model_dir):
        logger.info(f"Using local HF model from {model_dir.resolve()}")
        return model_dir

    logger.info(f"Downloading HF model {HF_REPO_ID} into {model_dir.resolve()} ...")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_REPO_ID, local_dir=str(model_dir))
    if not _model_files_present(model_dir):
        raise FileNotFoundError(
            f"HF model download finished but required files are missing in {model_dir}"
        )
    logger.info(f"HF model saved to {model_dir.resolve()}")
    return model_dir


def _get_hf_model():
    """Load TFSMLayer once from local hf_model/ folder."""
    global _hf_model
    if _hf_model is None:
        local_dir = ensure_hf_model_dir()
        _hf_model = keras.layers.TFSMLayer(str(local_dir), call_endpoint="serving_default")
        logger.info(f"HF model layer ready from {local_dir.resolve()}")
    return _hf_model


class PredictionPipeline2:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = _get_hf_model()

        img = Image.open(self.filename).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        raw_predictions = model(img_array)

        probability = None
        if isinstance(raw_predictions, dict):
            for _, val in raw_predictions.items():
                probability = float(np.array(val)[0][0])
                break
        else:
            probability = float(np.array(raw_predictions)[0][0])

        binary_outcome = 1 if probability >= 0.5 else 0
        logger.info(f"HF predict2 probability={probability:.4f} binary={binary_outcome}")

        # 0 -> Normal, 1 -> Adenocarcinoma Cancer
        if binary_outcome == 0:
            return [{"image": "Normal"}]
        return [{"image": "Adenocarcinoma Cancer"}]
