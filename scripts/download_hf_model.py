"""One-time download of the Hugging Face model into ./hf_model."""
from cnnClassifier.pipeline.prediction2 import ensure_hf_model_dir

if __name__ == "__main__":
    path = ensure_hf_model_dir()
    print(f"HF model ready at: {path.resolve()}")
