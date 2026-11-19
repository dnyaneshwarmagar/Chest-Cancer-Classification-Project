import os
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.tensorflow
import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        image_size = tuple(self.config.params_image_size[:-1])

        self.valid_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.config.training_data),
            validation_split=0.30,
            subset="validation",
            seed=42,
            image_size=image_size,
            batch_size=self.config.params_batch_size,
            label_mode="categorical",
            interpolation="bilinear",
            shuffle=False,
        )
        self.valid_ds = self.valid_ds.map(
            lambda image, label: (tf.cast(image, tf.float32) / 255.0, label),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_ds)
        self.save_score()

    def save_score(self):
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)

    @staticmethod
    def _should_register_model(tracking_uri: str) -> bool:
        register_flag = os.getenv("MLFLOW_REGISTER_MODEL", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # DagsHub tracking does not support Model Registry create API.
        is_dagshub = "dagshub.com" in tracking_uri.lower()
        return register_flag and not is_dagshub

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_uri = mlflow.get_tracking_uri()
        tracking_url_type_store = urlparse(tracking_uri).scheme
        logger.info(f"MLflow tracking URI: {tracking_uri} (scheme={tracking_url_type_store})")

        register_model = self._should_register_model(tracking_uri)
        registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "VGG16Model")

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
            )

            if register_model:
                mlflow.tensorflow.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name=registered_model_name,
                )
                logger.info(f"Model logged and registered as '{registered_model_name}'")
            else:
                # Artifact-only path (safe for DagsHub).
                mlflow.tensorflow.log_model(self.model, artifact_path="model")
                logger.info("Model logged to MLflow as a run artifact (not registered)")
