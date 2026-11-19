import shutil
from pathlib import Path

import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def _rescale(self, image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    def _build_augmentation(self):
        # Approx. of legacy ImageDataGenerator: flip, rotate ~40°, shift/zoom 0.2
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(40 / 360),
                tf.keras.layers.RandomTranslation(0.2, 0.2),
                tf.keras.layers.RandomZoom(0.2),
            ],
            name="data_augmentation",
        )

    def train_valid_generator(self):
        image_size = tuple(self.config.params_image_size[:-1])
        batch_size = self.config.params_batch_size
        seed = 42

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.config.training_data),
            validation_split=0.20,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            interpolation="bilinear",
        )

        self.valid_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.config.training_data),
            validation_split=0.20,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            interpolation="bilinear",
            shuffle=False,
        )

        self.train_ds = self.train_ds.map(self._rescale, num_parallel_calls=tf.data.AUTOTUNE)
        self.valid_ds = self.valid_ds.map(self._rescale, num_parallel_calls=tf.data.AUTOTUNE)

        if self.config.params_is_augmentation:
            augmentation = self._build_augmentation()

            def augment(image, label):
                return augmentation(image, training=True), label

            self.train_ds = self.train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

        self.train_ds = self.train_ds.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
        self.valid_ds = self.valid_ds.prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

    def copy_model_to_deployment(self):
        """Copy latest trained model into the deployment `model/` folder."""
        source = Path(self.config.trained_model_path)
        destination = Path(self.config.deployment_model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        logger.info(f"Copied trained model from {source} to {destination}")

    def train(self):
        self.model.fit(
            self.train_ds,
            epochs=self.config.params_epochs,
            validation_data=self.valid_ds,
            shuffle=False,
        )
        self.save_model(path=self.config.trained_model_path, model=self.model)
        self.copy_model_to_deployment()
