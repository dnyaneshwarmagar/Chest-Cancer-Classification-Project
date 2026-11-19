import os

import numpy as np
import tensorflow as tf


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = tf.keras.models.load_model(os.path.join("model", "model.keras"))
        imagename = self.filename
        test_image = tf.keras.utils.load_img(imagename, target_size=(224, 224))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        # Folder order from training: adenocarcinoma=0, normal=1
        if result[0] == 1:
            prediction = "Normal"
            return [{"image": prediction}]
        prediction = "Adenocarcinoma Cancer"
        return [{"image": prediction}]
