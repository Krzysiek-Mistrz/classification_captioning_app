from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Layer
import tensorflow as tf
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def build_classifier(input_shape=(224,224,3)):
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = False
    x = Flatten()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=outputs)

class BlipCaptionLayer(Layer):
    def __init__(self, processor: BlipProcessor, model: BlipForConditionalGeneration, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.blip_model = model

    def call(self, image_path):
        text = tf.py_function(self._generate, [image_path], tf.string)
        return text

    def _generate(self, image_path):
        path = image_path.numpy().decode("utf-8")
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        output_ids = self.blip_model.generate(**inputs)
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.encode("utf-8")
