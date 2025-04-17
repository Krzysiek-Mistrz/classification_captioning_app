import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def download_and_extract(url: str, target: Path):
    if not target.exists():
        archive = target.name + ".tar"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r") as tar_ref:
            tar_ref.extractall(path=target.parent)
        os.remove(archive)

def build_data_generators(base_dir: Path, img_size=(224, 224), batch_size=32, seed=42):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_gen = datagen.flow_from_directory(
        directory=base_dir / "train",
        target_size=img_size,
        batch_size=batch_size,
        seed=seed,
        class_mode="binary",
        shuffle=True
    )
    valid_gen = datagen.flow_from_directory(
        directory=base_dir / "valid",
        target_size=img_size,
        batch_size=batch_size,
        seed=seed,
        class_mode="binary",
        shuffle=False
    )
    test_gen = datagen.flow_from_directory(
        directory=base_dir / "test",
        target_size=img_size,
        batch_size=batch_size,
        seed=seed,
        class_mode="binary",
        shuffle=False
    )
    return train_gen, valid_gen, test_gen

def build_classifier(input_shape=(224, 224, 3)):
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

def plot_history(history: tf.keras.callbacks.History):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    plt.figure()
    plt.plot(epochs, hist["loss"], label="train loss")
    plt.plot(epochs, hist["val_loss"], label="val loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, hist["accuracy"], label="train acc")
    plt.plot(epochs, hist["val_accuracy"], label="val acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def test_single_image(generator, model, idx=0):
    imgs, labels = next(generator)
    preds = (model.predict(imgs) > 0.5).astype(int).flatten()
    class_names = {v: k for k, v in generator.class_indices.items()}
    img = imgs[idx]
    true = class_names[int(labels[idx])]
    pred = class_names[int(preds[idx])]
    plt.imshow(img)
    plt.title(f"True: {true} | Pred: {pred}")
    plt.axis("off")
    plt.show()

class BlipCaptionLayer(tf.keras.layers.Layer):
    def __init__(self, processor, blip_model, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.blip_model = blip_model

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

def main():
    #data prep
    set_seed(42)
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1"
    base = Path("aircraft_damage_dataset_v1")
    download_and_extract(url, base)
    train_gen, valid_gen, test_gen = build_data_generators(
        base_dir=base,
        img_size=(224, 224),
        batch_size=32,
        seed=42
    )
    #classification model
    model = build_classifier(input_shape=(224, 224, 3))
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(
        train_gen,
        epochs=5,
        validation_data=valid_gen
    )
    plot_history(history)
    loss, acc = model.evaluate(test_gen)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    test_single_image(test_gen, model, idx=1)
    #transformer
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_layer = BlipCaptionLayer(processor, blip_model)
    img_path = tf.constant(str(base / "test" / "dent" / "144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg"))
    caption = blip_layer(img_path).numpy().decode("utf-8")
    print("Caption:", caption)

if __name__ == "__main__":
    main()
