import os
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def set_seed(seed: int = 42):
    import random
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
    def make_gen(subdir, shuffle):
        return datagen.flow_from_directory(
            directory=base_dir / subdir,
            target_size=img_size,
            batch_size=batch_size,
            seed=seed,
            class_mode="binary",
            shuffle=shuffle
        )
    return (
        make_gen("train", True),
        make_gen("valid", False),
        make_gen("test", False),
    )
