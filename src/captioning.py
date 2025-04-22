import tensorflow as tf
from .models import BlipCaptionLayer
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption(image_path: str):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    layer = BlipCaptionLayer(processor, blip_model)
    img_tensor = tf.constant(str(image_path))
    caption = layer(img_tensor).numpy().decode("utf-8")
    return caption
