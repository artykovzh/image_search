import numpy as np
import hashlib
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from sklearn.preprocessing import normalize

def resize_with_padding(pil_img, target_size=(380, 380), fill_color=(0, 0, 0)):
    """
    Resize image to target size while maintaining aspect ratio, with padding.
    """
    img = img_to_array(pil_img)
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = np.array(array_to_img(img).resize((nw, nh)))
    padded = np.full((*target_size, 3), fill_color, dtype=np.uint8)
    top = (target_size[0] - nh) // 2
    left = (target_size[1] - nw) // 2
    padded[top:top + nh, left:left + nw] = resized
    return Image.fromarray(padded)

def get_efficientnet_embedding(pil_img, extractor):
    """
    Extract and normalize embedding from image using EfficientNetB4.
    """
    img = resize_with_padding(pil_img.convert("RGB"), target_size=(380, 380))
    arr = preprocess_input(np.expand_dims(np.array(img), 0))
    emb = extractor.predict(arr, verbose=0)[0].astype("float32")
    return normalize(emb.reshape(1, -1))[0]

def compute_md5(pil_img):
    """
    Compute MD5 hash of the image pixel data.
    """
    with pil_img.convert("RGB") as img:
        arr = np.array(img)
        return hashlib.md5(arr.tobytes()).hexdigest()
