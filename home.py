import streamlit as st
st.set_page_config(layout="wide")
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
[data-testid="stSidebar"]{display:none!important;}
</style>""", unsafe_allow_html=True)

import os
import tempfile
from PIL import Image, ImageDraw
import numpy as np
import pickle
import faiss
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import normalize
from inference_sdk import InferenceHTTPClient
from embedding_utils import resize_with_padding, compute_md5

ROBOFLOW_API_KEY = "PMD7vOSRaV6qAtUSOV3D"
ROBOFLOW_MODEL_ID = "bro-wg6vn/3"
rf_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER   = os.path.join(CURRENT_DIR, "data")
CACHE_FOLDER  = os.path.join(DATA_FOLDER, "bbox_cache")
os.makedirs(CACHE_FOLDER, exist_ok=True)

INDEX_PATH = os.path.join(DATA_FOLDER, "efficientnet_index.faiss")
META_PATH  = os.path.join(DATA_FOLDER, "efficientnet_meta.pkl")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ------------------------------------------------------------------------------------ #
#                                    Model & Utils                                     #
# ------------------------------------------------------------------------------------ #

@st.cache_resource(show_spinner=False)
def build_extractor():
    """Return EfficientNet-B4 backbone without classification head."""
    base = EfficientNetB4(weights="imagenet", include_top=False)
    return Model(base.input, GlobalAveragePooling2D()(base.output))


extractor = build_extractor()


def get_embedding(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to L2-normalised feature vector."""
    img = resize_with_padding(pil_img.convert("RGB"), target_size=(380, 380))
    arr = preprocess_input(np.expand_dims(np.array(img), 0))
    emb = extractor.predict(arr, verbose=0)[0].astype("float32")
    return normalize(emb.reshape(1, -1))[0]


def get_bbox_roboflow(pil_img: Image.Image):
    """
    Detect objects via Roboflow and return cached bounding boxes
    together with raw prediction JSON.
    """
    img_hash   = compute_md5(pil_img)
    cache_path = os.path.join(CACHE_FOLDER, f"{img_hash}.pkl")

    if os.path.exists(cache_path):                       # cached → load and return
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # otherwise run inference
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        pil_img.save(tmp, format="JPEG")
        tmp.flush()
        result = rf_client.infer(tmp.name, model_id=ROBOFLOW_MODEL_ID)

    preds = result.get("predictions", [])
    boxes = []
    for pred in preds:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1 = int(x - w / 2); y1 = int(y - h / 2)
        x2 = int(x + w / 2); y2 = int(y + h / 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
        boxes.append(((x1, y1, x2, y2), pred))

    with open(cache_path, "wb") as f:                    # save cache
        pickle.dump((boxes, result), f)

    return boxes, result


def load_index_and_meta():
    """Load Faiss index and metadata list if available."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


# ------------------------------------------------------------------------------------ #
#                                       UI                                             #
# ------------------------------------------------------------------------------------ #

def main():
    st.title("🔍 Find Similar Products by Image")

    index, meta = load_index_and_meta()
    if index is None or meta is None:
        st.warning("No data available. Please upload images in the admin panel to build the database.")
        st.stop()

    uploaded = st.file_uploader("Upload an image (JPG / PNG / WebP / BMP)")
    if not uploaded:
        return

    if os.path.splitext(uploaded.name)[1].lower() not in ALLOWED_EXT:
        st.error("Unsupported file format.")
        return

    try:
        pil_img = Image.open(uploaded).convert("RGB")
        use_crop = st.checkbox("✂️ Enable object detection & cropping")
        preview_size = 300

        if use_crop:
            bboxes, _ = get_bbox_roboflow(pil_img)
            if not bboxes:
                st.warning("Object not detected. Please try another image.")
                return

            if len(bboxes) == 1:
                crop_coords, _ = bboxes[0]
            else:
                bbox_options = [f"Object {i + 1}" for i in range(len(bboxes))]
                selected_idx = st.selectbox(
                    "Choose an object in the image:",
                    list(range(len(bboxes))),
                    format_func=lambda i: bbox_options[i]
                )
                crop_coords, _ = bboxes[selected_idx]

            x1, y1, x2, y2 = crop_coords

            # preview with bounding boxes
            img_with_bbox = pil_img.copy()
            draw = ImageDraw.Draw(img_with_bbox)
            for i, (coords, _) in enumerate(bboxes):
                cx1, cy1, cx2, cy2 = coords
                color = "red" if len(bboxes) == 1 or i == selected_idx else "green"
                draw.rectangle([cx1, cy1, cx2, cy2], outline=color, width=3)
                draw.text((cx1 + 3, cy1 + 3), f"{i + 1}", fill=color)
            st.image(img_with_bbox, caption="Original image with bounding boxes", width=preview_size)

            cropped_img = pil_img.crop((x1, y1, x2, y2)).resize((pil_img.width, pil_img.height))
            q_emb = get_embedding(cropped_img).reshape(1, -1)
        else:
            st.image(pil_img, caption="Original image", width=preview_size)
            q_emb = get_embedding(pil_img).reshape(1, -1)

    except Exception as e:
        st.error(f"Error while processing image: {e}")
        return

    # -------------------------------------------------------------------------------- #
    #                                   Search                                         #
    # -------------------------------------------------------------------------------- #

    TOP_K = 6
    faiss.normalize_L2(q_emb)

    index, meta = load_index_and_meta()   # reload to ensure consistency
    if index is None or meta is None:
        st.warning("No data available. Please upload images in the admin panel to build the database.")
        st.stop()

    index.hnsw.efSearch = 32              # HNSW search parameter
    scores, indices = index.search(q_emb.astype("float32"), 100)

    st.subheader("🎯 Similar Products")
    shown_pids = set()
    items = []

    for idx in indices[0]:
        if idx >= len(meta):
            continue
        img_id, path, link, title, pid = meta[idx]

        if pid in shown_pids:
            continue
        shown_pids.add(pid)

        # Prefer the first image (suffix _0) for preview
        if not os.path.basename(path).startswith(f"woo_{pid}_0"):
            for mid, mpath, mlink, mtitle, mpid in meta:
                if mpid == pid and os.path.basename(mpath).startswith(f"woo_{pid}_0"):
                    path, link, title = mpath, mlink, mtitle
                    break

        items.append((path, title, link))

    # Display results two per row
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j >= len(items):
                continue
            path, title, link = items[i + j]
            with cols[j]:
                if not os.path.exists(path):
                    st.warning("File is missing.")
                    continue
                st.image(path, width=250)
                if title:
                    st.write(f"**{title}**")
                if link:
                    st.markdown(f"[Product link]({link})")


if __name__ == "__main__":
    main()
