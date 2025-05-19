import streamlit as st
st.set_page_config(layout="wide")
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
[data-testid="stSidebar"]{display:none!important;}
</style>""", unsafe_allow_html=True)

import os, io, requests, pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import faiss
from concurrent.futures import ThreadPoolExecutor
import streamlit_authenticator as stauth

import db_utils
from embedding_utils import get_efficientnet_embedding, compute_md5

WC_URL    = "http://falconshop.uz"
WC_KEY    = "ck_388ecbd01de0652b609efc141133e9ebf413a5ca"
WC_SECRET = "cs_96b9ce33f3cf1bd3d1cbbdd51dabbe84780ee868"

hashed = stauth.Hasher(["12345"]).generate()
auth = stauth.Authenticate(
    {"usernames": {"admin": {"name": "Admin", "password": hashed[0]}}},
    "cookie", "admin_key", 30
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

@st.cache_resource(show_spinner=False)
def build_extractor():
    base = EfficientNetB4(weights="imagenet", include_top=False)
    return Model(base.input, GlobalAveragePooling2D()(base.output))
extractor = build_extractor()

def get_emb(pil):
    return get_efficientnet_embedding(pil, extractor)

def fetch_products(page=1, per_page=100):
    r = requests.get(f"{WC_URL}/wp-json/wc/v3/products",
                     params={"page": page, "per_page": per_page, "status": "publish"},
                     auth=(WC_KEY, WC_SECRET), timeout=20)
    r.raise_for_status()
    return r.json()

def purge_missing():
    removed = 0
    for img_id, fpath, *_ in db_utils.get_all_images():
        if not os.path.exists(fpath):
            db_utils.delete_image(img_id, remove_file=True)
            removed += 1
    if removed:
        st.info(f"Удалено {removed} битых записей")

def build_faiss_index():
    vectors = []
    id_map = []
    updated = 0
    for row in db_utils.get_all_images():
        img_id, fname, link, title, pid, *_ = row
        if not os.path.exists(fname):
            continue
        try:
            pil = Image.open(fname).convert("RGB")
            emb = get_emb(pil)
            db_utils.update_image_embedding(img_id, emb)
            vectors.append(emb.astype("float32"))
            id_map.append((img_id, fname, link, title, pid))
            updated += 1
        except Exception as e:
            st.warning(f"Ошибка {fname}: {e}")

    if vectors:
        dim = len(vectors[0])
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
        faiss.normalize_L2(np.array(vectors))
        index.add(np.array(vectors))

        faiss.write_index(index, os.path.join(DATA_FOLDER, "efficientnet_index.faiss"))
        with open(os.path.join(DATA_FOLDER, "efficientnet_meta.pkl"), "wb") as f:
            pickle.dump(id_map, f)

        st.success(f"FAISS-индекс создан: {len(vectors)} объектов")
    else:
        st.warning("Не найдено эмбеддингов для индексации")
    st.success(f"Пересчитано эмбеддингов: {updated}")
    st.rerun()

def sync_woo():
    with st.spinner("Синхронизация WooCommerce…"):
        added = updated = skipped = 0
        page = 1
        all_tasks = []

        def process_product_image(pid, sku, link, title, idx, img):
            try:
                img_url = img["src"]
                is_primary = 1 if idx == 0 else 0
                rec_id = db_utils.exists_photo(pid, idx)
                resp = requests.get(img_url, timeout=15)
                if "image" not in resp.headers.get("Content-Type", "") or len(resp.content) < 1024:
                    return None
                pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
                fname = f"woo_{pid}_{idx}.jpg"
                fpath = os.path.join(DATA_FOLDER, fname)

                new_hash = compute_md5(pil)
                if os.path.exists(fpath):
                    try:
                        existing = Image.open(fpath)
                        if compute_md5(existing) == new_hash:
                            return None
                    except:
                        pass

                pil.save(fpath)
                emb = get_emb(pil)

                return (rec_id, fpath, emb, link, title, sku, pid, idx, is_primary)
            except:
                return None

        try:
            while True:
                prods = fetch_products(page)
                if not prods:
                    break
                for p in prods:
                    pid, sku, link, title = p["id"], p["sku"] or "", p["permalink"], p["name"]
                    if not p["images"]:
                        continue
                    for idx, img in enumerate(p["images"]):
                        all_tasks.append((pid, sku, link, title, idx, img))
                page += 1

            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(lambda args: process_product_image(*args), all_tasks))

            for res in results:
                if not res:
                    skipped += 1
                    continue
                rec_id, fpath, emb, link, title, sku, pid, idx, is_primary = res
                if rec_id:
                    db_utils.update_photo(rec_id, link, title, sku, is_primary, emb_np=emb, filename=fpath)
                    updated += 1
                else:
                    db_utils.insert_image(fpath, link, emb, title, pid, sku, is_primary, idx)
                    added += 1

            build_faiss_index()

        except Exception as e:
            st.error(f"Ошибка синхронизации: {e}")
            return

        st.success(f"Готово: новых {added}, обновлено {updated}, пропущено {skipped}.")
        st.rerun()

def main():
    name, ok, _ = auth.login("Вход", "main")
    if ok is False: st.error("Неверные данные"); return
    if ok is None: st.warning("Введите логин/пароль"); return

    auth.logout("Выйти", "sidebar")
    st.sidebar.success(f"Здравствуйте, {name}")
    st.title("Admin панель")

    purge_missing()

    if st.button("🔄 Синхронизировать с WooCommerce"):
        sync_woo()

    if st.button("🔁 Перегенерировать эмбеддинги и создать FAISS‑индекс"):
        build_faiss_index()

    st.markdown("### 📥 Ручное добавление товара")
    st.markdown("Загрузите одно или несколько изображений. Первое изображение станет основным для товара.")

    uploaded_files = st.file_uploader(
        "Изображения (JPG / PNG)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="upload_new"
    )

    new_link  = st.text_input("Ссылка на товар", key="Ссылка на товар")
    new_title = st.text_input("Название товара", key="Название товара")
    new_sku   = st.text_input("SKU товара", key="SKU товара")

    if uploaded_files and st.button("✅ Добавить изображения"):
        try:
            # Генерация нового product_id
            existing_ids = [r[4] for r in db_utils.get_all_images() if r[4] is not None]
            new_product_id = max(existing_ids, default=100000) + 1

            for i, uploaded_file in enumerate(uploaded_files):
                pil = Image.open(uploaded_file).convert("RGB")
                fname = f"manual_{new_sku or new_title.replace(' ', '_')}_{np.random.randint(10000)}.jpg"
                fpath = os.path.join(DATA_FOLDER, fname)
                pil.save(fpath)

                emb = get_emb(pil)

                db_utils.insert_image(
                    path=fpath,
                    link=new_link,
                    emb_np=emb,
                    title=new_title,
                    product_id=new_product_id,
                    product_sku=new_sku,
                    is_primary=1 if i == 0 else 0,
                    photo_idx=i
                )

            st.success("Изображения добавлены")
            st.rerun()

            st.session_state["upload_new"] = None
            st.session_state["Ссылка на товар"] = ""
            st.session_state["Название товара"] = ""
            st.session_state["SKU товара"] = ""

            st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {e}")

    all_rows = db_utils.get_all_images()
    st.markdown(f"**Всего изображений:** {len(all_rows)}")
    unique_pids = len(set(row[4] for row in all_rows if row[4]))
    st.markdown(f"**Уникальных товаров:** {unique_pids}")

    mode = st.selectbox("Фильтр по", ["Все", "SKU", "Название", "Ссылка"])
    q = st.text_input("Поиск").strip().lower()

    def filt(r):
        _, _, link, title, _, sku, *_ = r
        if mode == "Все" or not q: return True
        if mode == "SKU":       return q in (sku   or "").lower()
        if mode == "Название":  return q in (title or "").lower()
        if mode == "Ссылка":    return q in (link  or "").lower()
        return True

    filtered = list(filter(filt, all_rows))
    grouped = {}
    for row in filtered:
        pid = row[4]
        grouped.setdefault(pid, []).append(row)

    for idx, (pid, images) in enumerate(grouped.items(), start=1):
        images.sort(key=lambda x: (x[6] != 0, x[6]))  # сортировка по is_primary, затем по photo_idx
        st.markdown(f"---\n### №{idx}")

        col_del = st.columns([1, 5])[0]
        if col_del.button("🗑 Удалить весь товар", key=f"del_product_{pid}"):
            db_utils.delete_by_product_id(pid)
            st.success(f"Удалено всё по product_id {pid}")
            st.rerun()

        for pos, (img_id, fname, link, title, pid, sku, idx_, is_pr, emb) in enumerate(images):
            cols = st.columns([1, 3, 3, 2, 1])
            with cols[0]:
                st.image(fname, width=70)
            with cols[1]:
                new_link = st.text_input("Ссылка", link or "", key=f"L{img_id}")
                new_sku = st.text_input("SKU", sku or "", key=f"S{img_id}")
                if st.button("💾", key=f"upd_link_sku_{img_id}"):
                    db_utils.update_link(img_id, new_link)
                    db_utils.update_sku(img_id, new_sku)
                    st.rerun()
            with cols[2]:
                new_title = st.text_input("Название", title or "", key=f"T{img_id}")
                if st.button("💾 ", key=f"upd_title_{img_id}"):
                    db_utils.update_title(img_id, new_title)
                    st.rerun()
            with cols[3]:
                confirm = f"confirm_{img_id}"
                if st.button("🗑️", key=f"del_{img_id}"):
                    st.session_state[confirm] = True
                if st.session_state.get(confirm):
                    col_y, col_n = st.columns(2)
                    if col_y.button("✅", key=f"yes_{img_id}"):
                        db_utils.delete_image(img_id, remove_file=True)
                        st.session_state.pop(confirm); st.rerun()
                    if col_n.button("❌", key=f"no_{img_id}"):
                        st.session_state.pop(confirm); st.rerun()

if __name__ == "__main__":
    main()