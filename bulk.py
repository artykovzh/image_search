import os
import pandas as pd
from collections import defaultdict

DATA_FOLDER = "data/bulk_upload"
CSV_PATH = "data/bulk_import.csv"

def extract_sku(filename):
    return filename.split("_")[0]  # P001_img1.jpg → P001

def generate_csv_from_filenames():
    files = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    if not files:
        print("Нет изображений в папке.")
        return

    sku_groups = defaultdict(list)
    for fname in files:
        sku = extract_sku(fname)
        sku_groups[sku].append(fname)

    rows = []
    product_counter = 100000  # Начальный product_id

    for sku, images in sku_groups.items():
        product_id = product_counter
        title = f"Product {sku}"
        link = f"https://example.com/product/{sku}"

        for idx, fname in enumerate(sorted(images)):
            rows.append({
                "image_filename": fname,
                "product_id": product_id,
                "product_sku": sku,
                "title": title,
                "link": link,
                "is_primary": 1 if idx == 0 else 0,
                "photo_idx": idx
            })

        product_counter += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV создан: {CSV_PATH} (товаров: {len(sku_groups)})")

if __name__ == "__main__":
    generate_csv_from_filenames()
