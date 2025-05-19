import sqlite3, os, numpy as np

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "my_database.db"
)

def _ensure_column(cur, name: str, definition: str):
    cur.execute("PRAGMA table_info(images)")
    if name not in [c[1] for c in cur.fetchall()]:
        cur.execute(f"ALTER TABLE images ADD COLUMN {name} {definition}")

def _ensure_indexes(cur):
    cur.execute("CREATE INDEX IF NOT EXISTS idx_product_id ON images (product_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_photo_idx  ON images (photo_idx)")

def create_table():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                filename      TEXT,
                product_link  TEXT,
                product_title TEXT,
                product_id    INTEGER,
                product_sku   TEXT,
                photo_idx     INTEGER,
                is_primary    INTEGER DEFAULT 0,
                embedding     BLOB
            )
            """
        )
        _ensure_column(cur, "product_id",  "INTEGER")
        _ensure_column(cur, "product_sku", "TEXT")
        _ensure_column(cur, "photo_idx",   "INTEGER")
        _ensure_column(cur, "is_primary",  "INTEGER DEFAULT 0")
        _ensure_indexes(cur)
        con.commit()

create_table()

def insert_image(
    path: str,
    link: str,
    emb_np: np.ndarray,
    title: str = "",
    product_id: int | None = None,
    product_sku: str = "",
    is_primary: int = 0,
    photo_idx: int = 0,
):
    if emb_np is None:
        raise ValueError("Embedding array is None")
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO images
            (filename, product_link, product_title,
             product_id, product_sku, photo_idx, is_primary, embedding)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                path,
                link,
                title,
                product_id,
                product_sku,
                photo_idx,
                is_primary,
                emb_np.tobytes(),
            ),
        )
        con.commit()

def get_all_images():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, filename, product_link, product_title,
                   product_id, product_sku, photo_idx, is_primary, embedding
            FROM images
            """
        )
        rows = cur.fetchall()

    out = []
    for r in rows:
        *head, emb_blob = r
        emb_np = np.frombuffer(emb_blob, np.float32) if emb_blob else np.array([], dtype=np.float32)
        out.append((*head, emb_np))
    return out

def exists_photo(pid: int, idx: int):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id FROM images WHERE product_id=? AND photo_idx=?", (pid, idx))
        row = cur.fetchone()
        return row[0] if row else None

def update_photo(
    rec_id: int,
    link: str,
    title: str,
    sku: str,
    is_primary: int,
    emb_np: np.ndarray | None = None,
    filename: str | None = None,
):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            """
            UPDATE images
            SET product_link=?, product_title=?, product_sku=?, is_primary=?
            WHERE id=?
            """,
            (link, title, sku, is_primary, rec_id),
        )
        if emb_np is not None:
            cur.execute("UPDATE images SET embedding=? WHERE id=?", (emb_np.tobytes(), rec_id))
        if filename:
            cur.execute("UPDATE images SET filename=? WHERE id=?", (filename, rec_id))
        con.commit()

def update_link(img_id: int, new_link: str):   _upd(img_id, "product_link", new_link)
def update_title(img_id: int, new_title: str): _upd(img_id, "product_title", new_title)
def update_sku(img_id: int,  new_sku: str):    _upd(img_id, "product_sku",  new_sku)
def update_image_embedding(img_id: int, emb_np: np.ndarray):
    _upd(img_id, "embedding", emb_np.tobytes())

def _upd(img_id: int, col: str, val):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(f"UPDATE images SET {col}=? WHERE id=?", (val, img_id))
        con.commit()

def delete_image(img_id: int, remove_file: bool = False):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        if remove_file:
            cur.execute("SELECT filename FROM images WHERE id=?", (img_id,))
            row = cur.fetchone()
            if row:
                fname = row[0]
                if fname and os.path.exists(fname):
                    try:
                        os.remove(fname)
                    except OSError:
                        pass
        cur.execute("DELETE FROM images WHERE id=?", (img_id,))
        con.commit()

def delete_by_product_id(pid: int):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM images WHERE product_id=?", (pid,))
        con.commit()
