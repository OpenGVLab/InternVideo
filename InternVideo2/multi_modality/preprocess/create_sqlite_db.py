import json
import os
import sqlite3
import time

import numpy as np


def convert_to_sqlite_db(src_path: str, dst_path: str, media_type: str):
    """TODO: Docstring for convert_to_sqlite_db.

    Args:
        src_path (str): The src json annotation file path.
        dst_path (str): The saved sqlite db path.
        media_type (str): The media type. Either "image" or "video".

    """

    # con = sqlite3.connect("file:"+dst_path+"?mode=ro",uri=True)
    con = sqlite3.connect(dst_path)
    cur = con.cursor()
    print(f"creating table")
    cur.execute("DROP TABLE IF EXISTS annos")
    table_sql = f""" CREATE TABLE IF NOT EXISTS `annos` (
                                        `id` integer PRIMARY KEY,
                                        `{media_type}` text,
                                        `caption` text
                                        )"""
    print(table_sql)
    cur.execute(table_sql)

    with open(src_path, "r") as f:
        anno_list = json.load(f)
    filenames = [anno[media_type] for anno in anno_list]
    captions = [anno["caption"] for anno in anno_list]
    ids = list(range(len(filenames)))
    records = list(zip(ids, filenames, captions))

    cur.executemany(f"INSERT INTO annos (id, {media_type}, caption) VALUES (?,?,?)", records)
    con.commit()
    con.close()


def read_sqlite(db_path):
    """TODO: Docstring for read_sqlite.

    Args:
        db_path (TODO): TODO

    Returns: TODO

    """
    con = sqlite3.connect("file:" + db_path + "?mode=ro", uri=True)
    cur = con.cursor()
    ids = cur.execute("SELECT id FROM annos").fetchall()
    ids = [id[0] for id in ids]
    print("number medias:", len(ids))
    np.random.shuffle(ids)
    for id in ids[:100]:
        t1 = time.time()
        query = f"SELECT * FROM annos WHERE id = {id};"
        res = cur.execute(query)
        newid, filename, caption = res.fetchone()
        t2 = time.time()
        print(f"time: {t2-t1}s", id, newid, filename, caption)
    con.close()


def convert(json_filename, media_type):
    """convert json annotations to sqlite.
    Returns: TODO

    """
    print(f"\n--------converting {filename}----------------")
    # src_path = os.path.join(os.environ["SL_DATA_DIR"], "anno_pretrain", json_filename)
    path = 'your_data_path/anno/anno_pretrain'
    src_path = os.path.join(path, json_filename)
    dst_path = src_path.replace(".json", ".sqlite.db")
    convert_to_sqlite_db(src_path, dst_path, media_type)
    read_sqlite(dst_path)


if __name__ == "__main__":
    filenames = [
        # ["cc12m.json", "image"],
        ["cc3m_train.json", "image"],
        # ["cc3m_val.json", "image"],
        # ["coco.json", "image"],
        # ["sbu.json", "image"],
        # ["vg.json", "image"],
        # ["webvid_10m_train.json", "video"],
        # ["webvid_10m_val.json", "video"],
        ["webvid_train.json", "video"],
    ]
    for filename, media_type in filenames:
        convert(filename, media_type)
