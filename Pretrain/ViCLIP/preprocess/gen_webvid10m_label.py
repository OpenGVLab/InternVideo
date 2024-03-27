import json
import os
from multiprocessing import Pool

import pandas
import tqdm

from utils import get_video_duration

data_dir = os.path.join(os.environ["SL_DATA_DIR"], "videos_images/webvid_10m_2fps_224")
downloaded_vidlist = data_dir.replace("webvid_10m_2fps_224", "webvid_10m_vidlist.txt")

def gen_valid_vidlist():
    """generate the valid video list.
    Returns: set. The valid 

    """
    with open(downloaded_vidlist, 'r') as f:
        videos = f.read().splitlines()
    return set(videos)


def gen_labels(src_file, dst_file):
    """TODO: Docstring for gen_labels.

    Args:
        src_file (str): The original csv file
        dst_file (str): the output json file
        data_dir (str): The data to store the videos.

    """
    df = pandas.read_csv(src_file)
    vids = df["videoid"].values.tolist()
    captions = df["name"].values.tolist()

    valid_videos = gen_valid_vidlist()

    labels = []
    num_invalid = 0
    for vid, caption in tqdm.tqdm(zip(vids, captions), total=len(vids)):
        vid_name = f"{vid}.mp4"
        if vid_name in valid_videos:
            example = {"video": vid_name, "caption": caption, "duration": 0}
            labels.append(example)
        else:
            num_invalid += 1

    # pool = Pool(128)
    # labels = []
    # for example in tqdm.tqdm(pool.imap_unordered(gen_one_example, zip(vids,captions)), total=len(vids)):
    #     labels.append(example)
    print(f"number of valid videos: {len(labels)}. invalid: {num_invalid}")
    with open(dst_file, "w") as f:
        json.dump(labels, f)


def webvid10m(subset):
    print(f"generate labels for subset: {subset}")
    assert subset in ["train", "val"]
    src_file = f"/data/shared/datasets/webvid-10M/raw_data/results_10M_{subset}.csv"
    dst_file = os.path.join(
        os.environ["SL_DATA_DIR"], "anno_pretrain", f"webvid_10m_{subset}.json"
    )
    gen_labels(src_file, dst_file)


if __name__ == "__main__":
    webvid10m("val")
    webvid10m("train")
