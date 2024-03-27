import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

data_dir = f'{VL_DATA_DIR}'
if data_dir is None:
    raise ValueError("please set environment `VL_DATA_DIR` before continue")

data_root = __os.path.join(data_dir, "videos_images")
anno_root_pt = __os.path.join(data_dir, "anno_pretrain")
anno_root_downstream = __os.path.join(data_dir, "anno_downstream")

# ============== pretraining datasets=================
available_corpus = dict(
    # pretraining datasets
    cc3m=[
        f"{anno_root_pt}/cc3m_train.json", 
        "{your_data_root}"
    ],
    cc12m=[
        f"{anno_root_pt}/cc12m_train.json", 
        "{your_data_root}"
    ],
    sbu=[
        f"{anno_root_pt}/sbu.json", 
        "{your_data_root}"
    ],
    vg=[
        f"{anno_root_pt}/vg.json", 
        "{your_data_root}"
    ],
    coco=[
        f"{anno_root_pt}/coco.json", 
        "{your_data_root}"
    ],
    imagenet1k=[
        f"{anno_root_pt}/imagenet1k_train.json", 
        "{your_data_root}"
    ],
    webvid=[
        f"{anno_root_pt}/webvid_train.json", 
        "{your_data_root}",
        "video"
    ],
    webvid_10m=[
        f"{anno_root_pt}/webvid_10m_train.json",
        "{your_data_root}",
        "video",
    ],
    kinetics400=[
        f"{anno_root_pt}/kinetics400_train.json",
        "{your_data_root}",
        "video",
    ],
    kinetics710=[
        f"{anno_root_pt}/kinetics710_train.json",
        "{your_data_root}",
        "video",
    ],
    kinetics710_raw=[
        f"{anno_root_pt}/kinetics710_raw_train.json",
        "{your_data_root}",
        "only_video",
    ],
    internvid_10m_flt=[  
        f"{anno_root_pt}/internvid_10m_flt.json",
        "{your_data_root}",
        "video"
    ],
)

# composed datasets.
available_corpus["coco_vg"] = [available_corpus["coco"], available_corpus["vg"]]
available_corpus["in1k_k710"] = [
    available_corpus["imagenet1k"],
    available_corpus["kinetics710"],
]
available_corpus["webvid_cc3m"] = [available_corpus["webvid"], available_corpus["cc3m"]]
available_corpus["webvid_cc3m_in1k_k710"] = [
    available_corpus["webvid"], 
    available_corpus["cc3m"],
    available_corpus["imagenet1k"],
    available_corpus["kinetics710"],
]
available_corpus["webvid_cc3m_k710raw"] = [
    available_corpus["webvid"], 
    available_corpus["cc3m"],
    available_corpus["kinetics710_raw"],
]
available_corpus["webvid_14m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid12m_14m"] = [
    available_corpus["webvid"],
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_14m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["simple_17m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"],
    available_corpus["cc12m"],
]
available_corpus["simple_25m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["cc12m"],
]
available_corpus["viclip_20m"] = [
    available_corpus["internvid_10m_flt"],
    available_corpus["webvid_10m"],
]
available_corpus["viclip"] = [
    available_corpus["internvid_10m_flt"],
]

# ============== for validation =================
available_corpus["msrvtt_1k_test"] = [
    f"{anno_root_downstream}/msrvtt_test1k.json",
    "{your_data_root}",
    "video",
]
available_corpus["k400_act_val"] = [
    f"{anno_root_downstream}/kinetics400_validate.json",
    "{your_data_root}",
    "video",
]
available_corpus["k600_act_val"] = [
    f"{anno_root_downstream}/kinetics600_validate.json",
    "{your_data_root}",
    "video",
]
available_corpus["k700_act_val"] = [
    f"{anno_root_downstream}/kinetics700_validate.json",
    "{your_data_root}",
    "video",
]
available_corpus["sthsthv1_act_val"] = [
    f"{anno_root_downstream}/sthsthv1_validate_clean2.json",
    "{your_data_root}",
    "video",
]
available_corpus["sthsthv2_act_val"] = [
    f"{anno_root_downstream}/sthsthv2_validate_clean2.json",
    "{your_data_root}",
    "video",
]

