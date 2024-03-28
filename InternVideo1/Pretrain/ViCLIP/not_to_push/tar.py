import os
import subprocess
import json
from p_tqdm import p_imap
from functools import partial
from petrel_client.client import Client
import pdb
import pandas as pd
from tqdm import tqdm
import torch
from decord import VideoReader
import io

# def untar(path):
#     full_path = os.path.join(dir_path, path)
#     cmd = f"tar -xvf {full_path} -C {target_path}"
#     print(full_path)
#     subprocess.call(cmd, shell=True)
#     return full_path

# dir_path = '/mnt/petrelfs/videointern/SBU/sbucaptions'
# target_path = '/mnt/petrelfs/videointern/SBU/images'
# file_list = []
# for path in os.listdir(dir_path):
#     if '.tar' in path:
#         file_list.append(path)

# iterator = p_imap(partial(untar), file_list, position=0, leave=True, ncols=100, dynamic_ncols=False)
    
# for img_name in iterator:
#     pass

# def get(path):
#     full_path  = os.path.join(dir_path, path)
#     video = client.get(full_path)
#     if video is not None:
#         return True, path
#     else:
#         return False, path


# client = Client('~/petreloss.conf', enable_mc=True)
# dir_path = "s3://video_pub_new/WebVid10M"
# with open("/mnt/petrelfs/likunchang/code/vindlu/anno/anno_pretrain/webvid_10m_train.json", "r") as f:
#     json_data = json.load(f)
# path_list = [
#     data['video']
#     for data in json_data
# ]

# iterator = p_imap(partial(get), path_list, position=0, leave=True, ncols=100, dynamic_ncols=False)
    
# for flag, path in iterator:
#     if not flag:
#         path_list.append(path)

# print(len(path_list))
# with open('./error.json', 'w') as f:
#     json.dump(path_list, f)


# lyst=[]
# for i in client.list(dir_path):
#     lyst.append(i)
# pdb.set_trace()

# path = '/mnt/cache/share_data/DSK_datasets/webvid/results_10M_train.csv'
# data = pd.read_csv(path)

# wv_data = []
# for index, row in data.iterrows():
#     videoid, page_dir, name = row['videoid'], row['page_dir'], row['name']
#     wv_data.append(
#         {
#             'video': f"{page_dir}/{videoid}.mp4",
#             'caption': name
#         }
#     )
# print(len(wv_data))

# with open("/mnt/petrelfs/likunchang/code/vindlu/anno/anno_pretrain/webvid_10m_train.json", "w") as f:
#     json.dump(wv_data, f)

# from error_list import error_set
# with open("anno/anno_pretrain/webvid_10m_train.json", "r") as f:
#     json_data = json.load(f)

# print(len(json_data))

# new_json_data = []
# for data in tqdm(json_data):
#     if data["video"] not in error_set:
#         new_json_data.append(data)

# print(len(new_json_data))

# with open("/mnt/petrelfs/likunchang/code/vindlu/anno/anno_pretrain/webvid_10m_train_clean.json", "w") as f:
#     json.dump(new_json_data, f)

# path_list = [
#     "anno/anno_downstream/tvqa_train_with_answer.json",
#     "anno/anno_downstream/tvqa_val_with_answer.json",
#     "anno/anno_downstream/tvqa_test_public_with_answer.json"
# ]

# for path in path_list:
#     with open(path, "r") as f:
#         json_data = json.load(f)

#     prefix = "/mnt/petrelfs/videointern/tvqa/frames_fps3_hq"
#     for data in tqdm(json_data):
#         video_path = os.path.join(prefix, data["vid_name"])
#         num = len(os.listdir(video_path))
#         data["num_frame"] = num

#     with open(path, "w") as f:
#         json.dump(json_data, f)


# with open("anno/anno_downstream/lsmdc_ret_train.json", "r") as f:
#     json_data = json.load(f)
# client = Client('~/petreloss.conf', enable_mc=True)
# for data in tqdm(json_data[58239:58246]):
#     try:
#         prefix = "s3://video_pub/LSMDC"
#         video_path = os.path.join(prefix, data["video"])
#         print("fuck", video_path)
#         video_bytes = client.get(video_path)
#         video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
#     except Exception as e:
#         print(e)
#         print(video_path)


def get(path):
    video_path = os.path.join(prefix, path)
    try:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    except Exception as e:
        print(e)
        print(video_path)
    return None

client = Client('~/petreloss.conf', enable_mc=True)
prefix = "s3://video_pub/LSMDC"
# with open("anno/anno_downstream/lsmdc_ret_train.json", "r") as f:
# with open("anno/anno_downstream/lsmdc_ret_val.json", "r") as f:
# with open("anno/anno_downstream/lsmdc_ret_test.json", "r") as f:
with open("anno/anno_downstream/lsmdc_ret_test_1000.json", "r") as f:
    json_data = json.load(f)
path_list = [
    data['video']
    for data in json_data
    # for data in json_data[:35000]
    # for data in json_data[35000:71000]
    # for data in json_data[71000:]
]

print(len(path_list))

iterator = p_imap(partial(get), path_list, position=0, leave=True, ncols=100, dynamic_ncols=False)


for img_name in iterator:
    pass