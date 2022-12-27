"""
Used to compress video in: https://github.com/ArrowLuo/CLIP4Clip
Author: ArrowLuo
"""
import os
import argparse
import ffmpeg
import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
# multiprocessing.freeze_support()

def compress(paras):
    input_video_path, output_video_path = paras
    try:
        command = ['ffmpeg',
                   '-y',  # (optional) overwrite output file if it exists
                   '-i', input_video_path,
                   '-filter:v',
                   'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',  # scale to 224
                   '-map', '0:v',
                   '-r', '3',  # frames per second
                   output_video_path,
                   ]
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        retcode = ffmpeg.poll()
        # print something above for debug
    except Exception as e:
        raise e

def prepare_input_output_pairs(input_root, output_root):
    input_video_path_list = []
    output_video_path_list = []
    for root, dirs, files in os.walk(input_root):
        for file_name in files:
            input_video_path = os.path.join(root, file_name)
            output_video_path = os.path.join(output_root, file_name)
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                pass
            else:
                input_video_path_list.append(input_video_path)
                output_video_path_list.append(output_video_path)
    return input_video_path_list, output_video_path_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress video for speed-up')
    parser.add_argument('--input_root', type=str, help='input root')
    parser.add_argument('--output_root', type=str, help='output root')
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root

    assert input_root != output_root

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    input_video_path_list, output_video_path_list = prepare_input_output_pairs(input_root, output_root)

    print("Total video need to process: {}".format(len(input_video_path_list)))
    num_works = cpu_count()
    print("Begin with {}-core logical processor.".format(num_works))

    pool = Pool(num_works)
    data_dict_list = pool.map(compress,
                              [(input_video_path, output_video_path) for
                               input_video_path, output_video_path in
                               zip(input_video_path_list, output_video_path_list)])
    pool.close()
    pool.join()

    print("Compress finished, wait for checking files...")
    for input_video_path, output_video_path in zip(input_video_path_list, output_video_path_list):
        if os.path.exists(input_video_path):
            if os.path.exists(output_video_path) is False or os.path.getsize(output_video_path) < 1.:
                shutil.copyfile(input_video_path, output_video_path)
                print("Copy and replace file: {}".format(output_video_path))