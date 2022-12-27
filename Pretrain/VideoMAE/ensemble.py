import os

import numpy as np
from matplotlib import use
from scipy.special import softmax


def merge(eval_paths, num_tasks, use_softmax=False):
    dict_feats = {}
    dict_label = {}
    print("Reading individual output files")

    if not isinstance(eval_paths, list):
        eval_paths = [eval_paths]

    for eval_path in eval_paths:
        dict_pos = {}
        for x in range(num_tasks):
            file = os.path.join(eval_path, str(x) + '.txt')
            lines = open(file, 'r').readlines()[1:]
            for line in lines:
                line = line.strip()
                name = line.split('[')[0]
                label = line.split(']')[1].split(' ')[1]
                chunk_nb = line.split(']')[1].split(' ')[2]
                split_nb = line.split(']')[1].split(' ')[3]
                data = np.fromstring(line.split('[')[1].split(']')[0],
                                     dtype=np.float64,
                                     sep=',')
                if not name in dict_feats:
                    dict_feats[name] = []
                    dict_label[name] = 0
                if not name in dict_pos:
                    dict_pos[name] = []
                if chunk_nb + split_nb in dict_pos[name]:
                    continue
                if use_softmax:
                    dict_feats[name].append(softmax(data))
                else:
                    dict_feats[name].append(data)
                dict_pos[name].append(chunk_nb + split_nb)
                dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    # print(feat.shape)
    try:
        pred = np.argmax(feat)
        top1 = (int(pred) == int(label)) * 1.0
        top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    except:
        pred = 0
        top1 = 1.0
        top5 = 1.0
        label = 0
    return [pred, top1, top5, int(label)]


if __name__ == '__main__':
    # eval_path = '/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/hmdb_pretrain_moco_v3_base_patch16_224_lr_1e-3_32_GPUS_e800/eval_0128xxxx'
    # eval_path = '/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/hmdb_pretrain_mae_from_k400/eval_deepspeed_100ep_lr_5e-4'
    # eval_path = '/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/hmdb_pretrain_moco_v3_base_patch16_224_lr_5e-4_e800/eval_deepspeed_799_ep_60'
    # eval_path = '/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/hmdb_pretrain_mae_base_patch16_224_frame_16x2_tc_mask_0.9_lr_3e-4_new_e9600/eval_deepspeed_9599_ep_60'
    # eval_path = '/apdcephfs/share_1290939/elliottong/work_dir/output_dir/MAE_video/hmdb_pretrain_mae_base_patch16_224_frame_16x2_tc_mask_0.75_lr_3e-4_new_e4800/eval_deepspeed_4799_ep_50'
    eval_path1 = '/mnt/petrelfs/huangbingkun/VideoMAE-clean/work_dir/full_sta_web_k400_finetune'
    eval_path2 = '/mnt/petrelfs/huangbingkun/VideoMAE-clean/work_dir/pred_txts'
    num_tasks = 32
    eval_paths = [eval_path1, eval_path2]
    final_top1, final_top5 = merge(eval_paths, num_tasks, use_softmax=True)
    print(
        f"Accuracy of the network on the test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
    )
