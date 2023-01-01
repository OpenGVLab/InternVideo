"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = ""
    DATASETS = {
        "ava_video_train_v2.2": {
            "video_root": "/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/",
            "ann_file": "/mnt/lustre/share_data/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2_min.json",
            "box_file": "",
            "eval_file_paths": {
                "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2.csv",
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_excluded_timestamps_v2.2.csv",
            },
        },
        "ava_video_val_v2.2": {
            "video_root": "/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval",
            "ann_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2_min.json",
            "box_file": "/mnt/cache/xingsen/ava_dataset/AVA/boxes/ava_val_det_person_bbox.json",
            "eval_file_paths": {
                "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2.csv",
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            },
        },
        "kinetics_val":{
            'kinetics_annfile':"/mnt/cache/xingsen/xingsen2/kinetics_val_v1.0.json",
            "box_file":"/mnt/cache/xingsen/xingsen2/kinetics_person_box.json",
            "eval_file_paths":{
            "csv_gt_file" : '/mnt/cache/xingsen/xingsen2/kinetics_val_gt_v1.0.csv',
            "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
            "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            }
        },
        "ak_train":{
            "video_root" : "/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/",
            "ava_annfile": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2_min.json",
            "kinetics_annfile":"/mnt/cache/xingsen/xingsen2/kinetics_train_v1.0.json",
            "ava_eval_file_path":{
                "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2.csv",
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_excluded_timestamps_v2.2.csv"
            }
        },
        "ak_val":{
            "video_root":"/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/",
            "ava_annfile":"/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2_min.json",
            "kinetics_annfile":"/mnt/cache/xingsen/xingsen2/kinetics_val_v1.0.json",
            "kinetics_box_file":"/mnt/cache/xingsen/xingsen2/kinetics_person_box.json",
            "ava_box_file":"/mnt/cache/xingsen/ava_dataset/AVA/boxes/ava_val_det_person_bbox.json",
            "eval_file_paths":{
                "csv_gt_file":'/mnt/cache/xingsen/xingsen2/ak_val_gt.csv',
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            },
            "ava_eval_file_path":{
                "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2.csv",
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            },
            "kinetics_eval_file_path":{
                "csv_gt_file" : '/mnt/cache/xingsen/xingsen2/kinetics_val_gt_v1.0.csv',
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv"
            }
        }
    }

    @staticmethod
    def get(name):
        if "ava_video" in name:
            data_dir = DatasetCatalog.DATA_DIR
            args = DatasetCatalog.DATASETS[name]
            return dict(
                factory="Dataset",
                args=args
            )
        raise RuntimeError("Dataset not available: {}".format(name))
