import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

import pickle

import io
from petrel_client.client import Client
client = Client()

@register_dataset("thumos")
class THUMOS14Dataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling # force to upsample to max_seq_len
    ):
        # file path
        # # assert os.path.exists(feat_folder) and os.path.exists(json_file)
        # assert client.isdir(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
            # if not client.contains(feat_file):
                continue
            # if not os.path.exists(os.path.join('/mnt/petrelfs/liuyi/tsp_th14_mae_h',self.file_prefix + key+'.pkl')):
            #     continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)

        
        feats_i3d = np.load(filename).astype(np.float32)
        
        # feats_mae=pickle.load(io.BytesIO(client.get(os.path.join('s3://video_pub/thumos_feature/stride_8',video_item['id']+'.pkl'))) )
        # feats_mae=np.load(io.BytesIO(client.get(os.path.join('s3://video_pub/thumos_feature/clip_16_4/',video_item['id']+'.npy'))) ).astype(np.float32)
        # feats_mae=pickle.load(io.BytesIO(client.get(os.path.join('s3://video_pub/thumos_feature/',video_item['id']+'.pkl'))) )



        feats_mae=np.load(os.path.join('/mnt/petrelfs/liuyi/data_feature/data_th14/th14_mae_g_16_4',video_item['id']+'.npy')).astype(np.float32)
        # feats_mae=pickle.load(io.BytesIO(client.get(os.path.join('/mnt/petrelfs/liuyi/tsp_th14_mae_h',video_item['id']+'.pkl'))) )

        feats_mae = F.interpolate(torch.from_numpy(feats_mae).permute(1,0).unsqueeze(0), size=feats_i3d.shape[0], mode='linear',align_corners=False)[0,...].permute(1,0)
        feats_mae = feats_mae.numpy()



        # feats_clip=np.load(io.BytesIO(client.get(os.path.join('/mnt/petrelfs/liuyi/th14_clipk700_16_4',video_item['id']+'.npy'))) ).astype(np.float32)
        # feats_clip = F.interpolate(torch.from_numpy(feats_clip).permute(1,0).unsqueeze(0), size=feats_i3d.shape[0], mode='linear',align_corners=False)[0,...].permute(1,0)
        # feats_clip = feats_clip.numpy()
        # feats = np.concatenate([feats_clip, feats_mae],  axis=1)

        feats=feats_mae

        # feats=feats_clip

        
        # feats = feats_i3d[:,0:1024]

        # feats=feats_i3d[:,1024:]

        




        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps'] - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict
