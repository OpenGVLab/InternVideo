import os
from venv import main
import torch.utils.data as data
import time
import torch
import numpy as np
import sys
import torch.nn.functional as F
sys.path.append('/mnt/cache/xingsen/xingsen2/VideoMAE_ava')
from alphaction.structures.bounding_box import BoxList
# from transforms import build_transforms
from collections import defaultdict
from alphaction.utils.video_decode import av_decode_video
from alphaction.dataset.datasets.ava import NpInfoDict, NpBoxDict
import pdb
import json
import decord
import logging
from tqdm import tqdm


class AKDataset(data.Dataset):
    def __init__(self,kinetics_annfile,ava_annfile,video_root,frame_span,remove_clips_without_annotations,kinetics_box_file=None,ava_box_file=None,
                 ava_eval_file_path={},kinetics_eval_file_path={},eval_file_paths={},box_thresh=0.0,action_thresh=0.0,transforms=None):
        self.action_thresh = action_thresh
        self.box_thresh = box_thresh
        self.eval_file_paths = eval_file_paths
        self.remove_clips_without_annotations = remove_clips_without_annotations
        self.kinetics_dataset = KineticsDataset(kinetics_annfile=kinetics_annfile,
                                                frame_span=frame_span,
                                                remove_clips_without_annotations=True,
                                                box_file=kinetics_box_file,
                                                eval_file_paths=kinetics_eval_file_path,
                                                box_thresh=box_thresh,
                                                action_thresh=action_thresh,
                                                transforms=transforms)
        self.ava_dataset = AVAVideoDataset(
            video_root=video_root,
            ann_file=ava_annfile,
            frame_span=frame_span,
            remove_clips_without_annotations=remove_clips_without_annotations,
            box_file=ava_box_file,
            eval_file_paths=ava_eval_file_path,
            box_thresh=box_thresh,
            action_thresh=action_thresh,
            transforms=transforms)
        
    def __len__(self):
        return len(self.kinetics_dataset) + len(self.ava_dataset)
    
    def __getitem__(self, index):
        if self.remove_clips_without_annotations:
            missing_id = [17261,35964,52484,97042]
            if index in missing_id:
                return self.kinetics_dataset[-1]
        if index < len(self.kinetics_dataset):
            return self.kinetics_dataset[index]
        else:
            return self.ava_dataset[index - len(self.kinetics_dataset)]

    def get_video_info(self, index):
        if index < len(self.kinetics_dataset):
            return self.kinetics_dataset.get_video_info(index)
        else:
            return self.ava_dataset.get_video_info(index - len(self.kinetics_dataset))
        


class KineticsDataset(data.Dataset):
    def __init__(self,kinetics_annfile,frame_span,remove_clips_without_annotations, box_file=None,
                 eval_file_paths={},box_thresh=0.0,action_thresh=0.0,transforms=None):
        print('loading annotations into memory...')
        tic = time.time()
        self.kinetics_annfile = kinetics_annfile
        kinetics_dict = json.load(open(kinetics_annfile,'r'))
        assert type(kinetics_dict) == dict, 'annotation file format {} not supported'.format(type(kinetics_dict))
        self.transforms = transforms
        self.frame_span = frame_span
        self.video_map = np.load('../annotations/video_path_k700.npy',allow_pickle=True).item()
        self.video_size_map = np.load('../annotations/movie_size.npy',allow_pickle=True).item()
        # These two attributes are used during ava evaluation...
        # Maybe there is a better implementation
        self.eval_file_paths = eval_file_paths
        self.action_thresh = action_thresh
        self.clip2ann_kinetics = self.load_ann_file(kinetics_dict)
        self.movie_info_kinetics, self.clip_ids_kinetics, self.clips_info_kinetics = self.get_videos_info(kinetics_dict)
        
        if remove_clips_without_annotations:
            self.clip_ids_kinetics = [clip_id for clip_id in self.clip_ids_kinetics if clip_id in self.clip2ann_kinetics]
        # pdb.set_trace()
        valid_id = self.del_invalid_id()    #23315
        self.clip_ids_kinetics = valid_id
        if box_file:
            imgToBoxes = self.load_box_file(box_file,box_thresh)    # val len 23224
            clip_ids = [
                img_id
                for img_id in self.clip_ids_kinetics    #
                if len(imgToBoxes[img_id]) > 0
            ]   #val len 21738
            self.clip_ids_kinetics = clip_ids
            self.det_persons = NpBoxDict(imgToBoxes,clip_ids,
                               value_types=[("bbox", np.float32), ("score", np.float32)])   #21738
        else:
            self.det_persons = None

        self.anns = NpBoxDict(self.clip2ann_kinetics,self.clip_ids_kinetics,value_types=[("bbox", np.float32), ("packed_act", np.uint8)])
        clips_info = {
            clip_id:
            [
                self.movie_info_kinetics.convert_key(self.clips_info_kinetics[clip_id][0]),
                self.clips_info_kinetics[clip_id][1]
            ] for clip_id in self.clip_ids_kinetics
        }
        self.clips_info = NpInfoDict(clips_info,value_type=np.int32)
        # pdb.set_trace()
        super().__init__()
    
    def __getitem__(self, idx):
        _, clip_info = self.clips_info[idx]

        mov_id, timestamp = clip_info

        movie_id, _ = self.movie_info_kinetics[mov_id]     
        video_data = self._decode_video_data(movie_id, timestamp)
        _,im_h,im_w,_ = video_data.shape
        if self.det_persons is None:
            # Note: During training, we only use gt. Thus we should not provide box file,
            # otherwise we will use only box file instead.
            boxes, packed_act = self.anns[idx]
            boxes = boxes * np.array([im_w, im_h, im_w, im_h])
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # guard against no boxes
            boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xywh").convert("xyxy")

            # Decode the packed bits from uint8 to one hot, since AVA has 80 classes,
            # it can be exactly denoted with 10 bytes, otherwise we may need to discard some bits.
            one_hot_label = np.unpackbits(packed_act, axis=1)
            one_hot_label = torch.as_tensor(one_hot_label, dtype=torch.uint8)

            boxes.add_field("labels", one_hot_label)  # 80

        else:
            boxes, box_score = self.det_persons[idx]
            logging.info("box is {},h:{},w:{}".format(boxes,im_h,im_w))
            boxes = boxes * np.array([im_w, im_h, im_w, im_h])
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xyxy")

            box_score_tensor = torch.as_tensor(box_score, dtype=torch.float32).reshape(-1, 1)
            boxes.add_field("scores", box_score_tensor)

        boxes = boxes.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            video_data, boxes, transform_randoms = self.transforms(video_data, boxes)

        return video_data, boxes, idx

    def __len__(self):
        return len(self.clips_info)

    def _decode_video_data(self, dirname, timestamp):
        # decode target video data from segment per second.
        video_path = self.video_map[dirname]
        time_list = video_path[:-4].split('_')[-2:]
        start_time = int(time_list[0])
        end_time = int(time_list[1])
        time_offset = (timestamp - start_time)/(end_time-start_time)
        frames = av_decode_video(video_path)
        mid_frame = int(time_offset * len(frames))
        right_span = self.frame_span//2 + mid_frame
        left_span = right_span - self.frame_span
        if left_span < 0 and right_span <= len(frames):
            frames_left = self.numpy2tensorinterpolate(frames[0:mid_frame],self.frame_span-(self.frame_span//2))
            frames_right = frames[mid_frame:right_span]
            frames = np.concatenate([frames_left,np.stack(frames_right)],axis=0)
        elif left_span >= 0 and right_span > len(frames):
            frames_left = frames[left_span:mid_frame]
            try:
                frames_right = self.numpy2tensorinterpolate(frames[mid_frame:],self.frame_span//2)
                frames = np.concatenate([np.stack(frames_left),frames_right],axis=0)
            except ValueError:
                print(mid_frame,len(frames),time_offset,timestamp,start_time,end_time,dirname)
                exit(0)
        else:
            frames = np.stack(frames[left_span:right_span])

        return frames

    def numpy2tensorinterpolate(self,aArray,time_len):
        aArray = torch.from_numpy(np.stack(aArray)).unsqueeze(0).float()
        aArray = aArray.permute(0,4,1,2,3)     # 1 * channel * time * h * w
        _, c, t, h, w = aArray.shape
        aArray = F.interpolate(aArray,size=[time_len,h,w],mode='trilinear',align_corners=False)
        aArray = aArray.squeeze()
        aArray = np.uint8(aArray.permute(1,2,3,0).numpy())
        return aArray

    def del_invalid_id(self):
        valid_clip_ids = []
        for clip_id in self.clip_ids_kinetics:
            timestamp = int(clip_id.split('_')[-1])
            assert timestamp == self.clips_info_kinetics[clip_id][1]
            mov_id = self.movie_info_kinetics.convert_key(self.clips_info_kinetics[clip_id][0])
            movie_id,_ = self.movie_info_kinetics[mov_id]
            video_path = self.video_map[movie_id]
            try:
                time_list = video_path[:-4].split('_')[-2:]
            except TypeError:
                continue
            start_time = int(time_list[0])
            end_time = int(time_list[1])
            if timestamp > start_time and timestamp < end_time:
                valid_clip_ids.append(clip_id)
        return valid_clip_ids

    def load_ann_file(self,json_dict):
        clip2ann = defaultdict(list)
        if "annotations" in json_dict:
            for ann in json_dict["annotations"]:
                try:
                    action_ids = np.array(ann["action_ids"],dtype=np.uint8)
                except ValueError:
                    continue
                one_hot = np.zeros(81, dtype=np.bool)
                one_hot[action_ids] = True
                packed_act = np.packbits(one_hot[1:])
                clip2ann[ann["image_id"]].append(dict(bbox=ann["bbox"], packed_act=packed_act))
        return clip2ann

    def get_videos_info(self,json_dict):
        movies_size = {}
        clips_info = {}
        for img in json_dict["images"]:
            mov = img["movie"]
            if mov not in movies_size:
                movies_size[mov] = [img["width"],img["height"]]
            clips_info[img["id"]] = [mov, img["timestamp"]]
        movie_info = NpInfoDict(movies_size, value_type=np.int32)
        clip_ids = sorted(list(clips_info.keys()))
        return movie_info, clip_ids, clips_info
    
    def get_video_info(self, index):
        _, clip_info = self.clips_info[index]
        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info_kinetics[mov_id]
        h, w = self.video_size_map[movie_id]
        return dict(width=w, height=h, movie=movie_id, timestamp=timestamp)
    
    def load_box_file(self, box_file, score_thresh=0.0):
        import json

        print('Loading box file into memory...')
        tic = time.time()
        with open(box_file, "r") as f:
            box_results = json.load(f)
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        boxImgIds = [box['image_id'] for box in box_results]

        imgToBoxes = defaultdict(list)
        for img_id, box in zip(boxImgIds, box_results):
            if box['score'] >= score_thresh:
                imgToBoxes[img_id].append(box)
        return imgToBoxes

    def getitem(self,idx):
        return self.__getitem__(idx)


class AVAVideoDataset(data.Dataset):
    def __init__(self, video_root, ann_file, remove_clips_without_annotations, frame_span, box_file=None,
                 eval_file_paths={}, box_thresh=0.0, action_thresh=0.0, transforms=None):

        print('loading annotations into memory...')
        tic = time.time()
        json_dict = json.load(open(ann_file, 'r'))
        assert type(json_dict) == dict, 'annotation file format {} not supported'.format(type(json_dict))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        self.video_root = video_root
        self.transforms = transforms
        self.frame_span = frame_span

        # These two attributes are used during ava evaluation...
        # Maybe there is a better implementation
        self.eval_file_paths = eval_file_paths
        self.action_thresh = action_thresh

        clip2ann = defaultdict(list)
        if "annotations" in json_dict:
            for ann in json_dict["annotations"]:
                action_ids = ann["action_ids"]
                one_hot = np.zeros(81, dtype=np.bool)
                one_hot[action_ids] = True
                packed_act = np.packbits(one_hot[1:])
                clip2ann[ann["image_id"]].append(dict(bbox=ann["bbox"], packed_act=packed_act))

        movies_size = {}
        clips_info = {}
        for img in json_dict["images"]:
            mov = img["movie"]
            if mov not in movies_size:
                movies_size[mov] = [img["width"], img["height"]]
            clips_info[img["id"]] = [mov, img["timestamp"]]
        self.movie_info = NpInfoDict(movies_size, value_type=np.int32)
        clip_ids = sorted(list(clips_info.keys()))

        if remove_clips_without_annotations:
            clip_ids = [clip_id for clip_id in clip_ids if clip_id in clip2ann]

        if box_file:
            # this is only for validation or testing
            # we use detected boxes, so remove clips without boxes detected.
            imgToBoxes = self.load_box_file(box_file, box_thresh)
            clip_ids = [
                img_id
                for img_id in clip_ids
                if len(imgToBoxes[img_id]) > 0
            ]
            self.det_persons = NpBoxDict(imgToBoxes, clip_ids,
                                         value_types=[("bbox", np.float32), ("score", np.float32)])
        else:
            self.det_persons = None

        self.anns = NpBoxDict(clip2ann, clip_ids, value_types=[("bbox", np.float32), ("packed_act", np.uint8)])

        clips_info = {  # key
            clip_id:
                [
                    self.movie_info.convert_key(clips_info[clip_id][0]),
                    clips_info[clip_id][1]
                ] for clip_id in clip_ids
        }
        self.clips_info = NpInfoDict(clips_info, value_type=np.int32)

    def __getitem__(self, idx):

        _, clip_info = self.clips_info[idx]

        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info[mov_id]
        video_data = self._decode_video_data(movie_id, timestamp)

        im_w, im_h = movie_size

        if self.det_persons is None:
            # Note: During training, we only use gt. Thus we should not provide box file,
            # otherwise we will use only box file instead.

            boxes, packed_act = self.anns[idx]

            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # guard against no boxes
            boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xywh").convert("xyxy")

            # Decode the packed bits from uint8 to one hot, since AVA has 80 classes,
            # it can be exactly denoted with 10 bytes, otherwise we may need to discard some bits.
            one_hot_label = np.unpackbits(packed_act, axis=1)
            one_hot_label = torch.as_tensor(one_hot_label, dtype=torch.uint8)

            boxes.add_field("labels", one_hot_label)  # 80

        else:
            boxes, box_score = self.det_persons[idx]
            boxes_tensor = torch.as_tensor(boxes).reshape(-1, 4)
            boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xywh").convert("xyxy")

            box_score_tensor = torch.as_tensor(box_score, dtype=torch.float32).reshape(-1, 1)
            boxes.add_field("scores", box_score_tensor)

        boxes = boxes.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            video_data, boxes, transform_randoms = self.transforms(video_data, boxes)

        return video_data, boxes, idx

    def get_video_info(self, index):
        _, clip_info = self.clips_info[index]
        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info[mov_id]
        w, h = movie_size
        return dict(width=w, height=h, movie=movie_id, timestamp=timestamp)

    def load_box_file(self, box_file, score_thresh=0.0):
        import json

        print('Loading box file into memory...')
        tic = time.time()
        with open(box_file, "r") as f:
            box_results = json.load(f)
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        boxImgIds = [box['image_id'] for box in box_results]

        imgToBoxes = defaultdict(list)
        for img_id, box in zip(boxImgIds, box_results):
            if box['score'] >= score_thresh:
                imgToBoxes[img_id].append(box)
        return imgToBoxes

    def _decode_video_data(self, dirname, timestamp):
        # decode target video data from segment per second.

        video_folder = os.path.join(self.video_root, dirname)
        right_span = self.frame_span//2
        left_span = self.frame_span - right_span

        #load right
        cur_t = timestamp
        right_frames = []
        while len(right_frames)<right_span:
            video_path = os.path.join(video_folder, "{}.mp4".format(cur_t))
            # frames = cv2_decode_video(video_path)
            frames = av_decode_video(video_path)
            if len(frames)==0:
                raise RuntimeError("Video {} cannot be decoded.".format(video_path))
            right_frames = right_frames+frames
            cur_t += 1

        #load left
        cur_t = timestamp-1
        left_frames = []
        while len(left_frames)<left_span:
            video_path = os.path.join(video_folder, "{}.mp4".format(cur_t))
            # frames = cv2_decode_video(video_path)
            frames = av_decode_video(video_path)
            if len(frames)==0:
                raise RuntimeError("Video {} cannot be decoded.".format(video_path))
            left_frames = frames+left_frames
            cur_t -= 1

        #adjust key frame to center, usually no need
        min_frame_num = min(len(left_frames), len(right_frames))
        frames = left_frames[-min_frame_num:] + right_frames[:min_frame_num]

        video_data = np.stack(frames)

        return video_data

    def __len__(self):
        return len(self.clips_info)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Video Root Location: {}\n'.format(self.video_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def build_ak_dataset(is_train,transforms):
    input_filename = "ak_train" if is_train else "ak_val"
    assert input_filename
    data_args = {}
    if is_train:
        data_args['video_root'] = '/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/'
        data_args['ava_annfile'] = "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2_min.json"
        data_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_train_v1.0.json"
        data_args['transforms'] = transforms
        data_args['remove_clips_without_annotations'] = True
        data_args['frame_span'] = 64
        data_args["ava_eval_file_path"] = {
                "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2.csv",
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_excluded_timestamps_v2.2.csv",
            }
    else:
        data_args['video_root'] = '/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/'
        data_args['ava_annfile'] = "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2_min.json"
        data_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_val_v1.0.json"
        data_args['box_thresh'] = 0.8
        data_args['action_thresh'] = 0.
        data_args['transforms'] = transforms
        data_args['remove_clips_without_annotations'] = True
        data_args['kinetics_box_file'] = '/mnt/cache/xingsen/xingsen2/kinetics_person_box.json'
        data_args['ava_box_file'] = '/mnt/cache/xingsen/ava_dataset/AVA/boxes/ava_val_det_person_bbox.json'
        data_args['frame_span'] = 64  # 64
        data_args['eval_file_paths'] = {
                "csv_gt_file":'/mnt/cache/xingsen/xingsen2/ak_val_gt.csv',
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            }
        data_args["ava_eval_file_path"] = {
                "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2.csv",
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            }
        data_args['kinetics_eval_file_path'] = {
            "csv_gt_file" : '/mnt/cache/xingsen/xingsen2/kinetics_val_gt_v1.0.csv',
            "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
            "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
        }
        

    dataset = AKDataset(**data_args)
    return dataset



movie_size = {}
def check_valid(path,movie):
    try:
        reader = decord.VideoReader(path)
        frame = reader.get_batch([0]).asnumpy()
    except Exception:
        print(path)
        return False
    return True

if __name__ == '__main__':
    transform_val = None
    dataset = build_ak_dataset(True,transform_val)
    for i in tqdm(range(len(dataset))):
        dataset[i]

    
    