import io
import os
import random

import cv2
import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from petrel_client.client import Client
from PIL import Image


class HybridVideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_{:05}.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 num_sample=1):

        super(HybridVideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        # temp for hybrid train
        self.ava_fname_tmpl = 'image_{:06}.jpg'
        self.ssv2_fname_tmpl = 'img_{:05}.jpg'

        self.decord = True

        self.client = Client()

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 root + "\n"
                                 "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        while True:
            try:
                video_name, start_idx, total_frame, target = self.clips[index]
                if total_frame < 0:
                    self.new_step = 4
                    self.skip_length = self.new_length * self.new_step
                    video_bytes = self.client.get(video_name)
                    decord_vr = VideoReader(io.BytesIO(video_bytes),
                                            num_threads=1,
                                            ctx=cpu(0))
                    duration = len(decord_vr)

                    segment_indices, skip_offsets = self._sample_train_indices(
                        duration)
                    frame_id_list = self.get_frame_id_list(
                        duration, segment_indices, skip_offsets)
                    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
                    images = [
                        Image.fromarray(
                            video_data[vid, :, :, :]).convert('RGB')
                        for vid, _ in enumerate(frame_id_list)
                    ]
                    break

                else:
                    # ssv2 & ava
                    if 'SomethingV2' in video_name:
                        self.new_step = 2
                        self.skip_length = self.new_length * self.new_step
                        fname_tmpl = self.ssv2_fname_tmpl
                    elif 'AVA' in video_name:
                        self.new_step = 4
                        self.skip_length = self.new_length * self.new_step
                        fname_tmpl = self.ava_fname_tmpl
                    else:
                        self.new_step = 4
                        self.skip_length = self.new_length * self.new_step
                        fname_tmpl = self.name_pattern

                    segment_indices, skip_offsets = self._sample_train_indices(
                        total_frame)
                    frame_id_list = self.get_frame_id_list(
                        total_frame, segment_indices, skip_offsets)

                    images = []
                    for idx in frame_id_list:
                        frame_fname = os.path.join(
                            video_name, fname_tmpl.format(idx + start_idx))
                        img_bytes = self.client.get(frame_fname)
                        img_np = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                        images.append(Image.fromarray(img))
                    break
                break

            except Exception as e:
                print("Failed to load video from {} with error {}".format(
                    video_name, e))
                index = random.randint(0, len(self.clips) - 1)

        if self.num_sample > 1:
            process_data_list = []
            mask_list = []
            for _ in range(self.num_sample):
                process_data, mask = self.transform((images, None))
                process_data = process_data.view(
                    (self.new_length, 3) + process_data.size()[-2:]).transpose(
                        0, 1)
                process_data_list.append(process_data)
                mask_list.append(mask)
            return process_data_list, mask_list
        else:
            process_data, mask = self.transform((images, None))
            # T*C,H,W -> T,C,H,W -> C,T,H,W
            process_data = process_data.view(
                (self.new_length, 3) + process_data.size()[-2:]).transpose(
                    0, 1)
            return process_data, mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, root, setting):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s'
                        % line))
                clip_path = os.path.join(root, line_info[0])
                start_idx = int(line_info[1])
                total_frame = int(line_info[2])
                target = int(line_info[-1])  #往往最后一列才是类别标签
                item = (clip_path, start_idx, total_frame, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(num_frames - self.skip_length + 1,
                                  size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step,
                                             size=self.skip_length //
                                             self.new_step)
        else:
            skip_offsets = np.zeros(self.skip_length // self.new_step,
                                    dtype=int)
        return offsets + 1, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return frame_id_list


class VideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_{:05}.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 num_sample=1):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        self.decord = True

        self.client = Client()

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 root + "\n"
                                 "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        while True:
            try:
                video_name, start_idx, total_frame, target = self.clips[index]
                if total_frame < 0:  # load video
                    if video_name.startswith('s3:'):
                        video_bytes = self.client.get(video_name)
                        decord_vr = VideoReader(io.BytesIO(video_bytes),
                                                num_threads=1,
                                                ctx=cpu(0))
                    else:
                        decord_vr = VideoReader(video_name,
                                                num_threads=1,
                                                ctx=cpu(0))
                    duration = len(decord_vr)

                    segment_indices, skip_offsets = self._sample_train_indices(
                        duration)
                    frame_id_list = self.get_frame_id_list(
                        duration, segment_indices, skip_offsets)
                    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
                    images = [
                        Image.fromarray(
                            video_data[vid, :, :, :]).convert('RGB')
                        for vid, _ in enumerate(frame_id_list)
                    ]
                else:  # load frames
                    segment_indices, skip_offsets = self._sample_train_indices(
                        total_frame)
                    frame_id_list = self.get_frame_id_list(
                        total_frame, segment_indices, skip_offsets)

                    images = []
                    for idx in frame_id_list:
                        frame_fname = os.path.join(
                            video_name,
                            self.name_pattern.format(idx + start_idx))
                        if frame_fname.startswith('s3:'):
                            img_bytes = self.client.get(frame_fname)
                            img_np = np.frombuffer(img_bytes, np.uint8)
                            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                            img = Image.fromarray(img)
                        else:
                            with open(frame_fname, 'rb') as f:
                                img = Image.open(f)
                                img = img.convert('RGB')
                        images.append(img)
                # walk through the if-else branch without error
                # break out the while-true loop
                break

            except Exception as e:
                print("Failed to load video from {} with error {}".format(
                    video_name, e))
                index = random.randint(0, len(self.clips) - 1)

        if self.num_sample > 1:
            process_data_list = []
            mask_list = []
            for _ in range(self.num_sample):
                process_data, mask = self.transform((images, None))
                process_data = process_data.view(
                    (self.new_length, 3) + process_data.size()[-2:]).transpose(
                        0, 1)
                process_data_list.append(process_data)
                mask_list.append(mask)
            return process_data_list, mask_list
        else:
            process_data, mask = self.transform((images, None))
            # T*C,H,W -> T,C,H,W -> C,T,H,W
            process_data = process_data.view(
                (self.new_length, 3) + process_data.size()[-2:]).transpose(
                    0, 1)
            return process_data, mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, root, setting):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s'
                        % line))
                clip_path = os.path.join(root, line_info[0])
                start_idx = int(line_info[1])
                total_frame = int(line_info[2])
                target = int(line_info[-1])
                item = (clip_path, start_idx, total_frame, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(num_frames - self.skip_length + 1,
                                  size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step,
                                             size=self.skip_length //
                                             self.new_step)
        else:
            skip_offsets = np.zeros(self.skip_length // self.new_step,
                                    dtype=int)
        return offsets + 1, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return frame_id_list


class OldVideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 num_sample=1):

        super(OldVideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        self.decord = True

        conf_path = '~/petreloss.conf'
        client = Client(conf_path)
        self.client = client
        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 root + "\n"
                                 "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        while True:
            try:
                directory, target = self.clips[index]
                if self.video_loader:
                    if '.' in directory.split('/')[-1]:
                        video_name = directory
                    else:
                        video_name = '{}.{}'.format(directory, self.video_ext)
                    if video_name.startswith('s3'):
                        video_bytes = self.client.get(video_name)
                        decord_vr = VideoReader(io.BytesIO(video_bytes),
                                                num_threads=1,
                                                ctx=cpu(0))
                    else:
                        decord_vr = VideoReader(video_name,
                                                num_threads=1,
                                                ctx=cpu(0))
                    duration = len(decord_vr)
                    if duration > 0:
                        break

            except Exception as e:
                print("Failed to load video from {} with error {}".format(
                    video_name, e))
            index = random.randint(0, len(self.clips) - 1)

        if self.num_sample > 1:
            process_data_list = []
            mask_list = []
            for _ in range(self.num_sample):
                process_data, mask = self.get_one_sample(
                    decord_vr, duration, directory)
                process_data_list.append(process_data)
                mask_list.append(mask)
            return process_data_list, mask_list
        else:
            return self.get_one_sample(decord_vr, duration, directory)

    def get_one_sample(self, decord_vr, duration, directory):
        decord_vr.seek(0)

        segment_indices, skip_offsets = self._sample_train_indices(duration)
        images = self._video_TSN_decord_batch_loader(directory, decord_vr,
                                                     duration, segment_indices,
                                                     skip_offsets)

        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data, mask = self.transform((images, None))
        process_data = process_data.view((self.new_length, 3) +
                                         process_data.size()[-2:]).transpose(
                                             0, 1)
        return process_data, mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s'
                        % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[-1])
                item = (clip_path, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(num_frames - self.skip_length + 1,
                                  size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step,
                                             size=self.skip_length //
                                             self.new_step)
        else:
            skip_offsets = np.zeros(self.skip_length // self.new_step,
                                    dtype=int)
        return offsets + 1, skip_offsets

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration,
                                       indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [
                Image.fromarray(video_data[vid, :, :, :]).convert('RGB')
                for vid, _ in enumerate(frame_id_list)
            ]
        except:
            raise RuntimeError(
                'Error occured in reading frames {} from video {} of duration {}.'
                .format(frame_id_list, directory, duration))
        return sampled_list
