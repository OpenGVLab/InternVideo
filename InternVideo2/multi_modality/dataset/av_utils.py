import av
import gc
import torch
import torchaudio
import numpy as np
import random
import logging
import io
from torchvision.transforms.functional import pil_to_tensor

logger = logging.getLogger(__name__)



def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def lazy_load_s3video(s3path_video, num_frames, video_start_frame, video_end_frame, client):
    # load video from ceph
    assert client is not None
    video_bytes_stream = client.get(s3path_video, enable_stream_lazyloding=True)
    container = av.open(video_bytes_stream)
    stream = container.streams.video[0]
    # duration = stream.duration
    real_fps = container.streams.video[0].average_rate
    time_base = container.streams.video[0].time_base
    start, end = video_start_frame, video_end_frame
    # Convert time to pts
    duration_frams = end - start + 1
    frames_index = get_index(duration_frams, num_frames)

    pts_list = []

    start_pts = int((start/real_fps) / time_base)
    end_pts = int((end/real_fps) / time_base)
    for frame_index in frames_index:
        pts_list.append(int((frame_index / real_fps)) /  time_base)

    # Seek to nearest key frame from the start
    container.seek(max(start_pts, 0), stream=stream)
    
    frames = []
    for frame in container.decode(**{"video":0}):
        if frame.pts < start_pts:
            continue
        # if frame.pts <= end_pts:
        if len(pts_list) >0:
            if frame.pts >= pts_list[0]:
                frames.append(frame)
                pts_list.pop(0)
        else:
            break
    frames = [pil_to_tensor(frames[idx].to_rgb().to_image()).unsqueeze(0) for idx in range(len(frames))]
    container.close()
    del video_bytes_stream # T C H W

    return torch.cat(frames, dim=0) # , start, end, float(real_fps)


def load_audio_av(video_path, video_start_frame, video_end_frame, sr, max_audio_length, client):  # sr should be 16000
    assert client is not None
    video_bytes_stream = client.get(video_path, enable_stream_lazyloding=True)
    try:
        container = av.open(video_bytes_stream)
    except:
        logger.warn(f"Something wrong when av.open (video_path: {video_path})!")
        return None
    if len(container.streams.audio) == 0:
        logger.warn(f"There is no audio! (video_path: {video_path})!")
        return None
    audio_stream = container.streams.audio[0]
    real_fps = container.streams.video[0].average_rate
    time_base = audio_stream.time_base
    csr = audio_stream.sample_rate
    start_frame, end_frame = video_start_frame, video_end_frame
    start_pts = int((start_frame/real_fps) / time_base)
    end_pts = int((end_frame/real_fps) / time_base)
    frames = []
    container.seek(max(start_pts, 0), stream=audio_stream)
    try:
        for frame in container.decode(**{"audio":0}):
            if frame.pts < start_pts:
                continue
            frames.append(frame.to_ndarray())
            if frame.pts > end_pts:
                break
    except:
        gc.collect()
        pass
    # gc.collect()
    container.close()
    del video_bytes_stream

    audio_raw = np.concatenate(frames, 1)
    audio = torch.from_numpy(audio_raw)
    if audio.size(0) == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    assert max_audio_length == 10, max_audio_length
    max_length = max_audio_length * sr
    if csr != sr:
        trans = torchaudio.transforms.Resample(csr, sr)
        audio = trans(audio)
    if audio.shape[1] >= max_length:
        max_start = audio.shape[1] - max_length
        start = random.randint(0, max_start)
        audio = audio[:, start: start + max_length]
    audio = audio * 2 ** 15
    fbank = torchaudio.compliance.kaldi.fbank(audio, num_mel_bins=64, sample_frequency=16000, frame_length=25, frame_shift=10)
    fbank_mean = 15.41663
    fbank_std = 6.55582
    fbank = (fbank - fbank_mean) / (fbank_std * 2) # 998, 64

    src_length = fbank.shape[0]
    pad_len = 998 - src_length
    fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)
    padding_mask = torch.cat((torch.zeros(1, src_length), torch.ones(1, pad_len)), -1).bool()

    return fbank#, padding_mask

def load_full_audio_av(video_path, sr, max_audio_length, client):
    assert client is not None
    video_bytes_stream = client.get(video_path) #, enable_stream_lazyloding=False)
    try:
        container = av.open(io.BytesIO(video_bytes_stream))
    except Exception as e:
        logger.warn(f"Something wrong {e} when av.open (video_path: {video_path})!")
        return None
    if len(container.streams.audio) == 0:
        logger.warn(f"There is no audio! (video_path: {video_path})!")
        return None
    audio_stream = container.streams.audio[0]
    csr = audio_stream.sample_rate
    frames = []
    try:
        for frame in container.decode(**{"audio":0}):
            frames.append(frame.to_ndarray())
    except:
        gc.collect()
        pass
    # gc.collect()
    container.close()
    del video_bytes_stream

    audio_raw = np.concatenate(frames, 1)
    audio = torch.from_numpy(audio_raw)
    if audio.size(0) == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if len(audio.shape)==1:
        audio = audio.unsqueeze(0)
    assert max_audio_length == 10, max_audio_length
    max_length = max_audio_length * sr
    if csr != sr:
        trans = torchaudio.transforms.Resample(csr, sr)
        audio = trans(audio)
    if audio.shape[1] >= max_length:
        max_start = audio.shape[1] - max_length
        start = random.randint(0, max_start)
        audio = audio[:, start: start + max_length]
    audio = audio * 2 ** 15
    fbank = torchaudio.compliance.kaldi.fbank(audio, num_mel_bins=64, sample_frequency=16000, frame_length=25, frame_shift=10)
    fbank_mean = 15.41663
    fbank_std = 6.55582
    fbank = (fbank - fbank_mean) / (fbank_std * 2) # 998, 64

    src_length = fbank.shape[0]
    pad_len = 998 - src_length
    fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)
    padding_mask = torch.cat((torch.zeros(1, src_length), torch.ones(1, pad_len)), -1).bool()

    return fbank #, padding_mask


    # frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    # # frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8


