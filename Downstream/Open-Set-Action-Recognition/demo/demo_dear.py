import os
import numpy as np
from numpy.core.fromnumeric import size
import cv2
import torch
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import CompositeVideoClip, ImageSequenceClip, TextClip, VideoFileClip



def read_mapping(mapfile):
    assert os.path.exists(mapfile), 'Mapping file does not exist! %s'%(mapfile)
    mapping_dict = dict()
    with open(mapfile, 'r') as f:
        for line in f.readlines():
            cls_id = int(line.strip().split(' ')[0]) - 1 # class_ID starts with 0
            cls_name = line.strip().split(' ')[1]
            mapping_dict.update({cls_id: cls_name})
    return mapping_dict


def read_list(list_file, mapping):
    assert os.path.exists(list_file), 'List file does not exist! %s'%(list_file)
    videos_path = os.path.join(os.path.dirname(list_file), 'videos')
    video_list = dict()
    for k, v in mapping.items():
        video_list.update({k: []})
    with open(list_file, 'r') as f:
        for line in f.readlines():
            vid_file = os.path.join(videos_path, line.strip().split(' ')[0])
            gt_cls = int(line.strip().split(' ')[1])
            video_list[gt_cls].append(vid_file)
    return video_list


def inference_recognizer(model, video_path):
    """Inference a video with the detector.

    Args:
        model (nn.Module): The loaded recognizer.
        video_path (str): The video file path/url or the rawframes directory
            path. If ``use_frames`` is set to True, it should be rawframes
            directory path. Otherwise, it should be video file path.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data (by default, we use videodata)
    start_index = cfg.data.test.get('start_index', 0)
    data = dict(filename=video_path, label=-1, start_index=start_index, modality='RGB')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)[0]
    return scores


def evidential_prediction(logits):
    # get evidence
    evidence = np.exp(np.clip(logits, -10, 10))   # (K,)
    # get predicted class_ID
    pred_cls = int(np.argmax(evidence))  # scalar, int()
    # get uncertainty
    alpha = evidence + 1
    S = np.sum(alpha)  # scalar
    uncertainty = logits.size / S  # scalar, vacuity uncertainty
    return pred_cls, uncertainty, evidence


def read_video(video_file):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    while (ret):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_data.append(frame)
        ret, frame = cap.read()
    return video_data


def visualize(outfile, video_data, gt_label, pred_label, uncertainty, evidence, max_evidence=20000, fps=30, fontsize=25):
    """ video_data: list(), size of each element is (H, W, C)
        pred_cls: int(), scalar value
        uncertainty: float(), scalar value
        evidence: ndarray(), size = (K,)
        mapping: dict(), length=K
    """
    height, width = video_data[0].shape[:2]
    num_cls = len(evidence)
    # class_names = list(mapping.values())
    class_ids = range(num_cls)
    # produce the video frames of evidence diagram
    fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi=300)  # by default, dpi=100, pixel_width=4000
    ax.bar(class_ids, evidence)
    plt.xlim(0, num_cls)
    plt.ylim(0, max_evidence)
    plt.ylabel('Evidence', fontsize=fontsize)
    plt.xlabel('Known Action Types', fontsize=fontsize)
    plt.xticks([], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.text(10, int(max_evidence * 0.8), 'Ground Truth: %s'%(gt_label), color='y', fontsize=fontsize)
    plt.text(10, int(max_evidence * 0.7), 'Prediction: %s'%(pred_label), color='r', fontsize=fontsize)
    plt.text(10, int(max_evidence * 0.6), 'Uncertainty: %.6f'%(uncertainty), color='g', fontsize=fontsize)
    plt.tight_layout()

    fig.canvas.draw()
    image_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_fig = image_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    h_fig, w_fig = image_fig.shape[:2]  # (250, 2000)
    # reshape it to align with the width of video frames
    
    h_align = max(min(int(width * h_fig // w_fig), height), 1)
    image_fig = cv2.resize(image_fig, (width, h_align))

    last_frame = video_data[-1]
    last_frame[height - h_align:] = cv2.addWeighted(last_frame[height - h_align:], 0.3, image_fig, 0.7, 0)
    vis_data = video_data + [last_frame] * fps * 5  # last for 5 seconds after the video finished playing

    video_clips = ImageSequenceClip(vis_data, fps=fps)
    video_clips.write_gif(outfile)



def vis_known(result_dir, test_data, model, threshold, mapping_open):
    for gt_cls, video_list in test_data.items():
        # for each class, we randomly select 1 videos
        videos_keep = np.random.choice(video_list, size=1, replace=False).tolist()
        for vid_file in videos_keep:
            # get the NN output logits
            logits = inference_recognizer(model, vid_file)  # (101,) ndarray
            pred_cls, uncertainty, evidence = evidential_prediction(logits)
            if uncertainty > threshold:
                pred_cls = logits.size  # K
            # visualization
            video_data = read_video(vid_file)
            vis_file = os.path.join(result_dir, '%s_%s.gif'%(mapping_open[gt_cls], vid_file.split('/')[-1].split('.')[0]))
            visualize(vis_file, video_data, mapping_open[gt_cls], mapping_open[pred_cls], uncertainty, evidence)


def vis_unknown(result_dir, test_data, model, threshold, mapping_open, mapping_unknown):
    i= 0
    for gt_cls, video_list in test_data.items():
        # for each class, we randomly select 1 videos
        videos_keep = np.random.choice(video_list, size=1, replace=False).tolist()
        for vid_file in videos_keep:
            i+= 1
            if i < 6:
                continue
            # get the NN output logits
            logits = inference_recognizer(model, vid_file)  # (101,) ndarray
            pred_cls, uncertainty, evidence = evidential_prediction(logits)
            if uncertainty > threshold:
                pred_cls = logits.size  # K
            # visualization
            video_data = read_video(vid_file)
            vis_file = os.path.join(result_dir, '%s_%s.gif'%(mapping_unknown[gt_cls], vid_file.split('/')[-1].split('.')[0]))
            visualize(vis_file, video_data, 'Unknown', mapping_open[pred_cls], uncertainty, evidence)
            

def main():
    np.random.seed(123)
    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer('configs/recognition/slowfast/inference_slowfast_enn.py',
                            'work_dirs/slowfast/finetune_ucf101_slowfast_edlnokl_avuc_debias/latest.pth',
                             device=torch.device('cuda:0'),
                             use_frames=False)
    threshold = 0.004552
    # make sure the outputs of the mode are the NN logits
    assert model.cfg.evidence == 'exp', 'Use exponential evidence by setting cfg.evidence=exp !'
    assert model.cfg.test_cfg['average_clips'] == 'score', 'Please set average_clips==score in cfg.test_cfg!'

    # read class mapping file
    mapping_dict = read_mapping('data/ucf101/annotations/classInd.txt')
    mapping_dict.update({max(list(mapping_dict.keys()))+1: 'Unknown'})
    # read test data list (known)
    test_known = read_list('data/ucf101/ucf101_val_split_1_videos.txt', mapping_dict)
    result_known_dir = 'demo/ucf101'
    os.makedirs(result_known_dir, exist_ok=True)
    vis_known(result_known_dir, test_known, model, threshold, mapping_dict)

    # read class mapping file
    mapping_unknown_dict = read_mapping('data/hmdb51/annotations/classInd.txt')
    # read test data list (unknown)
    test_unknown = read_list('data/hmdb51/hmdb51_val_split_1_videos.txt', mapping_unknown_dict)
    result_unknown_dir = 'demo/hmdb51'
    os.makedirs(result_unknown_dir, exist_ok=True)
    vis_unknown(result_unknown_dir, test_unknown, model, threshold, mapping_dict, mapping_unknown_dict)


if __name__ == '__main__':
    
    main()


    

            