import argparse
import os
import torch
from mmaction.apis import init_recognizer
from mmcv.parallel import collate, scatter
from mmaction.datasets.pipelines import Compose
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--ckpt_dear', help='checkpoint file/url')
    parser.add_argument('--ckpt_nodebias', help='checkpoint file/url')
    parser.add_argument('--split_file', help='the split file for evaluation')
    parser.add_argument('--video_path', help='the video path for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--result_list', help='result file prefix')
    args = parser.parse_args()
    return args


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def init_inference(config, checkpoint):
    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(config, checkpoint, device=device, use_frames=False)
    cfg = model.cfg
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.split_file
    cfg.data.test.data_prefix = args.video_path
    evidence = cfg.get('evidence', 'exp')
    return model, evidence


def parse_listfile(list_file, videos_path):
    assert os.path.exists(list_file), 'split file does not exist! %s'%(list_file)
    assert os.path.exists(videos_path), 'video path does not exist! %s'%(videos_path)
    # parse file list
    filelist, labels = [], []
    with open(list_file, 'r') as f:
        for line in f.readlines():
            videofile = line.strip().split(' ')[0]
            label = int(line.strip().split(' ')[1])
            videofile_full = os.path.join(videos_path, videofile)
            assert os.path.exists(videofile_full), 'video file does not exist! %s'%(videofile_full)
            filelist.append(videofile_full)
            labels.append(label)
    return filelist, labels


def run_evidence_inference(model, video_path, evidence='exp'):
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

    # get the evidence function
    if evidence == 'relu':
        from mmaction.models.losses.edl_loss import relu_evidence as get_evidence
    elif evidence == 'exp':
        from mmaction.models.losses.edl_loss import exp_evidence as get_evidence
    elif evidence == 'softplus':
        from mmaction.models.losses.edl_loss import softplus_evidence as get_evidence
    else:
        raise NotImplementedError
    num_classes = model.cls_head.num_classes

    # forward the model
    with torch.no_grad():
        output = model(return_loss=False, **data)[0]  # batchsize = 1
        evidence = get_evidence(torch.from_numpy(output))
        alpha = evidence + 1
        uncertainty = num_classes / torch.sum(alpha, dim=0)
        scores = alpha / torch.sum(alpha, dim=0, keepdim=True)
    return scores.cpu().numpy(), uncertainty.cpu().numpy()


def main():
    model_dear, evidence_dear = init_inference(args.config, args.ckpt_dear)
    model_nodebias, evidence_nodebias = init_inference(args.config, args.ckpt_nodebias)
    # result file
    result_path = os.path.dirname(args.result_list)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    fid = open(args.result_list, 'w')
    # run inference
    videofiles, labels = parse_listfile(args.split_file, args.video_path)
    for i, (videofile, label) in tqdm(enumerate(zip(videofiles, labels)), total=len(videofiles)):
        scores_dear, uncertainty_dear = run_evidence_inference(model_dear, videofile, evidence_dear)  # (101,)
        scores_nodebias, uncertainty_nodebias = run_evidence_inference(model_nodebias, videofile, evidence_nodebias)  # (101,)
        # save
        pred_dear = int(np.argmax(scores_dear))
        pred_nodebias = int(np.argmax(scores_nodebias))
        if pred_dear == label and pred_nodebias != label:
            fid.writelines('%s %d %d %.6lf %d %.6lf\n'%(videofile, label, pred_dear, float(uncertainty_dear), pred_nodebias, float(uncertainty_nodebias)))
    fid.close()


if __name__ == '__main__':

    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)
    set_deterministic(1234)

    main()
    