import argparse
import os
import os.path as osp
import torch
import mmcv
from mmaction.apis import init_recognizer
from mmcv.parallel import collate, scatter
from operator import itemgetter
from mmaction.datasets.pipelines import Compose
from mmaction.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel
import numpy as np
from scipy.special import xlogy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model and data config
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file/url')
    parser.add_argument('--uncertainty', default='BALD', choices=['BALD', 'Entropy', 'EDL'], help='the uncertainty estimation method')
    parser.add_argument('--train_data', help='the split file of in-distribution training data')
    parser.add_argument('--forward_pass', type=int, default=10, help='the number of forward passes')
    parser.add_argument('--batch_size', type=int, default=8, help='the testing batch size')
    # env config
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--result_prefix', help='result file prefix')
    args = parser.parse_args()
    return args

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def update_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compute_uncertainty(predictions, method='BALD'):
    """Compute the entropy
       scores: (B x C x T)
    """
    expected_p = np.mean(predictions, axis=-1)  # mean of all forward passes (C,)
    entropy_expected_p = - np.sum(xlogy(expected_p, expected_p), axis=1)  # the entropy of expect_p (across classes)
    if method == 'Entropy':
        uncertain_score = entropy_expected_p
    elif method == 'BALD':
        expected_entropy = - np.mean(np.sum(xlogy(predictions, predictions), axis=1), axis=-1)  # mean of entropies (across classes), (scalar)
        uncertain_score = entropy_expected_p - expected_entropy
    else:
        raise NotImplementedError
    if not np.all(np.isfinite(uncertain_score)):
        uncertain_score[~np.isfinite] = 9999
    return uncertain_score

def run_stochastic_inference(model, data_loader, forward_passes):
    model = MMDataParallel(model, device_ids=[0])
    all_uncertainties = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        all_scores = []
        with torch.no_grad():
            for n in range(forward_passes):
                # set new random seed
                update_seed(n * 1234)
                scores = model(return_loss=False, **data)
                # gather results
                all_scores.append(np.expand_dims(scores, axis=-1))
        all_scores = np.concatenate(all_scores, axis=-1)  # (B, C, T)
        # compute the uncertainty
        uncertainty = compute_uncertainty(all_scores, method=args.uncertainty)
        all_uncertainties.append(uncertainty)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)

    return all_uncertainties


def run_evidence_inference(model, data_loader, evidence):

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

    model = MMDataParallel(model, device_ids=[0])
    all_uncertainties = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            output = model(return_loss=False, **data)
            evidence = get_evidence(torch.from_numpy(output))
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1)
            scores = alpha / torch.sum(alpha, dim=1, keepdim=True)
        all_uncertainties.append(uncertainty.numpy())

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)

    return all_uncertainties


def run_inference():

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(
        args.config,
        args.checkpoint,
        device=device,
        use_frames=False)
    cfg = model.cfg

    if not args.uncertainty == 'EDL':
        # use dropout in testing stage
        if 'dnn' in args.config:
            model.apply(apply_dropout)
        if 'bnn' in args.config:
            model.test_cfg.npass = 1
    # set cudnn benchmark
    torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    cfg.data.videos_per_gpu = args.batch_size
    evidence = cfg.get('evidence', 'exp')

    # We use training data to obtain the threshold
    cfg.data.test.ann_file = args.train_data
    cfg.data.test.data_prefix = os.path.join(os.path.dirname(args.train_data), 'videos')
    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False,
        pin_memory=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # run inference
    if not args.uncertainty == 'EDL':
        all_uncertainties = run_stochastic_inference(model, data_loader, args.forward_pass)
    else:
        all_uncertainties = run_evidence_inference(model, data_loader, evidence)
    return all_uncertainties


if __name__ == '__main__':

    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)

    result_file = os.path.join(args.result_prefix + '_trainset_uncertainties.npz')
    if not os.path.exists(result_file):
        # run the inference on the entire training set (takes long time)
        all_uncertainties = run_inference()
        np.savez(result_file[:-4], uncertainty=all_uncertainties)
    else:
        result = np.load(result_file)
        all_uncertainties = result['uncertainty']

    # evaluation by macro-F1 within (C1 + 1) classes
    uncertain_sort = np.sort(all_uncertainties)[::-1]  # sort the uncertainties with descending order
    N = all_uncertainties.shape[0]
    topK = N - int(N * 0.95)
    threshold = uncertain_sort[topK-1]

    print('The model %s uncertainty threshold on UCF-101 train set: %lf'%(args.result_prefix.split('/')[-1], threshold))