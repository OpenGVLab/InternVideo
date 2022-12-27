import argparse
import os
import torch
import mmcv
from mmaction.apis import init_recognizer
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.core.evaluation import top_k_accuracy
from mmcv.parallel import MMDataParallel
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('--split_file', help='the split file for evaluation')
    parser.add_argument('--video_path', help='the video path for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--result_prefix', help='result file prefix')
    args = parser.parse_args()
    return args


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def init_inference():
    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(args.config, args.checkpoint, device=device, use_frames=False)
    cfg = model.cfg
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.split_file
    cfg.data.test.data_prefix = args.video_path
    evidence = cfg.get('evidence', 'exp')
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
    return model, data_loader, evidence


def run_evidence_inference(model, data_loader, evidence='exp'):
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

    # run inference
    model = MMDataParallel(model, device_ids=[0])
    all_scores, all_uncertainties, all_labels = [], [], []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            output = model(return_loss=False, **data)
            evidence = get_evidence(torch.from_numpy(output))
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1)
            scores = alpha / torch.sum(alpha, dim=1, keepdim=True)
        all_uncertainties.append(uncertainty.numpy())
        all_scores.append(scores.numpy())
        labels = data['label'].numpy()
        all_labels.append(labels)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    all_scores = np.concatenate(all_scores, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_scores, all_uncertainties, all_labels

    
def main():

    result_file = os.path.join(args.result_prefix + '_result.npz')
    if not os.path.exists(result_file):
        # prepare result path
        result_dir = os.path.dirname(result_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # init
        model, data_loader, evidence = init_inference()
        # run inference
        all_scores, all_uncertainties, all_labels = run_evidence_inference(model, data_loader, evidence)
        # save
        np.savez(result_file[:-4], score=all_scores, uncertainty=all_uncertainties, label=all_labels)
    else:
        results = np.load(result_file, allow_pickle=True)
        all_scores = results['score']
        all_uncertainties = results['uncertainty']
        all_labels = results['label']  # (N2,)

    # evaluation on bias/unbiased closed set
    top_k_acc = top_k_accuracy(all_scores, all_labels, (1, 5))
    print('Evaluation Results:\ntop1_acc: %.5lf, top5_acc: %.5lf\n'%(top_k_acc[0], top_k_acc[1]))
        

if __name__ == '__main__':

    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)
    set_deterministic(1234)

    main()
