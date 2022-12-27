import argparse
import os
import os.path as osp
import torch
import mmcv
from mmcv import Config, DictAction
from mmaction.apis import init_recognizer
from mmcv.parallel import collate, scatter
from operator import itemgetter
from mmaction.datasets.pipelines import Compose
from mmaction.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import numpy as np
from scipy.special import xlogy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pdb
from mmaction.models import build_model
from mmcv.cnn import fuse_conv_bn
from mmaction.apis import collect_results_cpu

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file/url')
    parser.add_argument('--uncertainty', choices=['BALD', 'Entropy', 'EDL'], help='the uncertainty estimation method')
    parser.add_argument('--forward_pass', type=int, default=10, help='the number of forward passes')
    # data config
    parser.add_argument('--ind_data', help='the split file of in-distribution testing data')
    parser.add_argument('--ood_data', help='the split file of out-of-distribution testing data')
    # env config
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--result_prefix', help='result file prefix')
    parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
    '--cfg-options',
    nargs='+',
    action=DictAction,
    default={},
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file. For example, '
    "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
    '--fuse-conv-bn',
    action='store_true',
    help='Whether to fuse conv and bn, this will slightly increase'
    'the inference speed')

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
    

def run_stochastic_inference(model, data_loader, npass=10):
    # run inference
    # model = MMDataParallel(model, device_ids=[0])
    # model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    
    model.eval()

    all_confidences, all_uncertainties, all_results, all_gts = [], [], [], []
    # prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        all_scores = []
        with torch.no_grad():
            for n in range(npass):
                # set new random seed
                update_seed(n * 1234)
        #-original--
        #         scores = model(return_loss=False, **data)
        #         # gather results
        #         all_scores.append(np.expand_dims(scores, axis=-1))
        # all_scores = np.concatenate(all_scores, axis=-1)  # (B, C, T)
        # # compute the uncertainty
        # uncertainty = compute_uncertainty(all_scores, method=args.uncertainty)
        # all_uncertainties.append(uncertainty)
        #---vae---------------------------------
        #         # pdb.set_trace()
        #         scores, recon = model(return_loss=False, **data)
        #         uncertainty = recon
        #         all_scores.append(np.expand_dims(scores, axis=-1))
        # all_scores = np.concatenate(all_scores, axis=-1)  # (B, C, T)
        # all_uncertainties.append(uncertainty)
        #-----------------------------------------------
        #---FLOW---------------------------------
                # pdb.set_trace()
                scores, logpx = model(return_loss=False, **data)
                uncertainty = logpx
                all_scores.append(np.expand_dims(scores, axis=-1))
        all_scores = np.concatenate(all_scores, axis=-1)  # (B, C, T)
        all_uncertainties.append(uncertainty)
        #-----------------------------------------------

        # compute the predictions and save labels
        mean_scores = np.mean(all_scores, axis=-1)
        preds = np.argmax(mean_scores, axis=1)
        all_results.append(preds)
        conf = np.max(mean_scores, axis=1)
        all_confidences.append(conf)

        labels = data['label'].numpy()
        all_gts.append(labels)

        # # use the first key as main key to calculate the batch size
        # batch_size = len(next(iter(data.values())))
        # for _ in range(batch_size):
        #     prog_bar.update()

        if rank == 0:
        # use the first key as main key to calculate the batch size
            batch_size = len(next(iter(data.values())))
            for _ in range(batch_size * world_size):
                prog_bar.update()
    # pdb.set_trace()
    all_confidences = collect_results_cpu(all_confidences, len(data_loader.dataset), tmpdir=None) 
    all_uncertainties = collect_results_cpu(all_uncertainties, len(data_loader.dataset), tmpdir=None)  
    all_results = collect_results_cpu(all_results, len(data_loader.dataset), tmpdir=None)  
    all_gts = collect_results_cpu(all_gts, len(data_loader.dataset), tmpdir=None)
    rank, _ = get_dist_info()
    if rank == 0:
        all_confidences = np.concatenate(all_confidences, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
        all_results = np.concatenate(all_results, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
    # pdb.set_trace()
    return all_confidences, all_uncertainties, all_results, all_gts


def run_evidence_inference(model, data_loader, evidence='exp'):
    # set new random seed
    update_seed(1234)
    # get the evidence function
    if evidence == 'relu':
        from mmaction.models.losses.edl_loss import relu_evidence as get_evidence
    elif evidence == 'exp':
        from mmaction.models.losses.edl_loss import exp_evidence as get_evidence
    elif evidence == 'softplus':
        from mmaction.models.losses.edl_loss import softplus_evidence as get_evidence
    else:
        raise NotImplementedError
    # pdb.set_trace()
    num_classes = 101
    model.eval()

    # run inference
    # model = MMDataParallel(model, device_ids=[0])
    all_confidences, all_uncertainties, all_results, all_gts = [], [], [], []
    # prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            output = model(return_loss=False, **data)
            evidence = get_evidence(torch.from_numpy(output))
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1)
            scores = alpha / torch.sum(alpha, dim=1, keepdim=True)
        all_uncertainties.append(uncertainty.numpy())

            #---vae---------------------------------
        #     output, recon = model(return_loss=False, **data)
        #     evidence = get_evidence(torch.from_numpy(output))
        #     alpha = evidence + 1
        #     scores = alpha / torch.sum(alpha, dim=1, keepdim=True)
        #     uncertainty = recon
        # all_uncertainties.append(uncertainty)
        #-------------------------------------------
        # pdb.set_trace()
        # compute the predictions and save labels
        preds = np.argmax(scores.numpy(), axis=1)
        all_results.append(preds)
        conf = np.max(scores.numpy(), axis=1)
        all_confidences.append(conf)

        labels = data['label'].numpy()
        all_gts.append(labels)

        # use the first key as main key to calculate the batch size
        # batch_size = len(next(iter(data.values())))
        # for _ in range(batch_size):
        #     prog_bar.update()
        if rank == 0:
        # use the first key as main key to calculate the batch size
            batch_size = len(next(iter(data.values())))
            for _ in range(batch_size * world_size):
                prog_bar.update()
    # pdb.set_trace()
    all_confidences = collect_results_cpu(all_confidences, len(data_loader.dataset), tmpdir=None) 
    all_uncertainties = collect_results_cpu(all_uncertainties, len(data_loader.dataset), tmpdir=None)  
    all_results = collect_results_cpu(all_results, len(data_loader.dataset), tmpdir=None)  
    all_gts = collect_results_cpu(all_gts, len(data_loader.dataset), tmpdir=None)
    rank, _ = get_dist_info()
    if rank == 0:
        # pdb.set_trace()
        all_confidences = np.concatenate(all_confidences, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
        all_results = np.concatenate(all_results, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
    # pdb.set_trace()
    return all_confidences, all_uncertainties, all_results, all_gts


def run_inference(model, distributed, cfg, datalist_file, npass=10):
    # switch config for different dataset
    # cfg = model.cfg
    
    cfg.data.test.ann_file = datalist_file
    cfg.data.test.data_prefix = os.path.join(os.path.dirname(datalist_file), 'videos')
    evidence = cfg.get('evidence', 'exp')

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False,
        pin_memory=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # pdb.set_trace()
    if not args.uncertainty == 'EDL':
        all_confidences, all_uncertainties, all_results, all_gts = run_stochastic_inference(model, data_loader, npass)
    else:
        all_confidences, all_uncertainties, all_results, all_gts = run_evidence_inference(model, data_loader, evidence)
    return all_confidences, all_uncertainties, all_results, all_gts


def main():
    # pdb.set_trace()
    # build the recognizer from a config file and checkpoint file/url
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    print('cpk is loaded:', args.checkpoint)

    # model = init_recognizer(
    #     args.config,
    #     args.checkpoint,
    #     device=device,
    #     use_frames=False)
    # cfg = model.cfg

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)


    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)

    # if not args.uncertainty == 'EDL':
    #     # use dropout in testing stage
    #     if 'dnn' in args.config:
    #         model.apply(apply_dropout)
    #     if 'bnn' in args.config:
    #         model.test_cfg.npass = 1

    # set cudnn benchmark
    torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    
    result_file = os.path.join(args.result_prefix + '_result.npz')
    if not os.path.exists(result_file):
        # prepare result path
        result_dir = os.path.dirname(result_file)
        if not os.path.exists(result_dir):
            # os.makedirs(result_dir)
            os.makedirs(result_dir, exist_ok=True)

        # run inference (OOD)
        # pdb.set_trace()
        ood_confidences, ood_uncertainties, ood_results, ood_labels = run_inference(model, distributed, cfg, args.ood_data, npass=args.forward_pass)
        # run inference (IND)
        # pdb.set_trace()
        ind_confidences, ind_uncertainties, ind_results, ind_labels = run_inference(model, distributed, cfg, args.ind_data, npass=args.forward_pass)
 
        # save
        np.savez(result_file[:-4], ind_conf=ind_confidences, ood_conf=ood_confidences,
                                   ind_unctt=ind_uncertainties, ood_unctt=ood_uncertainties, 
                                   ind_pred=ind_results, ood_pred=ood_results,
                                   ind_label=ind_labels, ood_label=ood_labels)
    else:
        results = np.load(result_file, allow_pickle=True)
        ind_confidences = results['ind_conf']
        ood_confidences = results['ood_conf']
        ind_uncertainties = results['ind_unctt']  # (N1,)
        ood_uncertainties = results['ood_unctt']  # (N2,)
        ind_results = results['ind_pred']  # (N1,)
        ood_results = results['ood_pred']  # (N2,)
        ind_labels = results['ind_label']
        ood_labels = results['ood_label']
    # visualize
    # pdb.set_trace()
    # torch.distributed.barrier()
    # ood_confidences = np.concatenate(ood_confidences, axis=0)
    # ood_uncertainties = np.concatenate(ood_uncertainties, axis=0)
    # ood_results = np.concatenate(ood_results, axis=0)
    # ood_labels = np.concatenate(ood_labels, axis=0)
    # ind_confidences = np.concatenate(ind_confidences, axis=0)
    # ind_uncertainties = np.concatenate(ind_uncertainties, axis=0)
    # ind_results = np.concatenate(ind_results, axis=0)
    # ind_labels = np.concatenate(ind_labels, axis=0)
    # pdb.set_trace()
    rank, _ = get_dist_info()
    if rank == 0:
        ind_uncertainties = np.array(ind_uncertainties.squeeze())
        # ind_uncertainties = (ind_uncertainties-np.min(ind_uncertainties)) / (np.max(ind_uncertainties) - np.min(ind_uncertainties)) # normalize
        ood_uncertainties = np.array(ood_uncertainties.squeeze())
        # ood_uncertainties = (ood_uncertainties-np.min(ood_uncertainties)) / (np.max(ood_uncertainties) - np.min(ood_uncertainties)) # normalize

        # ind和ood合在一起nomalize
        all_uncertainties = np.concatenate((ind_uncertainties, ood_uncertainties))
        ind_uncertainties = (ind_uncertainties-np.min(all_uncertainties)) / (np.max(all_uncertainties) - np.min(all_uncertainties)) # normalize
        ood_uncertainties = (ood_uncertainties-np.min(all_uncertainties)) / (np.max(all_uncertainties) - np.min(all_uncertainties)) # normalize

        dataName_ind = args.ind_data.split('/')[-2].upper()
        dataName_ood = args.ood_data.split('/')[-2].upper()
        if dataName_ind == 'UCF101':
            dataName_ind = 'UCF-101'
        if dataName_ood == 'MIT':
            dataName_ood = 'MiT-v2'
        if dataName_ood == 'HMDB51':
            dataName_ood = 'HMDB-51'
        plt.figure(figsize=(5,4))  # (w, h)
        plt.rcParams["font.family"] = "Arial"  # Times New Roman
        fontsize = 15
        # pdb.set_trace()
        plt.hist([ind_uncertainties, ood_uncertainties], 50, 
                density=True, histtype='bar', color=['blue', 'red'], 
                label=['in-distribution (%s)'%(dataName_ind), 'out-of-distribution (%s)'%(dataName_ood)])
        plt.legend(fontsize=fontsize)
        plt.xlabel('%s uncertainty'%(args.uncertainty), fontsize=fontsize)
        plt.ylabel('density', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlim(0, 1.01)
        plt.ylim(0, 10.01)
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_prefix + '_distribution.png'))
        plt.savefig(os.path.join(args.result_prefix + '_distribution.pdf'))

if __name__ == '__main__':

    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)

    main()
