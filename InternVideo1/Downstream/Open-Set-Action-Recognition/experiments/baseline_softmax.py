import argparse, os, sys
import torch
import mmcv
import numpy as np
import torch.nn.functional as F
from mmcv.parallel import collate, scatter
from mmaction.datasets.pipelines import Compose
from mmaction.apis import init_recognizer
from mmaction.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


def parse_args():
    """ Example shell script:
        $ cd experiments
        $ source activate mmaction
        $ nohup python baseline_openmax.py --model i3d --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py
    """
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file/url')
    parser.add_argument('--train_data', default='data/ucf101/ucf101_train_split_1_videos.txt', help='the split file of in-distribution training data')
    parser.add_argument('--batch_size', type=int, default=8, help='the testing batch size')
    parser.add_argument('--ood_ncls', type=int, help='the number of classes in unknwon dataset')
    # device
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


def run_inference(config, checkpoint, train_data, batch_size, device):
    # initialize recognition model
    model = init_recognizer(config, checkpoint, device=device, use_frames=False)
    torch.backends.cudnn.benchmark = True
    model.cfg.data.test.test_mode = True
    model.cfg.test_cfg.average_clips = 'prob'  # we need the probability socore from softmax layer
    model.cfg.data.videos_per_gpu = batch_size  # batch size of training set
    # We use training data to obtain the threshold
    model.cfg.data.test.ann_file = train_data
    model.cfg.data.test.data_prefix = os.path.join(os.path.dirname(train_data), 'videos')
    
    # build the dataloader
    dataset = build_dataset(model.cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=model.cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=model.cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False,
        pin_memory=False)
    dataloader_setting = dict(dataloader_setting, **model.cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # running the inference
    model = MMDataParallel(model, device_ids=[0])
    all_scores = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            scores = model(return_loss=False, **data)  # (B, C)
            all_scores.append(scores)
        # use the first key as main key to calculate the batch size
        bs = len(next(iter(data.values())))
        for _ in range(bs):
            prog_bar.update()
    all_scores = np.concatenate(all_scores, axis=0)
    return all_scores


def evaluate_softmax(ind_softmax, ood_softmax, ind_labels, ood_labels, ood_ncls, thresh, num_rand=10):

    ind_ncls = ind_softmax.shape[1]
    ind_results = np.argmax(ind_softmax, axis=1)
    ood_results = np.argmax(ood_softmax, axis=1)
    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)

    # open-set auc-roc (binary class)
    ind_conf = np.max(ind_softmax, axis=1)
    ood_conf = np.max(ood_softmax, axis=1)
    preds = np.concatenate((ind_results, ood_results), axis=0)
    confs = np.concatenate((ind_conf, ood_conf), axis=0)
    preds[confs < threshold] = 1  # unknown class
    preds[confs >= threshold] = 0  # known class
    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
    auc = roc_auc_score(labels, preds)
    print('SoftMax: ClosedSet Accuracy (multi-class): %.3lf, OpenSet AUC (bin-class): %.3lf'%(acc * 100, auc * 100))

    ind_results[ind_conf < thresh] = ind_ncls  # incorrect rejection
    # open set F1 score (multi-class)
    macro_F1_list = [f1_score(ind_labels, ind_results, average='macro')]
    std_list = [0]
    openness_list = [0]
    for n in range(ood_ncls):
        ncls_novel = n + 1
        openness = (1 - np.sqrt((2 * ind_ncls) / (2 * ind_ncls + ncls_novel))) * 100
        openness_list.append(openness)
        # randoml select the subset of ood samples
        macro_F1_multi = np.zeros((num_rand), dtype=np.float32)
        for m in range(num_rand):
            cls_select = np.random.choice(ood_ncls, ncls_novel, replace=False) 
            ood_sub_results = np.concatenate([ood_results[ood_labels == clsid] for clsid in cls_select])
            ood_sub_labels = np.ones_like(ood_sub_results) * ind_ncls
            ood_sub_confs = np.concatenate([ood_conf[ood_labels == clsid] for clsid in cls_select])
            ood_sub_results[ood_sub_confs < thresh] = ind_ncls  # correct rejection
            # construct preds and labels
            preds = np.concatenate((ind_results, ood_sub_results), axis=0)
            labels = np.concatenate((ind_labels, ood_sub_labels), axis=0)
            macro_F1_multi[m] = f1_score(labels, preds, average='macro')
        macro_F1 = np.mean(macro_F1_multi)
        std = np.std(macro_F1_multi)
        macro_F1_list.append(macro_F1)
        std_list.append(std)

    # draw comparison curves
    macro_F1_list = np.array(macro_F1_list)
    std_list = np.array(std_list)

    w_openness = np.array(openness_list) / 100.
    open_maF1_mean = np.sum(w_openness * macro_F1_list) / np.sum(w_openness)
    open_maF1_std = np.sum(w_openness * std_list) / np.sum(w_openness)
    print('Open macro-F1 score: %.3f, std=%.3lf'%(open_maF1_mean * 100, open_maF1_std * 100))

    return openness_list, macro_F1_list, std_list


if __name__ == '__main__':
    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)
    set_deterministic(0)
    modelname, methodname, dataname = args.result_prefix.split('/')[-1].split('_')

    ######## Compute threshold with training data ########
    result_file = os.path.join(os.path.dirname(args.result_prefix), modelname +'_SoftMax_trainset_softmax.npz')
    if not os.path.exists(result_file):
        # prepare result path
        result_dir = os.path.dirname(result_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # run the inference on training data
        trainset_scores = run_inference(args.config, args.checkpoint, args.train_data, args.batch_size, device)
        # save
        np.savez(result_file[:-4], trainset_scores=trainset_scores)
    else:
        result = np.load(result_file)
        trainset_scores = result['trainset_scores']
    
    max_scores = np.max(trainset_scores, axis=1)
    scores_sort = np.sort(max_scores)[::-1]  # sort the uncertainties with descending order
    N = max_scores.shape[0]
    threshold = scores_sort[int(N * 0.95)-1]

    print('\nThe model %s softmax threshold on UCF-101 train set: %lf'%(args.result_prefix.split('/')[-1], threshold))

    # load the softmax results on testing dataset (in OpenMax baseline)
    ######## OOD and IND detection ########
    openmax_result = os.path.join(os.path.dirname(args.result_prefix), '../openmax', modelname +'_OpenMax_'+ dataname +'_result.npz')
    if not os.path.exists(openmax_result):
        print('File does not exist! %s'%(openmax_result))
        print('Run baseline_openmax.py first to get softmax testing results!')
    else:
        results = np.load(openmax_result, allow_pickle=True)
        ind_softmax = results['ind_softmax']  # (N1, C)
        ood_softmax = results['ood_softmax']  # (N2, C)
        ind_labels = results['ind_label']  # (N1,)
        ood_labels = results['ood_label']  # (N2,)
    
    openness_list, macro_F1_list, std_list = evaluate_softmax(ind_softmax, ood_softmax, ind_labels, ood_labels, args.ood_ncls, threshold)