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
import scipy.spatial.distance as spd
try:
    import libmr
except ImportError:
    print("LibMR not installed or libmr.so not found")
    print("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

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
    parser.add_argument('--trainset_split', default='data/ucf101/ucf101_train_split_1_videos.txt', help='the split file path of the training set')
    parser.add_argument('--num_cls', type=int, default=101, help='The number of classes in training set.')
    parser.add_argument('--cache_mav_dist', help='the result path to cache the mav and distances for each class.')
    # test data config
    parser.add_argument('--ind_data', help='the split file of in-distribution testing data')
    parser.add_argument('--ood_data', help='the split file of out-of-distribution testing data')
    parser.add_argument('--ood_ncls', type=int, help='the number of classes in unknwon dataset')
    parser.add_argument('--num_rand', type=int, default=10, help='the number of random selection for ood classes')
    # device
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--result_prefix', help='result file prefix')
    args = parser.parse_args()
    return args


def get_datalist(split_file):
    assert os.path.exists(split_file), 'split file does not exist! %s'%(split_file)
    video_dir = os.path.join(os.path.dirname(split_file), 'videos')
    filelist, labels = [], []
    with open(split_file, 'r') as f:
        for line in f.readlines():
            videofile = os.path.join(video_dir, line.strip().split(' ')[0])
            clsid = int(line.strip().split(' ')[-1])
            filelist.append(videofile)
            labels.append(clsid)
    return filelist, labels

def spatial_temporal_pooling(feat_blob):
    if isinstance(feat_blob, tuple):  # slowfast model returns a tuple of features
        assert len(feat_blob) == 2, "invalid feature tuple!"
        avg_pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        x_fast, x_slow = feat_blob
        x_fast = avg_pool3d(x_fast)
        x_slow = avg_pool3d(x_slow)
        # [N, channel_fast + channel_slow, 1, 1, 1]
        feat_clips = torch.cat((x_slow, x_fast), dim=1).squeeze(-1).squeeze(-1).squeeze(-1)
    else:
        if len(feat_blob.size()) == 5:  # 3D Network
            # spatial temporal average pooling 
            kernel_size = (feat_blob.size(-3), feat_blob.size(-2), feat_blob.size(-1))
            avg_pool3d = torch.nn.AvgPool3d(kernel_size, stride=1, padding=0)
            feat_clips = avg_pool3d(feat_blob).view(feat_blob.size(0), feat_blob.size(1))  # (c, D)
        elif len(feat_blob.size()) == 4:  # 2D Network
            # spatial temporal average pooling 
            kernel_size = (feat_blob.size(-2), feat_blob.size(-1))
            avg_pool2d = torch.nn.AvgPool2d(kernel_size, stride=1, padding=0)
            feat_clips = avg_pool2d(feat_blob).view(feat_blob.size(0), feat_blob.size(1))  # (c, D)
        else:
            print('Unsupported feature dimension: {}'.format(feat_blob.size()))
    # get the mean features of all clips and crops
    feat_final = torch.mean(feat_clips, dim=0, keepdim=True)  # (c=1, D)
    return feat_final

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
        feat_blob, score = model(return_loss=False, return_score=True, get_feat=True, **data) # (c, D, t, h, w)
        feat_blob = spatial_temporal_pooling(feat_blob)
        feat_final = feat_blob.cpu().numpy()
        score = score.cpu().numpy()
    return feat_final, score


def extract_class_features(videolist, model, cls_gt):
    features = []
    for videofile in tqdm(videolist, total=len(videolist), desc='Extract Class %d Features'%(cls_gt)):
        feat, score = inference_recognizer(model, videofile)  # (c, D)
        cls_pred = np.argmax(score, axis=1)
        if cls_gt in cls_pred:
            features.append(feat)
    features = np.array(features)  # (N, c, D)
    return features


def compute_distance(mav, features):
    # extract features and compute distances for each class
    num_channels = mav.shape[0]
    eucos_dist, eu_dist, cos_dist = [], [], []
    for feat in features:
        # compute distance of each channel
        eu_channel, cos_channel, eu_cos_channel = [], [], []
        for c in range(num_channels):
            eu_channel += [spd.euclidean(mav[c, :], feat[c, :])/200.]
            cos_channel += [spd.cosine(mav[c, :], feat[c, :])]
            eu_cos_channel += [spd.euclidean(mav[c, :], feat[c, :]) / 200. 
                             + spd.cosine(mav[c, :], feat[c, :])]  # Here, 200 is from the official OpenMax code
        eu_dist += [eu_channel]
        cos_dist += [cos_channel]
        eucos_dist += [eu_cos_channel]
    return np.array(eucos_dist), np.array(eu_dist), np.array(cos_dist)


def compute_channel_distance(mav_channel, feat_channel, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mav_channel, feat_channel)/200. + spd.cosine(mav_channel, feat_channel)
    elif distance_type == 'eu':
        query_distance = spd.euclidean(mav_channel, feat_channel)/200.
    elif distance_type == 'cos':
        query_distance = spd.cosine(mav_channel, feat_channel)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def compute_mav_dist(videolist, labels, model, mav_dist_cachedir):
    num_cls = model.cls_head.num_classes
    mav_dist_list = []
    for cls_gt in range(num_cls):
        mav_dist_file = os.path.join(mav_dist_cachedir, 'mav_dist_cls%03d.npz'%(cls_gt))
        mav_dist_list.append(mav_dist_file)
        if os.path.exists(mav_dist_file):
            continue
        # data for the current class
        inds = np.where(np.array(labels) == cls_gt)[0]
        videos_cls = [videolist[i] for i in inds]
        # extract MAV features
        features = extract_class_features(videos_cls, model, cls_gt)
        mav_train = np.mean(features, axis=0)
        # compute distance
        eucos_dist, eu_dist, cos_dist = compute_distance(mav_train, features)
        # save MAV and distances
        np.savez(mav_dist_file[:-4], mav=mav_train, eucos=eucos_dist, eu=eu_dist, cos=cos_dist)
    return mav_dist_list


def weibull_fitting(mav_dist_list, distance_type='eucos', tailsize=20):
    weibull_model = {}
    for cls_gt in range(len(mav_dist_list)):
        # load the mav_dist file
        cache = np.load(mav_dist_list[cls_gt], allow_pickle=True)
        mav_train = cache['mav']
        distances = cache[distance_type]

        weibull_model[cls_gt] = {}
        weibull_model[cls_gt]['mean_vec'] = mav_train

        # weibull fitting for each channel
        weibull_model[cls_gt]['weibull_model'] = []
        num_channels = mav_train.shape[0]
        for c in range(num_channels):
            mr = libmr.MR()
            tailtofit = sorted(distances[:, c])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[cls_gt]['weibull_model'] += [mr]
    return weibull_model


def compute_openmax_prob(openmax_score, openmax_score_u):
    num_channels, num_cls = openmax_score.shape
    prob_scores, prob_unknowns = [], []
    for c in range(num_channels):
        channel_scores, channel_unknowns = [], []
        for gt_cls in range(num_cls):
            channel_scores += [np.exp(openmax_score[c, gt_cls])]
        
        total_denominator = np.sum(np.exp(openmax_score[c, :])) + np.exp(np.sum(openmax_score_u[c, :]))
        prob_scores += [channel_scores/total_denominator ]
        prob_unknowns += [np.exp(np.sum(openmax_score_u[c, :]))/total_denominator]
        
    prob_scores = np.array(prob_scores)
    prob_unknowns = np.array(prob_unknowns)

    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores =  scores.tolist() + [unknowns]
    assert len(modified_scores) == num_cls + 1
    modified_scores = np.expand_dims(np.array(modified_scores), axis=0)
    return modified_scores


def openmax_recalibrate(weibull_model, feature, score, rank=1, distance_type='eucos'):
    num_channels, num_cls = score.shape
    # get the ranked alpha
    alpharank = min(num_cls, rank)
    ranked_list = np.mean(score, axis=0).argsort().ravel()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = np.zeros((num_cls,))
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    # calibrate
    openmax_score, openmax_score_u = [], []
    for c in range(num_channels):
        channel_scores = score[c, :]
        openmax_channel = []
        openmax_unknown = []
        for cls_gt in range(num_cls):
            # get distance between current channel and mean vector
            mav_train = weibull_model[cls_gt]['mean_vec']
            category_weibull = weibull_model[cls_gt]['weibull_model']
            channel_distance = compute_channel_distance(mav_train[c, :], feature[c, :], distance_type=distance_type)
            # obtain w_score for the distance and compute probability of the distance
            wscore = category_weibull[c].w_score(channel_distance)
            modified_score = channel_scores[cls_gt] * ( 1 - wscore*ranked_alpha[cls_gt] )
            openmax_channel += [modified_score]
            openmax_unknown += [channel_scores[cls_gt] - modified_score]
        # gather modified scores for each channel
        openmax_score += [openmax_channel]
        openmax_score_u += [openmax_unknown]
    openmax_score = np.array(openmax_score)
    openmax_score_u = np.array(openmax_score_u)
    # Pass the recalibrated scores into openmax
    openmax_prob = compute_openmax_prob(openmax_score, openmax_score_u)

    return openmax_prob


def run_inference(model, weibull_model, datalist_file):
    # switch config for different dataset
    cfg = model.cfg
    cfg.data.test.ann_file = datalist_file
    cfg.data.test.data_prefix = os.path.join(os.path.dirname(datalist_file), 'videos')
    cfg.test_cfg.average_clips = 'score'  # we only need scores before softmax layer
    model.cfg.data.videos_per_gpu = 1
    model.cfg.data.workers_per_gpu = 0
    num_cls = model.cls_head.num_classes

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

    model = MMDataParallel(model, device_ids=[0])
    all_softmax, all_openmax, all_gts = [], [], []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            feat_blob, score = model(return_loss=False, return_score=True, get_feat=True, **data)
            softmax_prob = F.softmax(score, dim=1).cpu().numpy()
            # aggregate features
            feat_blob = spatial_temporal_pooling(feat_blob)
            feat_final = feat_blob.cpu().numpy()
        # Re-calibrate score before softmax with OpenMax
        openmax_prob = openmax_recalibrate(weibull_model, feat_final, score.cpu().numpy())
        # gather preds
        all_openmax.append(openmax_prob)
        all_softmax.append(softmax_prob)
        # gather label
        labels = data['label'].numpy()
        all_gts.append(labels)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()

    all_softmax = np.concatenate(all_softmax, axis=0)
    all_openmax = np.concatenate(all_openmax, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    return all_openmax, all_softmax, all_gts


def evaluate_openmax(ind_openmax, ood_openmax, ind_labels, ood_labels, ood_ncls, num_rand=10):
    ind_ncls = model.cls_head.num_classes
    ind_results = np.argmax(ind_openmax, axis=1)
    ood_results = np.argmax(ood_openmax, axis=1)

    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)

    # open-set auc-roc (binary class)
    preds = np.concatenate((ind_results, ood_results), axis=0)
    preds[preds == ind_ncls] = 1  # unknown class
    preds[preds != 1] = 0  # known class
    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
    auc = roc_auc_score(labels, preds)
    print('OpenMax: ClosedSet Accuracy (multi-class): %.3lf, OpenSet AUC (bin-class): %.3lf'%(acc * 100, auc * 100))

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

    # initialize recognition model
    model = init_recognizer(args.config, args.checkpoint, device=device, use_frames=False)
    torch.backends.cudnn.benchmark = True
    model.cfg.data.test.test_mode = True

    ######## Compute the Mean Activation Vector (MAV) and Distances ########
    if not os.path.exists(args.cache_mav_dist):
        os.makedirs(args.cache_mav_dist)
    # parse the video files list of training set
    videolist, labels = get_datalist(args.trainset_split)
    # compute mav and dist
    mav_dist_list = compute_mav_dist(videolist, labels, model, args.cache_mav_dist)


    ######## OOD and IND detection ########
    result_file = os.path.join(args.result_prefix + '_result.npz')
    if not os.path.exists(result_file):
        # prepare result path
        result_dir = os.path.dirname(result_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # Weibull Model by EVT Fitting
        print("Weibull fitting...")
        weibull_model = weibull_fitting(mav_dist_list)
        # run inference (OOD)
        ood_openmax, ood_softmax, ood_labels = run_inference(model, weibull_model, args.ood_data)
        # run inference (OOD)
        ind_openmax, ind_softmax, ind_labels = run_inference(model, weibull_model, args.ind_data)
        # save
        np.savez(result_file[:-4], ind_openmax=ind_openmax, ood_openmax=ood_openmax,
                                   ind_softmax=ind_softmax, ood_softmax=ood_softmax,
                                   ind_label=ind_labels, ood_label=ood_labels)
    else:
        results = np.load(result_file, allow_pickle=True)
        ind_openmax = results['ind_openmax']  # (N1, C+1)
        ood_openmax = results['ood_openmax']  # (N2, C+1)
        ind_softmax = results['ind_softmax']  # (N1, C)
        ood_softmax = results['ood_softmax']  # (N2, C)
        ind_labels = results['ind_label']  # (N1,)
        ood_labels = results['ood_label']  # (N2,)

    ######## Evaluation ########
    openness_list, macro_F1_list, std_list = evaluate_openmax(ind_openmax, ood_openmax, ind_labels, ood_labels, args.ood_ncls, num_rand=args.num_rand)
    
    # draw F1 curve
    plt.figure(figsize=(8,5))  # (w, h)
    plt.plot(openness_list, macro_F1_list, 'r-', linewidth=2)
    # plt.fill_between(openness_list, macro_F1_list - std_list, macro_F1_list + std_list, 'c')
    plt.ylim(0.5, 1.0)
    plt.xlabel('Openness (%)')
    plt.ylabel('macro F1')
    plt.grid('on')
    plt.legend('OpenMax')
    plt.tight_layout()
    dataset_name = args.result_prefix.split('_')[-1]
    png_file = os.path.join(os.path.dirname(args.result_prefix), 'F1_openness_%s.png'%(dataset_name))
    plt.savefig(png_file)
    print('Openness curve figure is saved in: %s'%(png_file))