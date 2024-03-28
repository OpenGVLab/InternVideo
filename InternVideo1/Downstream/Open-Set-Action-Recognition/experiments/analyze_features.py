import argparse
import os
import torch
from mmcv.parallel import collate, scatter
from mmaction.datasets.pipelines import Compose
from mmaction.apis import init_recognizer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file/url')
    parser.add_argument('--known_split', help='the split file path of the knowns')
    parser.add_argument('--unknown_split', help='the split file path of the unknowns')
    parser.add_argument('--result_file', help='the result file path')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def get_data(known_split, known_classes):
    known_data = []
    labels = []
    video_dir = os.path.join(os.path.dirname(known_split), 'videos')
    with open(known_split, 'r') as f:
        for line in f.readlines():
            clsname, videoname = line.strip().split(' ')[0].split('/')
            if clsname in known_classes.keys():
                videofile = os.path.join(video_dir, clsname, videoname)
                known_data.append(videofile)
                labels.append(known_classes[clsname])
    return known_data, labels


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
        feat_blob = model(return_loss=False, get_feat=True, **data) # (num_clips * num_crops, 2048, 1, 8, 8)
        # spatial average pooling 
        kernel_size = (1, feat_blob.size(-2), feat_blob.size(-1))
        avg_pool2d = torch.nn.AvgPool3d(kernel_size, stride=1, padding=0)
        feat_clips = avg_pool2d(feat_blob).view(feat_blob.size(0), feat_blob.size(1))  # (num_clips * num_crops, 2048)
        # get the mean features of all clips and crops
        feat_final = torch.mean(feat_clips, dim=0).cpu().numpy()  # (2048,)
    return feat_final

def extract_feature(video_files):
    
    model = init_recognizer(
        args.config,
        args.checkpoint,
        device=device,
        use_frames=False)
    cfg = model.cfg
    torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    if 'bnn' in args.config:
        model.test_cfg.npass = 1

    X = []
    for videofile in tqdm(video_files, total=len(video_files), desc='Extract Feature'):
        feature = inference_recognizer(model, videofile)  # (2048,)
        X.append(feature)
    X = np.vstack(X)
    return X


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)
    set_deterministic(0)

    ind_clsID = [2, 10, 16, 69, 71, 21, 32, 41, 73, 29]   # UCF-101  73 21 41 32 29 10 16 69 71  2
    ind_classes = {'Archery': 0, 'Biking': 1, 'BoxingPunchingBag': 2, 'PullUps': 3, 'PushUps': 4, 
                   'CliffDiving': 5, 'GolfSwing': 6, 'HorseRiding': 7, 'RockClimbingIndoor': 8, 'FloorGymnastics': 9}
    ood_clsID = [12, 20, 21, 22, 50, 15, 16, 17, 18, 19]  # HMDB-51  15 16 17 18 19 20 21 22 12 50
    ood_classes = {'fall_floor': 10, 'kick': 10, 'kick_ball': 10, 'kiss': 10, 'wave': 10, 
                   'golf': 10, 'handstand': 10, 'hit': 10, 'hug': 10, 'jump': 10}
    feature_file = args.result_file[:-4] + '_feature_10p1.npz'

    # get the data of known classes
    known_data, known_labels = get_data(args.known_split, ind_classes)
    num_knowns = len(known_data)
    # get the data of unknown classes
    unknown_data, unknown_labels = get_data(args.unknown_split, ood_classes)
    num_unknowns = len(unknown_data)

    if not os.path.exists(feature_file):
        # save the figure
        result_path = os.path.dirname(args.result_file)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # extracting the feature
        X = extract_feature(known_data + unknown_data)
        
        # save results
        np.savez(feature_file[:-4], feature=X)
    else:
        results = np.load(feature_file, allow_pickle=True)
        X = results['feature']

    open_classes = {**ind_classes, 'Unknowns': len(ind_classes)}
    open_labels = np.array(known_labels + [len(ind_classes)] * num_unknowns)
    # run tSNE
    print('running tSNE...')
    Y = TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.figure(figsize=(5,4))
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 10
    for k, v in open_classes.items():
        inds = np.where(open_labels == v)[0]
        if k == 'Unknowns':
            plt.scatter(Y[inds, 0], Y[inds, 1], s=10, c='k', marker='^', label=k)
        else:
            plt.scatter(Y[inds, 0], Y[inds, 1], s=3)
            plt.text(np.mean(Y[inds, 0])-5, np.mean(Y[inds, 1])+5, k, fontsize=fontsize)
    xmin, xmax, ymin, ymax = np.min(Y[:, 0]), np.max(Y[:, 0]), np.min(Y[:, 1]), np.max(Y[:, 1])
    plt.xlim(xmin-5, xmax + 15)
    plt.ylim(ymin-5, ymax + 10)
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.result_file)
    plt.savefig(args.result_file[:-4] + '.pdf')
