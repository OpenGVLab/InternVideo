import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description='Draw histogram')
    parser.add_argument('--uncertainty', default='EDL', choices=['BALD', 'Entropy', 'EDL'], help='the uncertainty estimation method')
    parser.add_argument('--ind_data', default='UCF-101', help='the split file of in-distribution testing data')
    parser.add_argument('--ood_data', default='HMDB', choices=['HMDB', 'MiT'], help='the split file of out-of-distribution testing data')
    parser.add_argument('--model', default='I3D', choices=['I3D', 'TSM', 'SlowFast', 'TPN'], help='the action recognition model.')
    parser.add_argument('--result_prefix', default='temp/temp.png', help='result file prefix')
    args = parser.parse_args()
    return args

def plot_by_uncertainty(result_file, uncertainty='EDL', auc=80, fontsize=16):
    assert os.path.exists(result_file), 'result file not exists! %s'%(result_file)
    results = np.load(result_file, allow_pickle=True)
    # ind_confidences = results['ind_conf']
    # ood_confidences = results['ood_conf']
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']

    # visualize
    ind_uncertainties = np.array(ind_uncertainties)
    ind_uncertainties = (ind_uncertainties-np.min(ind_uncertainties)) / (np.max(ind_uncertainties) - np.min(ind_uncertainties)) # normalize
    ood_uncertainties = np.array(ood_uncertainties)
    ood_uncertainties = (ood_uncertainties-np.min(ood_uncertainties)) / (np.max(ood_uncertainties) - np.min(ood_uncertainties)) # normalize

    fig = plt.figure(figsize=(5,4))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    data_label = 'HMDB-51' if args.ood_data == 'HMDB' else 'MiT-v2'
    plt.hist([ind_uncertainties, ood_uncertainties], 50, 
            density=True, histtype='bar', color=['blue', 'red'], 
            label=['in-distribution (%s)'%(args.ind_data), 'out-of-distribution (%s)'%(data_label)])
    plt.legend(fontsize=fontsize-3)
    plt.text(0.6, 6, 'AUC = %.2lf'%(auc), fontsize=fontsize-3)
    plt.xlabel('%s uncertainty'%(uncertainty), fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0, 1.01)
    plt.ylim(0, 10.01)
    plt.tight_layout()

    result_dir = os.path.dirname(args.result_prefix)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # save the figure
    plt.savefig(os.path.join(args.result_prefix + '_distribution.png'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig(os.path.join(args.result_prefix + '_distribution.pdf'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)


def get_confidence(result_file, conf='softmax'):
    # only for SoftMax and OpenMax
    assert os.path.exists(result_file), 'result file not exists! %s'%(result_file)
    results = np.load(result_file, allow_pickle=True)
    if conf == 'softmax':
        ind_score = results['ind_softmax']  # (N1, C)
        ood_score = results['ood_softmax']  # (N2, C)
    else:
        ind_score = results['ind_openmax']  # (N1, C+1)
        ood_score = results['ood_openmax']  # (N2, C+1)
    ind_conf = np.max(ind_score, axis=1)
    ood_conf = np.max(ood_score, axis=1)
    return ind_conf, ood_conf


def plot_by_confidence(ind_confidence, ood_confidence, auc=80, fontsize=16):
    # visualize
    ind_conf = ind_confidence.copy()
    ind_conf = (ind_conf-np.min(ind_conf)) / (np.max(ind_conf) - np.min(ind_conf) + 1e-6) # normalize
    ood_conf = ood_confidence.copy()
    ood_conf = (ood_conf-np.min(ood_conf)) / (np.max(ood_conf) - np.min(ood_conf) + 1e-6) # normalize

    fig = plt.figure(figsize=(5,4))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    data_label = 'HMDB-51' if args.ood_data == 'HMDB' else 'MiT-v2'
    plt.hist([ind_conf, ood_conf], 50, 
            density=True, histtype='bar', color=['blue', 'red'], 
            label=['in-distribution (%s)'%(args.ind_data), 'out-of-distribution (%s)'%(data_label)])
    plt.legend(fontsize=fontsize-3)
    plt.text(0.6, 6, 'AUC = %.2lf'%(auc), fontsize=fontsize-3)
    plt.xlabel('Confidence', fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0, 1.01)
    plt.ylim(0, 10.01)
    plt.tight_layout()

    result_dir = os.path.dirname(args.result_prefix)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # save the figure
    plt.savefig(os.path.join(args.result_prefix + '_distribution.png'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig(os.path.join(args.result_prefix + '_distribution.pdf'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)

def main_i3d():
    # common settings
    fontsize = 18 if args.ood_data == 'HMDB' else 20

    # SoftMax
    result_file = 'i3d/results_baselines/openmax/I3D_OpenMax_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'i3d/results_baselines/softmax/I3D_SoftMax_Conf_%s'%(args.ood_data)
    auc = 75.68 if args.ood_data == 'HMDB' else 79.94
    ind_conf, ood_conf = get_confidence(result_file, conf='softmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # OpenMax
    result_file = 'i3d/results_baselines/openmax/I3D_OpenMax_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'i3d/results_baselines/openmax/I3D_OpenMax_Conf_%s'%(args.ood_data)
    auc = 74.34 if args.ood_data == 'HMDB' else 77.76
    ind_conf, ood_conf = get_confidence(result_file, conf='openmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # RPL
    result_file = 'i3d/results_baselines/rpl/I3D_RPL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'i3d/results_baselines/rpl/I3D_RPL_Conf_%s'%(args.ood_data)
    auc = 75.20 if args.ood_data == 'HMDB' else 79.16
    ind_conf, ood_conf = get_confidence(result_file, conf='softmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # MC Dropout
    result_file = 'i3d/results/I3D_DNN_BALD_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'i3d/results/I3D_DNN_BALD_%s'%(args.ood_data)
    auc = 75.07 if args.ood_data == 'HMDB' else 79.14
    plot_by_uncertainty(result_file, uncertainty='BALD', auc=auc, fontsize=fontsize)

    # BNN SVI
    result_file = 'i3d/results/I3D_BNN_BALD_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'i3d/results/I3D_BNN_BALD_%s'%(args.ood_data)
    auc = 74.66 if args.ood_data == 'HMDB' else 79.50
    plot_by_uncertainty(result_file, uncertainty='BALD', auc=auc, fontsize=fontsize)

    # DRIVE (vanilla)
    result_file = 'i3d/results/I3D_EDLNoKL_EDL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'i3d/results/I3D_EDLNoKL_EDL_%s'%(args.ood_data)
    auc = 76.41 if args.ood_data == 'HMDB' else 81.43
    plot_by_uncertainty(result_file, uncertainty='EDL', auc=auc, fontsize=fontsize)

    # DRIVE (full)
    result_file = 'i3d/results/I3D_EDLNoKLAvUCCED_EDL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'i3d/results/I3D_EDLNoKLAvUCCED_EDL_%s'%(args.ood_data)
    auc = 77.08 if args.ood_data == 'HMDB' else 81.54
    plot_by_uncertainty(result_file, uncertainty='EDL', auc=auc, fontsize=fontsize)


def main_tsm():
    # common settings
    fontsize = 18 if args.ood_data == 'HMDB' else 20

    # SoftMax
    result_file = 'tsm/results_baselines/openmax/TSM_OpenMax_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tsm/results_baselines/softmax/TSM_SoftMax_Conf_%s'%(args.ood_data)
    auc = 77.99 if args.ood_data == 'HMDB' else 82.38
    ind_conf, ood_conf = get_confidence(result_file, conf='softmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # OpenMax
    result_file = 'tsm/results_baselines/openmax/TSM_OpenMax_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tsm/results_baselines/openmax/TSM_OpenMax_Conf_%s'%(args.ood_data)
    auc = 77.07 if args.ood_data == 'HMDB' else 83.05
    ind_conf, ood_conf = get_confidence(result_file, conf='openmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # RPL
    result_file = 'tsm/results_baselines/rpl/TSM_RPL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tsm/results_baselines/rpl/TSM_RPL_Conf_%s'%(args.ood_data)
    auc = 73.62 if args.ood_data == 'HMDB' else 77.28
    ind_conf, ood_conf = get_confidence(result_file, conf='softmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # MC Dropout
    result_file = 'tsm/results/TSM_DNN_BALD_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tsm/results/TSM_DNN_BALD_%s'%(args.ood_data)
    auc = 73.85 if args.ood_data == 'HMDB' else 78.35
    plot_by_uncertainty(result_file, uncertainty='BALD', auc=auc, fontsize=fontsize)

    # BNN SVI
    result_file = 'tsm/results/TSM_BNN_BALD_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tsm/results/TSM_BNN_BALD_%s'%(args.ood_data)
    auc = 73.42 if args.ood_data == 'HMDB' else 77.39
    plot_by_uncertainty(result_file, uncertainty='BALD', auc=auc, fontsize=fontsize)

    # DRIVE (full)
    result_file = 'tsm/results/TSM_EDLNoKLAvUCDebias_EDL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tsm/results/TSM_EDLNoKLAvUCDebias_EDL_%s'%(args.ood_data)
    auc = 78.65 if args.ood_data == 'HMDB' else 83.92
    plot_by_uncertainty(result_file, uncertainty='EDL', auc=auc, fontsize=fontsize)


def main_slowfast():
    # common settings
    fontsize = 18 if args.ood_data == 'HMDB' else 20

    # SoftMax
    result_file = 'slowfast/results_baselines/openmax/SlowFast_OpenMax_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'slowfast/results_baselines/softmax/SlowFast_SoftMax_Conf_%s'%(args.ood_data)
    auc = 79.16 if args.ood_data == 'HMDB' else 82.88
    ind_conf, ood_conf = get_confidence(result_file, conf='softmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # OpenMax
    result_file = 'slowfast/results_baselines/openmax/SlowFast_OpenMax_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'slowfast/results_baselines/openmax/SlowFast_OpenMax_Conf_%s'%(args.ood_data)
    auc = 78.76 if args.ood_data == 'HMDB' else 80.62
    ind_conf, ood_conf = get_confidence(result_file, conf='openmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # RPL
    result_file = 'slowfast/results_baselines/rpl/SlowFast_RPL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'slowfast/results_baselines/rpl/SlowFast_RPL_Conf_%s'%(args.ood_data)
    auc = 74.23 if args.ood_data == 'HMDB' else 77.42
    ind_conf, ood_conf = get_confidence(result_file, conf='softmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # MC Dropout
    result_file = 'slowfast/results/SlowFast_DNN_BALD_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'slowfast/results/SlowFast_DNN_BALD_%s'%(args.ood_data)
    auc = 75.41 if args.ood_data == 'HMDB' else 78.49
    plot_by_uncertainty(result_file, uncertainty='BALD', auc=auc, fontsize=fontsize)

    # BNN SVI
    result_file = 'slowfast/results/SlowFast_BNN_BALD_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'slowfast/results/SlowFast_BNN_BALD_%s'%(args.ood_data)
    auc = 74.78 if args.ood_data == 'HMDB' else 77.39
    plot_by_uncertainty(result_file, uncertainty='BALD', auc=auc, fontsize=fontsize)

    # DRIVE (full)
    result_file = 'slowfast/results/SlowFast_EDLNoKLAvUCDebias_EDL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'slowfast/results/SlowFast_EDLNoKLAvUCDebias_EDL_%s'%(args.ood_data)
    auc = 82.94 if args.ood_data == 'HMDB' else 86.99
    plot_by_uncertainty(result_file, uncertainty='EDL', auc=auc, fontsize=fontsize)


def main_tpn():
    # common settings
    fontsize = 18 if args.ood_data == 'HMDB' else 20

    # SoftMax
    result_file = 'tpn_slowonly/results_baselines/openmax/TPN_OpenMax_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tpn_slowonly/results_baselines/softmax/TPN_SoftMax_Conf_%s'%(args.ood_data)
    auc = 77.97 if args.ood_data == 'HMDB' else 81.35
    ind_conf, ood_conf = get_confidence(result_file, conf='softmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # OpenMax
    result_file = 'tpn_slowonly/results_baselines/openmax/TPN_OpenMax_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tpn_slowonly/results_baselines/openmax/TPN_OpenMax_Conf_%s'%(args.ood_data)
    auc = 74.12 if args.ood_data == 'HMDB' else 76.26
    ind_conf, ood_conf = get_confidence(result_file, conf='openmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # RPL
    result_file = 'tpn_slowonly/results_baselines/rpl/TPN_RPL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tpn_slowonly/results_baselines/rpl/TPN_RPL_Conf_%s'%(args.ood_data)
    auc = 75.32 if args.ood_data == 'HMDB' else 78.21
    ind_conf, ood_conf = get_confidence(result_file, conf='softmax')
    plot_by_confidence(ind_conf, ood_conf, auc=auc, fontsize=fontsize)

    # MC Dropout
    result_file = 'tpn_slowonly/results/TPN_SlowOnly_Dropout_BALD_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tpn_slowonly/results/TPN_SlowOnly_Dropout_BALD_%s'%(args.ood_data)
    auc = 74.13 if args.ood_data == 'HMDB' else 77.76
    plot_by_uncertainty(result_file, uncertainty='BALD', auc=auc, fontsize=fontsize)

    # BNN SVI
    result_file = 'tpn_slowonly/results/TPN_SlowOnly_BNN_BALD_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tpn_slowonly/results/TPN_SlowOnly_BNN_BALD_%s'%(args.ood_data)
    auc = 72.68 if args.ood_data == 'HMDB' else 75.32
    plot_by_uncertainty(result_file, uncertainty='BALD', auc=auc, fontsize=fontsize)

    # DRIVE (full)
    result_file = 'tpn_slowonly/results/TPN_SlowOnly_EDLlogNoKLAvUCDebias_EDL_%s_result.npz'%(args.ood_data)
    args.result_prefix = 'tpn_slowonly/results/TPN_SlowOnly_EDLlogNoKLAvUCDebias_EDL_%s'%(args.ood_data)
    auc = 79.23 if args.ood_data == 'HMDB' else 81.80
    plot_by_uncertainty(result_file, uncertainty='EDL', auc=auc, fontsize=fontsize)


if __name__ == '__main__':

    args = parse_args()

    if args.model == 'I3D':
        # draw results on I3D
        main_i3d()
    elif args.model == 'TSM':
        # draw results on TSM
        main_tsm()
    elif args.model == 'SlowFast':
        # draw results on SlowFast
        main_slowfast()
    elif args.model == 'TPN':
        # draw results on TPN
        main_tpn()
    else:
        raise NotImplementedError

    