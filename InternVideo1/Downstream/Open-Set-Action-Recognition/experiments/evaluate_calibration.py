import argparse, os
import numpy as np
import matplotlib.pyplot as plt


def eval_calibration(predictions, confidences, labels, M=15):
    """
    M: number of bins for confidence scores
    """
    num_Bm = np.zeros((M,), dtype=np.int32)
    accs = np.zeros((M,), dtype=np.float32)
    confs = np.zeros((M,), dtype=np.float32)
    for m in range(M):
        interval = [m / M, (m+1) / M]
        Bm = np.where((confidences > interval[0]) & (confidences <= interval[1]))[0]
        if len(Bm) > 0:
            acc_bin = np.sum(predictions[Bm] == labels[Bm]) / len(Bm)
            conf_bin = np.mean(confidences[Bm])
            # gather results
            num_Bm[m] = len(Bm)
            accs[m] = acc_bin
            confs[m] = conf_bin
    conf_intervals = np.arange(0, 1, 1/M)
    return accs, confs, num_Bm, conf_intervals

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def compute_eavuc(preds, labels, confs, uncertainties):
    eavuc = 0
    inds_accurate = np.where(preds == labels)[0]
    eavuc += -np.sum(confs[inds_accurate] * np.log(1 - uncertainties[inds_accurate]))
    inds_inaccurate = np.where(preds != labels)[0]
    eavuc += -np.sum((1 - confs[inds_inaccurate]) * np.log(uncertainties[inds_inaccurate]))
    return eavuc


def closedset_multicls(ind_results, ind_labels, ind_confidences, ind_uncertainties):
    ind_preds = ind_results.copy()
    accs, confs, num_Bm, conf_intervals = eval_calibration(ind_preds, 1-ind_uncertainties, ind_labels, M=args.M)
    # compute the EAvUC
    eavuc = compute_eavuc(ind_preds, ind_labels, ind_confidences, ind_uncertainties)
    # compute ECE
    ece = np.sum(np.abs(accs - confs) * num_Bm / np.sum(num_Bm))
    return ece, eavuc, conf_intervals, accs


def openset_multicls(ind_results, ood_results, ind_labels, ood_labels, ind_confidences, ood_confidences, ind_uncertainties, ood_uncertainties):
    ind_preds = ind_results.copy()
    ood_preds = ood_results.copy()
    ind_preds[ind_uncertainties > args.threshold] = args.ind_ncls
    ood_preds[ood_uncertainties > args.threshold] = args.ind_ncls
    preds = np.concatenate((ind_preds, ood_preds), axis=0)
    labels = np.concatenate((ind_labels, np.ones_like(ood_labels) * args.ind_ncls), axis=0)
    confs = np.concatenate((ind_confidences, ood_confidences), axis=0)
    unctns = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
    # compute the EAvUC
    eavuc = compute_eavuc(preds, labels, confs, unctns)
    # compute ECE
    accs, confs, num_Bm, conf_intervals = eval_calibration(preds, 1-unctns, labels, M=args.M)
    ece = np.sum(np.abs(accs - confs) * num_Bm / np.sum(num_Bm))
    return ece, eavuc, conf_intervals, accs


def openset_bincls(ind_results, ood_results, ind_labels, ood_labels, ind_confidences, ood_confidences, ind_uncertainties, ood_uncertainties):
    ind_preds = ind_results.copy()
    ood_preds = ood_results.copy()
    ind_preds[ind_uncertainties > args.threshold] = 1
    ind_preds[ind_uncertainties <= args.threshold] = 0
    ood_preds[ood_uncertainties > args.threshold] = 1
    ood_preds[ood_uncertainties < args.threshold] = 0
    preds = np.concatenate((ind_preds, ood_preds), axis=0)
    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)), axis=0)
    confs = np.concatenate((ind_confidences, ood_confidences), axis=0)
    unctns = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
    # compute the EAvUC
    eavuc = compute_eavuc(preds, labels, confs, unctns)
    # compute ECE
    accs, confs, num_Bm, conf_intervals = eval_calibration(preds, 1-unctns, labels, M=args.M)
    ece = np.sum(np.abs(accs - confs) * num_Bm / np.sum(num_Bm))
    return ece, eavuc, conf_intervals, accs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--ood_result', help='the result file of ood detection')
    parser.add_argument('--M', type=int, default=15, help='The number of bins')
    parser.add_argument('--ind_ncls', type=int, default=101, help='the number classes for in-distribution data')
    parser.add_argument('--threshold', type=float, help='the threshold to decide if it is an OOD')
    parser.add_argument('--save_prefix', help='the image file path of generated calibration figure')
    parser.add_argument('--draw_diagram', action='store_true', help='if to draw reliability diagram.')
    args = parser.parse_args()

    results = np.load(args.ood_result, allow_pickle=True)
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']
    if 'ind_conf' not in results:
        ind_confidences = 1 - ind_uncertainties
        ood_confidences = 1 - ood_uncertainties
    else:
        ind_confidences = results['ind_conf']
        ood_confidences = results['ood_conf']

    # result path
    result_path = os.path.dirname(args.save_prefix)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Closed Set: (K class)
    ece, eavuc, conf_intervals, accs = closedset_multicls(ind_results, ind_labels, ind_confidences, ind_uncertainties)
    print('The Closed Set (K class) ECE=%.3lf, EAvUC=%.3lf'%(ece, eavuc))

    # Open Set: K+1 class
    ece, eavuc, conf_intervals, accs = openset_multicls(ind_results, ood_results, ind_labels, ood_labels, ind_confidences, ood_confidences, ind_uncertainties, ood_uncertainties)
    print('The Open Set (K+1 class) ECE=%.3lf, EAvUC=%.3lf'%(ece, eavuc))

    # Open Set: 2 class
    ece, eavuc, conf_intervals, accs = openset_bincls(ind_results, ood_results, ind_labels, ood_labels, ind_confidences, ood_confidences, ind_uncertainties, ood_uncertainties)
    print('The Open Set (2-class) ECE=%.3lf, EAvUC=%.3lf'%(ece, eavuc))

    if args.draw_diagram:
        # plot the ECE figure
        fig, ax = plt.subplots(figsize=(4,4))
        plt.rcParams["font.family"] = "Arial"  # Times New Roman
        fontsize = 15
        plt.bar(conf_intervals, accs, width=1/args.M, linewidth=1, edgecolor='k', align='edge', label='Outputs')
        plt.bar(conf_intervals, np.maximum(0, conf_intervals - accs), bottom=accs, color='y', width=1/args.M, linewidth=1, edgecolor='k', align='edge', label='Gap')
        plt.text(0.1, 0.6, 'ECE=%.4f'%(ece), fontsize=fontsize)
        add_identity(ax, color='r', ls='--')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel('confidence', fontsize=fontsize)
        plt.ylabel('accuracy', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(args.save_prefix + '_ind.png')
        plt.savefig(args.save_prefix + '_ind.pdf')

