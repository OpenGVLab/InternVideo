import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
# from mmaction.core.evaluation import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm


def confusion_maxtrix(ind_labels, ind_results, ind_uncertainties,
                      ood_labels, ood_results, ood_uncertainties,
                      threshold, know_ood_labels=False, normalize=True):
    num_indcls = max(ind_labels) + 1
    num_oodcls = max(ood_labels) + 1
    confmat = np.zeros((num_indcls + num_oodcls, num_indcls + num_oodcls), dtype=np.float32)
    for rlabel, plabel, uncertain in zip(ind_labels, ind_results, ind_uncertainties):
        if uncertain > threshold:
            # known --> unknown (bottom-left)
            confmat[num_indcls:num_indcls+num_oodcls, rlabel] += 1.0 * num_oodcls
        else:
            # known --> known (top-left)
            confmat[plabel, rlabel] += 1.0 * num_indcls
    if know_ood_labels:
        for rlable, plabel, uncertain in zip(ood_labels, ood_results, ood_uncertainties):
            if uncertain > threshold:
                # unknown --> unknown (bottom-right)
                confmat[num_indcls:num_indcls+num_oodcls, num_indcls+rlable] += 1.0
            else:
                # unknown --> known (top-right)
                confmat[plabel, num_indcls+rlable] += 1.0 * num_oodcls
    else:
        for plabel, uncertain in zip(ood_results, ood_uncertainties):
            if uncertain > threshold:
                # unknown --> unknown (bottom-right)
                confmat[num_indcls:num_indcls+num_oodcls, num_indcls:num_indcls+num_oodcls] += 1.0
            else:
                # unknown --> known (top-right)
                confmat[plabel, num_indcls:num_indcls+num_oodcls] += 1 * num_oodcls
    if normalize:
        minval = np.min(confmat[np.nonzero(confmat)])
        maxval = np.max(confmat)
        confmat = (confmat - minval) / (maxval - minval + 1e-6)
        # confmat = np.nan_to_num(confmat)
    return confmat



def confusion_maxtrix_top(ind_labels, ind_results, ind_uncertainties,
                      ood_labels, ood_results, ood_uncertainties,
                      threshold, normalize=True):
    num_indcls = max(ind_labels) + 1
    num_oodcls = max(ood_labels) + 1
    confmat = np.ones((num_indcls, num_indcls + num_oodcls), dtype=np.float32)  # (K, K+C) white
    # process in-distribution results
    for rlabel, plabel, uncertain in zip(ind_labels, ind_results, ind_uncertainties):
        if uncertain < threshold:
            # known --> known (top-left)
            confmat[plabel, rlabel] -= 1.0 / len(ind_results)  # make it darker
    # process out-of-distribution results
    for rlable, plabel, uncertain in zip(ood_labels, ood_results, ood_uncertainties):
        if uncertain < threshold:
            # unknown --> known (top-right)
            confmat[plabel, num_indcls+rlable] -= 1.0 / len(ood_results)  # make it darker
    if normalize:
        minval = np.min(confmat)
        maxval = np.max(confmat)
        confmat = (confmat - minval) / (maxval - minval + 1e-6)  # normalize to [0, 1]
    return confmat



def plot_confmat(confmat, know_ood_labels=False):
    fig = plt.figure(figsize=(4,4))
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 20
    ax = plt.gca()
    confmat_vis = confmat.copy()
    im = ax.imshow(confmat_vis, cmap='hot')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    # cbar.locator = ticker.MaxNLocator(nbins=5)
    # # barticks = np.linspace(np.min(confmat) * 1000, np.max(confmat) * 1000, 5).tolist()
    # # cbar.set_ticks(barticks)
    # cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_ticks([])
    cbar.update_ticks()
    plt.tight_layout()
    save_file = args.save_file[:-4] + '_knownOOD.png' if know_ood_labels else args.save_file
    plt.savefig(save_file, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig(save_file[:-4] + '.pdf', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.close()

def get_topk_2d(arr, topk=5):
    col_indices = np.argmax(arr, axis=1)  # column indices
    vals = [arr[r, c] for r, c in enumerate(col_indices)]
    row_indices = np.argsort(vals)[::-1]  # decreasing sort
    result_inds = np.zeros((topk, 2), dtype=np.int32)
    result_vals = []
    for k in range(topk):
        result_inds[k, 0] = row_indices[k]
        result_inds[k, 1] = col_indices[row_indices[k]]
        result_vals.append(vals[row_indices[k]])
    return result_inds, result_vals

def read_classnames(list_file):
    names = []
    with open(list_file, 'r') as f:
        for line in f.readlines():
            names.append(line.strip().split(' ')[-1])
    return names

def plot_top_confmat(confmat):
    ind_mappings = 'data/ucf101/annotations/classInd.txt'
    ind_cls_names = read_classnames(ind_mappings)
    ood_mappings = 'data/hmdb51/annotations/classInd.txt'
    ood_cls_names = read_classnames(ood_mappings)

    fig = plt.figure(figsize=(8,4))
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    ax = plt.gca()
    confmat_vis = confmat.copy()
    im = ax.imshow(confmat_vis, cmap='hot')
    plt.axvline(num_indcls-1, 0, num_indcls-1, linestyle='--')
    # find the top-K mis-classification for unknown
    result_inds, result_vals = get_topk_2d(1 - confmat_vis[:, num_indcls+1:])
    ood_ids = np.argmin(confmat_vis[:, num_indcls+1:], axis=1)
    text_offset = [[-25, 1], [-25, -3], [-61, 1], [-28, 2], [-32, 1]]
    for i, (r, c) in enumerate(result_inds):
        hd = plt.Circle((c + num_indcls, r), 5, fill=False)
        ax.set_aspect(1) 
        ax.add_artist(hd )
        off_c, off_r = 6, 1
        if i == 1:
            off_c, off_r = -4, -6
        plt.text(c + num_indcls + off_c, r + off_r, ood_cls_names[c], color='blue', fontsize=fontsize)
        plt.plot([num_indcls, num_indcls + c], [r, r], 'r--')
        plt.text(num_indcls + text_offset[i][0], r+text_offset[i][1], ind_cls_names[r], color='red', fontsize=fontsize)
    plt.ylabel('Predicted Classes', fontsize=fontsize)
    plt.xlabel('UCF-101 (known) + HMDB-51 (unknown)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(args.save_file[:-4] + '_top.png', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig(args.save_file[:-4] + '_top.pdf', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--ood_result', help='the result file of ood detection')
    parser.add_argument('--uncertain_thresh', type=float, default=0.0001, help='the threshold value for prediction')
    parser.add_argument('--top_part', action='store_true', help='Whether to show the top part of confmat separately.')
    parser.add_argument('--save_file', help='the image file path of generated confusion matrix')
    args = parser.parse_args()

    results = np.load(args.ood_result, allow_pickle=True)
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']

    # result path
    result_path = os.path.dirname(args.save_file)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # OOD classes are unknown
    confmat1 = confusion_maxtrix(ind_labels, ind_results, ind_uncertainties,
                                ood_labels, ood_results, ood_uncertainties,
                                args.uncertain_thresh, know_ood_labels=False)
    plot_confmat(confmat1, know_ood_labels=False)

    num_indcls = max(ind_labels) + 1
    num_oodcls = max(ood_labels) + 1
    UKC_value = np.mean(confmat1[:num_indcls, num_indcls:])  # unknown --> known (top-right)
    UUC_value = np.mean(confmat1[num_indcls:, num_indcls:])  # unknown --> unknown  (bottom-right)
    KUC_value = np.mean(confmat1[num_indcls:, :num_indcls])  # known --> unknown (bottom-left)
    KKC_value = np.mean(np.diag(confmat1[:num_indcls, :num_indcls]))  # known --> known (top-left)
    print("The average UUC=: %.6lf, UKC=%.6lf, KUC=%.6lf, KKC=%.6lf"%(UUC_value, UKC_value, KUC_value, KKC_value))



    # # OOD classes are known
    # confmat2 = confusion_maxtrix(ind_labels, ind_results, ind_uncertainties,
    #                             ood_labels, ood_results, ood_uncertainties,
    #                             args.uncertain_thresh, know_ood_labels=True)
    # plot_confmat(confmat2, know_ood_labels=True)

    # # save the confusion matrix for further analysis
    # np.savez(args.save_file[:-4], confmat_unknown_ood=confmat1, confmat_known_ood=confmat2)
    # votes_ind = np.sum(confmat1[:101, 101:], axis=1)
    # print("Top-10 false positive IND classes: ", np.argsort(votes_ind)[-10:])

    # votes_ood = np.sum(confmat1[101:, :101], axis=1)
    # print("Top-10 false negative IND classes: ", np.argsort(votes_ood)[-10:])

    if args.top_part:
        top_confmat = confusion_maxtrix_top(ind_labels, ind_results, ind_uncertainties,
                                            ood_labels, ood_results, ood_uncertainties,
                                            args.uncertain_thresh)
        plot_top_confmat(top_confmat)