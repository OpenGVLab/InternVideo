import os
import numpy as np
import matplotlib.pyplot as plt


def plot_by_uncertainty(result_file, uncertainty='EDL', auc=80, fontsize=16, result_prefix=''):
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
    data_label = 'HMDB-51' if ood_data == 'HMDB' else 'MiT-v2'
    counts, bins, bars = plt.hist([ind_uncertainties, ood_uncertainties], 50, 
            density=True, histtype='bar', color=['blue', 'red'], 
            label=['in-distribution (%s)'%(ind_data), 'out-of-distribution (%s)'%(data_label)])
    plt.legend(fontsize=fontsize-3)
    plt.text(0.6, 6, 'AUC = %.2lf'%(auc), fontsize=fontsize-3)
    plt.xlabel('%s uncertainty'%(uncertainty), fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0, 1.01)
    plt.ylim(0, 10.01)
    plt.tight_layout()

    result_dir = os.path.dirname(result_prefix)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # save the figure
    plt.savefig(os.path.join(result_prefix + '_distribution.png'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig(os.path.join(result_prefix + '_distribution.pdf'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    return counts, bins, bars


if __name__ == '__main__':

    fontsize = 20
    ind_data = 'UCF-101'
    ood_data = 'MiT'
    # DRIVE (vanilla)
    result_file = 'i3d/results/I3D_EDLNoKL_EDL_%s_result.npz'%(ood_data)
    counts, bins, bars = plot_by_uncertainty(result_file, uncertainty='EDL', auc=81.43, fontsize=fontsize, result_prefix='temp_rebuttal/I3D_MiT_Vanilla')
    counts = counts[:, 3:]
    bins = bins[3:]
    # mode = np.argsort(counts[1, :])[:5]
    mode = np.argmax(counts[1, :])
    print('the most frequent bin:(' + str(bins[mode]) + ',' + str(bins[mode+1]) + ')')

    # DRIVE (full)
    result_file = 'i3d/results/I3D_EDLNoKLAvUCCED_EDL_%s_result.npz'%(ood_data)
    counts, bins, bars = plot_by_uncertainty(result_file, uncertainty='EDL', auc=81.54, fontsize=fontsize, result_prefix='temp_rebuttal/I3D_MiT_Full')
    counts = counts[:, 3:]
    bins = bins[3:]
    # mode = np.argsort(counts[1, :])[:5]
    mode = np.argmax(counts[1, :])
    print('the most frequent bin:(' + str(bins[mode]) + ',' + str(bins[mode+1]) + ')')