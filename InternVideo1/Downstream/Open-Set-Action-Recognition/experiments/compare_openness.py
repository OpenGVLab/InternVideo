import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

def parse_args():
    '''Command instruction:
        source activate mmaction
        python experiments/compare_openness.py --ind_ncls 101 --ood_ncls 51
    '''
    parser = argparse.ArgumentParser(description='Compare the performance of openness')
    # model config
    parser.add_argument('--base_model', default='i3d', help='the backbone model name')
    parser.add_argument('--baselines', nargs='+', default=['I3D_Dropout_BALD', 'I3D_BNN_BALD', 'I3D_EDLlog_EDL', 'I3D_EDLlogAvUC_EDL'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.000423, 0.000024, 0.495783, 0.495783])
    parser.add_argument('--styles', nargs='+', default=['-b', '-k', '-r', '-g', '-m'])
    parser.add_argument('--ind_ncls', type=int, default=101, help='the number of classes in known dataset')
    parser.add_argument('--ood_ncls', type=int, help='the number of classes in unknwon dataset')
    parser.add_argument('--ood_data', default='HMDB', help='the name of OOD dataset.')
    parser.add_argument('--num_rand', type=int, default=10, help='the number of random selection for ood classes')
    parser.add_argument('--result_png', default='F1_openness_compare_HMDB.png')
    args = parser.parse_args()
    return args


def main():

    result_path = os.path.join('./experiments', args.base_model, 'results')
    plt.figure(figsize=(8,5))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    for style, thresh, baseline in zip(args.styles, args.thresholds, args.baselines):
        result_file = os.path.join(result_path, baseline + '_%s'%(args.ood_data) + '_result.npz')
        assert os.path.exists(result_file), "File not found! Run ood_detection first!"
        # load the testing results
        results = np.load(result_file, allow_pickle=True)
        ind_uncertainties = results['ind_unctt']  # (N1,)
        ood_uncertainties = results['ood_unctt']  # (N2,)
        ind_results = results['ind_pred']  # (N1,)
        ood_results = results['ood_pred']  # (N2,)
        ind_labels = results['ind_label']
        ood_labels = results['ood_label']

        # close-set accuracy (multi-class)
        acc = accuracy_score(ind_labels, ind_results)
        # open-set auc-roc (binary class)
        preds = np.concatenate((ind_results, ood_results), axis=0)
        uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
        preds[uncertains > thresh] = 1
        preds[uncertains <= thresh] = 0
        labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
        aupr = roc_auc_score(labels, preds)
        print('Model: %s, ClosedSet Accuracy (multi-class): %.3lf, OpenSet AUC (bin-class): %.3lf'%(baseline, acc * 100, aupr * 100))
        
        # open set F1 score (multi-class)
        ind_results[ind_uncertainties > thresh] = args.ind_ncls  # falsely rejection
        macro_F1_list = [f1_score(ind_labels, ind_results, average='macro')]
        std_list = [0]
        openness_list = [0]
        for n in range(args.ood_ncls):
            ncls_novel = n + 1
            openness = (1 - np.sqrt((2 * args.ind_ncls) / (2 * args.ind_ncls + ncls_novel))) * 100
            openness_list.append(openness)
            # randoml select the subset of ood samples
            macro_F1_multi = np.zeros((args.num_rand), dtype=np.float32)
            for m in range(args.num_rand):
                cls_select = np.random.choice(args.ood_ncls, ncls_novel, replace=False) 
                ood_sub_results = np.concatenate([ood_results[ood_labels == clsid] for clsid in cls_select])
                ood_sub_uncertainties = np.concatenate([ood_uncertainties[ood_labels == clsid] for clsid in cls_select])
                ood_sub_results[ood_sub_uncertainties > thresh] = args.ind_ncls  # correctly rejection
                ood_sub_labels = np.ones_like(ood_sub_results) * args.ind_ncls
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
        plt.plot(openness_list, macro_F1_list * 100, style, linewidth=2)
        # plt.fill_between(openness_list, macro_F1_list - std_list, macro_F1_list + std_list, style)

        w_openness = np.array(openness_list) / 100.
        open_maF1_mean = np.sum(w_openness * macro_F1_list) / np.sum(w_openness)
        open_maF1_std = np.sum(w_openness * std_list) / np.sum(w_openness)
        print('Open macro-F1 score: %.3f, std=%.3lf'%(open_maF1_mean * 100, open_maF1_std * 100))

    plt.xlim(0, max(openness_list))
    plt.ylim(60, 80)
    plt.xlabel('Openness (%)', fontsize=fontsize)
    plt.ylabel('macro F1 (%)', fontsize=fontsize)
    plt.grid('on')
    # plt.legend(args.baselines)
    plt.legend(['MC Dropout BALD', 'BNN SVI BALD', 'DEAR (vanilla)', 'DEAR (alter)', 'DEAR (joint)'], loc='lower left', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    png_file = os.path.join(result_path, args.result_png)
    plt.savefig(png_file)
    plt.savefig(png_file[:-4] + '.pdf')
    print('Openness curve figure is saved in: %s'%(png_file))


if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()