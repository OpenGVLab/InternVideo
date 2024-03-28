import os
import argparse
from matplotlib.pyplot import axis
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, roc_curve
from terminaltables import AsciiTable

def parse_args():
    '''Command instruction:
        source activate mmaction
        python experiments/compare_openness.py
    '''
    parser = argparse.ArgumentParser(description='Compare the performance of openness')
    # model config
    parser.add_argument('--base_model', default='i3d', help='the backbone model name')
    parser.add_argument('--ood_data', default='HMDB', help='the name of OOD dataset.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[-1,-1,-1,-1,-1,-1])
    parser.add_argument('--baseline_results', nargs='+', help='the testing results files.')
    args = parser.parse_args()
    return args


def eval_osr(y_true, y_pred):
    # open-set auc-roc (binary class)
    auroc = roc_auc_score(y_true, y_pred)

    # open-set auc-pr (binary class)
    # as an alternative, you may also use `ap = average_precision_score(labels, uncertains)`, which is approximate to aupr.
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)

    # open-set fpr@95 (binary class)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    operation_idx = np.abs(tpr - 0.95).argmin()
    fpr95 = fpr[operation_idx]  # FPR when TPR at 95%

    return auroc, aupr, fpr95


def parse_results(result_file, method='softmax'):
    # Softmax and OpenMax
    assert os.path.exists(result_file), "File not found! Run baseline_openmax.py first to get softmax testing results!\n%s"%(result_file)
    results = np.load(result_file, allow_pickle=True)
    # parse results
    ind_labels = results['ind_label']  # (N1,)
    ood_labels = results['ood_label']  # (N2,)
    if method == 'softmax':
        ind_softmax = results['ind_softmax']  # (N1, C)
        ood_softmax = results['ood_softmax']  # (N2, C)
        return ind_softmax, ood_softmax, ind_labels, ood_labels
    elif method == 'openmax':
        ind_openmax = results['ind_openmax']  # (N1, C+1)
        ood_openmax = results['ood_openmax']  # (N2, C+1)
        return ind_openmax, ood_openmax, ind_labels, ood_labels


def eval_confidence_methods(ind_probs, ood_probs, ind_labels, ood_labels, score='max_prob', ind_ncls=101, threshold=-1):
    # close-set accuracy (multi-class)
    ind_results = np.argmax(ind_probs, axis=1)
    ood_results = np.argmax(ood_probs, axis=1)
    acc = accuracy_score(ind_labels, ind_results)

    # open-set evaluation (binary class)
    if score == 'binary':
        preds = np.concatenate((ind_results, ood_results), axis=0)
        idx_pos = preds == ind_ncls
        idx_neg = preds != ind_ncls
        preds[idx_pos] = 1  # unknown class
        preds[idx_neg] = 0  # known class
    elif score == 'max_prob':
        ind_conf = np.max(ind_probs, axis=1)
        ood_conf = np.max(ood_probs, axis=1)
        confs = np.concatenate((ind_conf, ood_conf), axis=0)
        if threshold > 0:
            preds = np.concatenate((ind_results, ood_results), axis=0)
            preds[confs < threshold] = 1  # unknown class
            preds[confs >= threshold] = 0  # known class
        else:
            preds = 1 - confs
    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
    auroc, aupr, fpr95 = eval_osr(labels, preds)

    return acc, auroc, aupr, fpr95


def eval_uncertainty_methods(result_file, threshold=-1):
    assert os.path.exists(result_file), "File not found! Run ood_detection first!\n%s"%(result_file)
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

    # open-set evaluation (binary class)
    if threshold > 0:
        preds = np.concatenate((ind_results, ood_results), axis=0)
        uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
        preds[uncertains > threshold] = 1
        preds[uncertains <= threshold] = 0
    else:
        preds = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)

    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
    auroc, aupr, fpr95 = eval_osr(labels, preds)

    return acc, auroc, aupr, fpr95


def main():

    print(f'\nNew Evaluation Results (open-set data: {args.ood_data}, backbone: {args.base_model})')
    display_data = [["Methods", "AUROC (%)", "AUPR (%)", "FPR@95 (%)", "Closed-Set ACC (%)"], 
                    ["OpenMax"], ["MC Dropout"], ["BNN SVI"], ["SoftMax"], ["RPL"], ["DEAR (ours)"]]  # table heads and rows
    exp_dir = os.path.join('./experiments', args.base_model)

    # OpenMax
    result_path = os.path.join(exp_dir, args.baseline_results[0])
    ind_openmax, ood_openmax, ind_labels, ood_labels = parse_results(result_path, method='openmax')
    acc, auroc, aupr, fpr95 = eval_confidence_methods(ind_openmax, ood_openmax, ind_labels, ood_labels, score='binary')
    display_data[1].extend(["%.3f"%(auroc * 100), "%.3f"%(aupr * 100), "%.3f"%(fpr95 * 100), "%.3f"%(acc * 100)])
    
    # MC Dropout
    result_path = os.path.join(exp_dir, args.baseline_results[1])
    acc, auroc, aupr, fpr95 = eval_uncertainty_methods(result_path, threshold=args.thresholds[1])
    display_data[2].extend(["%.3f"%(auroc * 100), "%.3f"%(aupr * 100), "%.3f"%(fpr95 * 100), "%.3f"%(acc * 100)])

    # BNN SVI
    result_path = os.path.join(exp_dir, args.baseline_results[2])
    acc, auroc, aupr, fpr95 = eval_uncertainty_methods(result_path, threshold=args.thresholds[2])
    display_data[3].extend(["%.3f"%(auroc * 100), "%.3f"%(aupr * 100), "%.3f"%(fpr95 * 100), "%.3f"%(acc * 100)])

    # SoftMax
    result_path = os.path.join(exp_dir, args.baseline_results[3])
    ind_softmax, ood_softmax, ind_labels, ood_labels = parse_results(result_path, method='softmax')
    acc, auroc, aupr, fpr95 = eval_confidence_methods(ind_softmax, ood_softmax, ind_labels, ood_labels, threshold=args.thresholds[3])
    display_data[4].extend(["%.3f"%(auroc * 100), "%.3f"%(aupr * 100), "%.3f"%(fpr95 * 100), "%.3f"%(acc * 100)])

    # RPL
    result_path = os.path.join(exp_dir, args.baseline_results[4])
    ind_softmax, ood_softmax, ind_labels, ood_labels = parse_results(result_path, method='softmax')
    acc, auroc, aupr, fpr95 = eval_confidence_methods(ind_softmax, ood_softmax, ind_labels, ood_labels, threshold=args.thresholds[4])
    display_data[5].extend(["%.3f"%(auroc * 100), "%.3f"%(aupr * 100), "%.3f"%(fpr95 * 100), "%.3f"%(acc * 100)])

    # DEAR (ours)
    result_path = os.path.join(exp_dir, args.baseline_results[5])
    acc, auroc, aupr, fpr95 = eval_uncertainty_methods(result_path, threshold=args.thresholds[5])
    display_data[6].extend(["%.3f"%(auroc * 100), "%.3f"%(aupr * 100), "%.3f"%(fpr95 * 100), "%.3f"%(acc * 100)])


    table = AsciiTable(display_data)
    table.inner_footing_row_border = True
    table.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center', 4: 'center'}
    print(table.table)
    print("\n")

if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()