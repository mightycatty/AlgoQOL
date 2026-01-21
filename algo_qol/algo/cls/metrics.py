from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve


def cls_report(gt, pred):
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    fpr, _, thresholds_roc = roc_curve(gt, pred, drop_intermediate=False)
    fpr = fpr[::-1]
    lines = []
    lines_dict = {}
    for fpr_item, pre_item, rec_item, thre in zip(fpr, precision, recall, thresholds):
        rec_item = round(rec_item, 3)
        line = f'fpr:{fpr_item}\tprecision:{pre_item}\trecall:{rec_item}\tthre:{thre}\t'
        # line = [thre, fpr_item, rec_item, pre_item]
        lines.append(line)
        if len(list(set(recall))) <= 2:
            lines_dict[fpr_item] = line
        else:
            lines_dict[rec_item] = line
    ap = average_precision_score(gt, pred)
    lines.append(f'ap:{ap}')
    return lines_dict.values()
