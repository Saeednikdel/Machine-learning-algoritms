import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for true, pred in zip(y_true, y_pred):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        matrix[true_index][pred_index] += 1
    return matrix


def plot_matrix(cf, title):
    df_cm = pd.DataFrame(cf, index=['Not Survived', 'Survived'],  columns=[
                         'Not Survived', 'Survived'])
    plt.figure(figsize=(3, 3))
    plt.title(f'{title} CF-M')
    sns.heatmap(df_cm, annot=True, cmap='Blues', cbar=False, fmt=".1f")


def cllassification_report(y_true, y_pred):
    report = f"{'Label':<10}{'Precision':>10}{'Recall':>10}{'F1-score':>10}{'Support':>10}\n"
    report += "-" * 50 + "\n"

    labels = np.unique(np.concatenate((y_true, y_pred))).astype(int)
    for label in labels:
        cf = confusion_matrix((y_true == label), (y_pred == label))
        tp, fp, fn, tn = cf[0][0], cf[0][1], cf[1][0], cf[1][1]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        support = np.sum(y_true == label)
        report += f"{label:<10}{precision:>10.2f}{recall:>10.2f}{f1:>10.2f}{support:>10}\n"

    acc = accuracy(y_true, y_pred)
    support = y_true.shape[0]

    report += "-" * 50 + "\n"
    report += f"{'Accuracy':<10}{acc:>40.2f}\n"
    report += f"{'Support':<10}{support:>40}\n"
    return report


def plot_roc_curve(y_true, y_score, title):
    tpr_list = []
    fpt_list = []
    thresholds = np.linspace(1.1, 0, 10)

    for t in thresholds:
        y_pred = np.zeros(y_true.shape[0])
        y_pred[y_score >= t] = 1
        TP = y_pred[(y_pred == y_true) & (y_true == 1)].shape[0]
        TN = y_pred[(y_pred == y_true) & (y_true == 0)].shape[0]
        FN = y_pred[(y_pred != y_true) & (y_true == 1)].shape[0]
        FP = y_pred[(y_pred != y_true) & (y_true == 0)].shape[0]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        tpr_list.append(TPR)
        fpt_list.append(FPR)
        
    auc = np.trapz(tpr_list, fpt_list)
    plt.figure(figsize=(6, 5))

    plt.plot(fpt_list, tpr_list,   color='red',
             lw=2, label=f'ROC curve (area = {auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} ROC')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
