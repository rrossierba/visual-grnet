class REPORT:
    METRICS_DICT={
        'exact_accuracy': 'Exact Accuracy',
        'mean_label_accuracy': 'Mean Label Accuracy',
        'precision_macro': 'Precision (macro)',
        'recall_macro': 'Recall (macro)',
        'f1_macro': 'F1 (macro)',
        'precision_micro': 'Precision (micro)',
        'recall_micro': 'Recall (micro)',
        'f1_micro': 'F1 (micro)',
        'avg_true_positive': 'Average true positives',
        'avg_pred_positive': 'Average predicted positives',
        'auc_micro':'AUC (Micro)',
        'auc_macro':'AUC (Macro)',
        'pr_auc_macro':'AUC-PR (Macro)'
    }

class THRESHOLDS:
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]