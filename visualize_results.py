
"""
Visualization module for training results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def visualize_all(results, test_data):


    fig = plt.figure(figsize=(18, 12))


    ax1 = plt.subplot(2, 3, 1)
    folds = list(range(1, len(results['cross_validation']['fold_accuracies']) + 1))
    fold_accs = results['cross_validation']['fold_accuracies']
    fold_val_accs = results['cross_validation']['fold_val_accuracies']

    ax1.bar([i - 0.2 for i in folds], fold_accs, width=0.4, label='Train Acc', alpha=0.8)
    ax1.bar([i + 0.2 for i in folds], fold_val_accs, width=0.4, label='Val Acc', alpha=0.8)
    ax1.axhline(y=0.90, color='r', linestyle='--', linewidth=2, label='90% Target')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Cross-Validation Accuracy per Fold')
    ax1.set_xticks(folds)
    ax1.legend()
    ax1.set_ylim([0.7, 1.0])
    ax1.grid(True, alpha=0.3)


    ax2 = plt.subplot(2, 3, 2)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    means = [
        results['cross_validation']['mean_accuracy'],
        results['cross_validation']['mean_precision'],
        results['cross_validation']['mean_recall'],
        results['cross_validation']['mean_f1'],
        results['cross_validation']['mean_roc_auc'],
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax2.barh(metrics, means, color=colors, alpha=0.8)
    ax2.set_xlabel('Score')
    ax2.set_title('Cross-Validation Metrics (Mean)')
    ax2.set_xlim([0, 1])
    ax2.axvline(x=0.90, color='r', linestyle='--', linewidth=2, alpha=0.5)


    for i, (bar, val) in enumerate(zip(bars, means)):
        ax2.text(val + 0.02, i, f'{val:.4f}', va='center')


    ax3 = plt.subplot(2, 3, 3)
    cm = np.array(results['test_results']['confusion_matrix'])


    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax3,
                xticklabels=['Healthy', 'Diseased'],
                yticklabels=['Healthy', 'Diseased'])
    ax3.set_title('Test Set Confusion Matrix (Normalized)')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')


    ax4 = plt.subplot(2, 3, 4)
    test_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    test_values = [
        results['test_results']['test_accuracy'],
        results['test_results']['test_precision'],
        results['test_results']['test_recall'],
        results['test_results']['test_f1'],
        results['test_results']['test_roc_auc'],
    ]
    colors_test = ['#27ae60', '#2980b9', '#c0392b', '#e67e22', '#8e44ad']
    bars = ax4.barh(test_metrics, test_values, color=colors_test, alpha=0.8)
    ax4.set_xlabel('Score')
    ax4.set_title(f"Test Set Metrics (Accuracy: {results['test_results']['test_accuracy']:.2%})")
    ax4.set_xlim([0, 1])
    ax4.axvline(x=0.90, color='r', linestyle='--', linewidth=2, alpha=0.5)


    for bar, val in zip(bars, test_values):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontweight='bold')


    ax5 = plt.subplot(2, 3, 5)
    test_labels, test_preds, test_probs = test_data
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)

    ax5.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC-AUC Curve (Test Set)')
    ax5.legend(loc="lower right")
    ax5.grid(True, alpha=0.3)


    ax6 = plt.subplot(2, 3, 6)
    test_healthy = results['test_results']['num_healthy_test']
    test_diseased = results['test_results']['num_diseased_test']

    sizes = [test_healthy, test_diseased]
    labels = [f'Healthy\n({test_healthy})', f'Diseased\n({test_diseased})']
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0.1)

    ax6.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 10})
    ax6.set_title('Test Set Class Distribution')

    plt.tight_layout()
    plt.savefig('training_results_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_results_visualization.png")
    plt.close()


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    folds = list(range(1, len(results['cross_validation']['fold_precisions']) + 1))


    axes[0, 0].plot(folds, results['cross_validation']['fold_precisions'],
                    'o-', linewidth=2, markersize=8, color='#3498db')
    axes[0, 0].axhline(y=0.90, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Precision by Fold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.7, 1.0])


    axes[0, 1].plot(folds, results['cross_validation']['fold_recalls'],
                    'o-', linewidth=2, markersize=8, color='#e74c3c')
    axes[0, 1].axhline(y=0.90, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Recall by Fold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.7, 1.0])


    axes[1, 0].plot(folds, results['cross_validation']['fold_f1_scores'],
                    'o-', linewidth=2, markersize=8, color='#f39c12')
    axes[1, 0].axhline(y=0.90, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('F1-Score by Fold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.7, 1.0])


    axes[1, 1].plot(folds, results['cross_validation']['fold_roc_aucs'],
                    'o-', linewidth=2, markersize=8, color='#9b59b6')
    axes[1, 1].axhline(y=0.90, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('ROC-AUC by Fold')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0.7, 1.0])

    plt.tight_layout()
    plt.savefig('cross_validation_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: cross_validation_metrics.png")
    plt.close()


    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    summary_data = [
        ['Metric', 'Cross-Validation', 'Test Set', 'Status'],
        ['Accuracy', f"{results['cross_validation']['mean_accuracy']:.4f}",
         f"{results['test_results']['test_accuracy']:.4f}",
         '✓ PASS' if results['test_results']['test_accuracy'] >= 0.90 else '✗ FAIL'],
        ['Precision', f"{results['cross_validation']['mean_precision']:.4f}",
         f"{results['test_results']['test_precision']:.4f}", ''],
        ['Recall', f"{results['cross_validation']['mean_recall']:.4f}",
         f"{results['test_results']['test_recall']:.4f}", ''],
        ['F1-Score', f"{results['cross_validation']['mean_f1']:.4f}",
         f"{results['test_results']['test_f1']:.4f}", ''],
        ['ROC-AUC', f"{results['cross_validation']['mean_roc_auc']:.4f}",
         f"{results['test_results']['test_roc_auc']:.4f}", ''],
    ]

    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)


    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')


    for i in range(1, len(summary_data)):
        if i == 1 and results['test_results']['test_accuracy'] >= 0.90:
            table[(i, 3)].set_facecolor('#2ecc71')
        elif i == 1:
            table[(i, 3)].set_facecolor('#e74c3c')

    plt.title('Performance Summary: Cross-Validation vs Test Set',
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig('performance_summary_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_summary_table.png")
    plt.close()

if __name__ == '__main__':

    with open('training_results.json', 'r') as f:
        results = json.load(f)

    print("Loaded training results")
