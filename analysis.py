import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    CLASS_NAMES,
    AUGMENTATION_TYPES,
    RANDOM_STATE,
    TEST_SIZE,
    RESULTS_DIR,
    CONFUSION_DIR,
    ROBUSTNESS_DIR,
    load_original_dataset,
    load_augmented_dataset,
    ensure_dirs,
    get_project_root
)
from classification import train_and_evaluate_all, evaluate_classifier


def plot_confusion_matrix(cm, classifier_name, condition, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{classifier_name} - {condition}')
    plt.tight_layout()

    fname = f"cm_{classifier_name}_{condition}.png".replace(' ', '_')
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def plot_robustness_curves(accuracy_table, save_dir):
    conditions = ['original'] + AUGMENTATION_TYPES
    clf_names = list(accuracy_table.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    for clf in clf_names:
        accs = [accuracy_table[clf].get(c, 0) for c in conditions]
        ax.plot(conditions, accs, marker='o', label=clf)

    ax.set_xlabel('Condition')
    ax.set_ylabel('Accuracy')
    ax.set_title('Robustness: Accuracy Under Each Augmentation')
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'robustness_curves.png'), dpi=150)
    plt.close(fig)


def plot_accuracy_drop(accuracy_table, save_dir):
    conditions = AUGMENTATION_TYPES
    clf_names = list(accuracy_table.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))
    width = 0.8 / max(len(clf_names), 1)

    for i, clf in enumerate(clf_names):
        baseline = accuracy_table[clf]['original']
        drops = [baseline - accuracy_table[clf].get(c, 0) for c in conditions]
        ax.bar(x + i * width, drops, width, label=clf)

    ax.set_xlabel('Augmentation Type')
    ax.set_ylabel('Accuracy Drop from Baseline')
    ax.set_title('Accuracy Degradation by Augmentation')
    ax.set_xticks(x + width * (len(clf_names) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=30, ha='right')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'accuracy_drop.png'), dpi=150)
    plt.close(fig)


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return 0.0, 1.0

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(stat, df=1)
    return stat, p_value


def run_statistical_tests(results_dict, y_test, save_dir):
    clf_names = list(results_dict.keys())
    n = len(clf_names)
    p_matrix = np.ones((n, n))

    print("\n=== Statistical Significance (McNemar's Test) ===")
    for i in range(n):
        for j in range(i + 1, n):
            y_pred_i = results_dict[clf_names[i]]['y_pred']
            y_pred_j = results_dict[clf_names[j]]['y_pred']
            stat, p = mcnemar_test(y_test, y_pred_i, y_pred_j)
            p_matrix[i, j] = p
            p_matrix[j, i] = p
            sig = "YES" if p < 0.05 else "no"
            print(
                f"  {clf_names[i]} vs {clf_names[j]}: "
                f"chi2={stat:.3f}, p={p:.4f} (significant={sig})"
            )

    df = pd.DataFrame(p_matrix, index=clf_names, columns=clf_names)
    df.to_csv(os.path.join(save_dir, 'mcnemar_pvalues.csv'))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax)
    ax.set_title("McNemar's Test p-values (< 0.05 = significant)")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'mcnemar_heatmap.png'), dpi=150)
    plt.close(fig)


def run_full_analysis(repo_root='.'):
    project_root = get_project_root(repo_root)

    ensure_dirs(RESULTS_DIR, CONFUSION_DIR, ROBUSTNESS_DIR)

    print("\n" + "=" * 60)
    print("STEP 1: Loading original dataset")
    print("=" * 60)
    X, y, _ = load_original_dataset(repo_root=project_root)

    print("\nSplitting dataset (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("STEP 2: Training classifiers on original data")
    print("=" * 60)
    models, baseline_results = train_and_evaluate_all(
        X_train_s, y_train, X_test_s, y_test
    )

    accuracy_table = {
        clf: {'original': res['accuracy']}
        for clf, res in baseline_results.items()
    }

    all_metrics = []
    for clf, res in baseline_results.items():
        all_metrics.append({
            'classifier': clf,
            'condition': 'original',
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
        })
        plot_confusion_matrix(res['confusion_matrix'], clf, 'original', CONFUSION_DIR)

    print("\n" + "=" * 60)
    print("STEP 3: Robustness testing (augmented data)")
    print("=" * 60)

    for aug_type in AUGMENTATION_TYPES:
        print(f"\n--- Evaluating on {aug_type} ---")
        X_aug, y_aug = load_augmented_dataset(aug_type, repo_root=project_root)

        if len(X_aug) == 0:
            print(f"  No augmented images found for {aug_type}. Skipping.")
            continue

        X_aug_s = scaler.transform(X_aug)

        for clf_name, model in models.items():
            res = evaluate_classifier(model, X_aug_s, y_aug)
            accuracy_table[clf_name][aug_type] = res['accuracy']
            all_metrics.append({
                'classifier': clf_name,
                'condition': aug_type,
                'accuracy': res['accuracy'],
                'precision': res['precision'],
                'recall': res['recall'],
                'f1': res['f1'],
            })
            plot_confusion_matrix(res['confusion_matrix'], clf_name, aug_type, CONFUSION_DIR)
            print(f"  {clf_name}: accuracy={res['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("STEP 4: Saving metrics & plots")
    print("=" * 60)

    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(RESULTS_DIR, 'performance_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Metrics saved to {csv_path}")

    plot_robustness_curves(accuracy_table, ROBUSTNESS_DIR)
    plot_accuracy_drop(accuracy_table, ROBUSTNESS_DIR)
    print(f"  Robustness plots saved to {ROBUSTNESS_DIR}/")

    run_statistical_tests(baseline_results, y_test, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\nBaseline accuracy (original test set):")
    for clf, res in baseline_results.items():
        print(f"  {clf:20s}: {res['accuracy']:.4f}")

    print("\nAccuracy on augmented sets:")
    for aug in AUGMENTATION_TYPES:
        print(f"\n  {aug}:")
        for clf in models:
            acc = accuracy_table[clf].get(aug, 'N/A')
            if isinstance(acc, float):
                print(f"    {clf:20s}: {acc:.4f}")
            else:
                print(f"    {clf:20s}: {acc}")

    print(f"\nAll results saved under {RESULTS_DIR}/")
    print("Done.")


if __name__ == '__main__':
    user_repo_root = '.'
    if len(sys.argv) > 1:
        user_repo_root = sys.argv[1]
    run_full_analysis(repo_root=user_repo_root)