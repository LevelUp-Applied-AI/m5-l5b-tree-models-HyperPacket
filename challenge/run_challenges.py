import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import from the local copied module
import lab_trees

class VotingEnsemble:
    """Custom voting ensemble that averages predict_proba."""
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)
        return self

    def predict_proba(self, X):
        probas = []
        for clf in self.classifiers:
            prob = clf.predict_proba(X)
            # Ensure probability ordering matches [0, 1]
            if hasattr(clf, "classes_") and len(clf.classes_) > 1:
                if clf.classes_[0] == 1:
                    prob = prob[:, [1, 0]]
            probas.append(prob)
        return np.mean(probas, axis=0)

    def predict(self, X):
        # Majority voting via probability threshold at 0.5
        # This is equivalent to hard voting if using threshold on probabilities?
        # The prompt says: "predict(X) via majority voting across all classifiers"
        # Oh, majority voting is based on individual predictions!
        votes = []
        for clf in self.classifiers:
            votes.append(clf.predict(X))
        votes = np.array(votes) # shape: (n_classifiers, n_samples)
        # sum of votes (assuming 1 is positive class)
        return (np.sum(votes, axis=0) > (len(self.classifiers) / 2)).astype(int)

def main():
    print("--- Loading Data ---")
    # Need to pass relative path to data because we are running from root or challenge dir?
    # We will run from root as `python challenge/run_challenges.py`, so 'data/telecom_churn.csv' works.
    X_train, X_test, y_train, y_test = lab_trees.load_and_split("data/telecom_churn.csv")
    
    print("\n--- Training Balanced RF ---")
    rf_balanced = lab_trees.build_random_forest(X_train, y_train, class_weight="balanced")
    
    print("\n" + "="*40)
    print("TIER 1: Threshold Tuning")
    print("="*40)
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    precisions, recalls, f1s = [], [], []
    probs = rf_balanced.predict_proba(X_test)[:, 1]
    
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, label="Precision", marker='o', markersize=4)
    plt.plot(thresholds, recalls, label="Recall", marker='s', markersize=4)
    plt.plot(thresholds, f1s, label="F1", marker='^', markersize=4)
    plt.axhline(0.8, color='k', linestyle='--', alpha=0.5, label='80% Recall Target')
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs Threshold (Balanced RF)")
    plt.legend()
    plt.grid(True)
    plt.savefig("challenge/results/threshold_sweep.png", bbox_inches="tight")
    plt.close()
    
    best_f1_idx = np.argmax(f1s)
    best_f1_thresh = thresholds[best_f1_idx]
    
    # highest threshold that still has recall >= 0.8
    valid_recalls = [(t, r) for t, r in zip(thresholds, recalls) if r >= 0.8]
    t_80 = max(t for t, r in valid_recalls) if valid_recalls else None
    
    print(f"Max F1 threshold: {best_f1_thresh:.2f} (F1: {f1s[best_f1_idx]:.3f})")
    print(f"Threshold for >=80% recall: {t_80:.2f} (Recall: {recalls[list(thresholds).index(t_80)]:.3f})")
    print("Saved to challenge/results/threshold_sweep.png")

    print("\n" + "="*40)
    print("TIER 2: Permutation Importance")
    print("="*40)
    
    print("Computing permutation importance (may take a moment)...")
    result = permutation_importance(rf_balanced, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances = result.importances_mean
    mdi_importances = rf_balanced.feature_importances_
    
    feature_names = lab_trees.NUMERIC_FEATURES
    
    mdi_dict = dict(zip(feature_names, mdi_importances))
    perm_dict = dict(zip(feature_names, perm_importances))
    
    top_10_mdi = sorted(mdi_dict.keys(), key=lambda x: mdi_dict[x], reverse=True)[:10]
    
    mdi_vals = [mdi_dict[f] for f in top_10_mdi]
    perm_vals = [perm_dict[f] for f in top_10_mdi]
    
    # Normalize values for fair visual comparison (optional, but good since scales differ)
    # Actually, the instructions say "compare MDI and permutation importance", standard to just plot raw or normalized. 
    # MDI sums to 1. Permutation importance does not necessarily sum to 1.
    # Let's plot raw but on two axes or just side-by-side if they're roughly same magnitude.
    # We will just plot raw.
    x = np.arange(len(top_10_mdi))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mdi_vals, width, label='MDI (Gini Importance)')
    ax.bar(x + width/2, perm_vals, width, label='Permutation Importance')
    
    ax.set_ylabel('Importance Score')
    ax.set_title('MDI vs Permutation Importance (Top 10 by MDI)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_10_mdi, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    plt.savefig("challenge/results/permutation_vs_mdi.png")
    plt.close()
    
    print("Top 3 features by MDI:")
    for f in top_10_mdi[:3]: print(f"  {f}: {mdi_dict[f]:.4f}")
    
    top_10_perm = sorted(perm_dict.keys(), key=lambda x: perm_dict[x], reverse=True)[:10]
    print("\nTop 3 features by Permutation Importance:")
    for f in top_10_perm[:3]: print(f"  {f}: {perm_dict[f]:.4f}")
    
    print("\nSaved to challenge/results/permutation_vs_mdi.png")

    print("\n" + "="*40)
    print("TIER 3: Custom Voting Ensemble")
    print("="*40)
    
    lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    dt_balanced = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    
    ensemble = VotingEnsemble([lr_pipeline, dt_balanced, rf_balanced])
    
    print("Training Ensemble...")
    ensemble.fit(X_train, y_train)
    
    print("\nEvaluating Individual Models on Test Set:")
    for name, model in [("LR", lr_pipeline), ("DT_Bal", dt_balanced), ("RF_Bal", rf_balanced)]:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        print(f"--- {name} ---")
        print(f"F1 Score: {f1_score(y_test, preds):.3f}")
        print(f"PR-AUC:   {average_precision_score(y_test, proba):.3f}")
        
    print("\nEvaluating Ensemble on Test Set:")
    ens_preds = ensemble.predict(X_test)
    ens_proba = ensemble.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, ens_preds))
    print(f"Ensemble PR-AUC: {average_precision_score(y_test, ens_proba):.3f}")


if __name__ == "__main__":
    main()
