"""Verify the forest works and that you understand the importance trap."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data import load, FEATURES, TARGET


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)

    rf = RandomForestClassifier(n_estimators=300, oob_score=True,
                                random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)

    auc_rf = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
    auc_tree = roc_auc_score(y_te, tree.predict_proba(X_te)[:, 1])

    print(f"Single tree AUC : {auc_tree:.3f}")
    print(f"Forest AUC      : {auc_rf:.3f}")
    print(f"Forest OOB acc  : {rf.oob_score_:.1%}")

    perm = permutation_importance(rf, X_te, y_te, n_repeats=15,
                                  random_state=0, scoring="roc_auc")
    noise_idx = FEATURES.index("listing_id")
    noise_perm = perm.importances_mean[noise_idx]
    noise_imp = rf.feature_importances_[noise_idx]

    print(f"\nNoise column 'listing_id':")
    print(f"  impurity importance    {noise_imp:.4f}   (should be ~0, but is not)")
    print(f"  permutation importance {noise_perm:.4f}   (correctly near 0)")

    assert auc_rf > auc_tree, "Forest should beat a single tree"
    assert noise_perm < 0.03, "Permutation importance flagged pure noise as useful"
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
