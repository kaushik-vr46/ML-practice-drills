import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from data import load, FEATURES, TARGET


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)

    rf = RandomForestClassifier(n_estimators=300, oob_score=True,
                                random_state=0, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    proba = rf.predict_proba(X_te)[:, 1]
    print(f"Test accuracy : {rf.score(X_te, y_te):.1%}")
    print(f"OOB accuracy  : {rf.oob_score_:.1%}   <- no held-out data used")
    print(f"ROC-AUC       : {roc_auc_score(y_te, proba):.3f}\n")

    print(classification_report(y_te, rf.predict(X_te),
                                target_names=["fine", "scam"]))

    impurity = pd.Series(rf.feature_importances_, index=FEATURES)

    perm = permutation_importance(rf, X_te, y_te, n_repeats=20,
                                  random_state=0, scoring="roc_auc", n_jobs=-1)
    permutation = pd.Series(perm.importances_mean, index=FEATURES)

    comparison = pd.DataFrame({
        "impurity": impurity, "permutation": permutation,
    }).sort_values("permutation", ascending=False)

    print("Two ways of measuring importance:")
    print(comparison.round(4).to_string())
    print(f"\n  listing_id is RANDOM NOISE. It should score zero.")
    print(f"  impurity gives it    {impurity['listing_id']:.4f}")
    print(f"  permutation gives it {permutation['listing_id']:.4f}")
    print("  Impurity importance is fooled because a high-cardinality column")
    print("  offers many split points, so it reduces impurity by luck.")
    print("  ALWAYS report permutation importance.\n")

    seller = pd.DataFrame([[14, 1, 62, 1, 2, 543210]], columns=FEATURES)
    p = rf.predict_proba(seller)[0, 1]
    print("14-day-old account, 1 review, 62% below market, off-platform, 2 photos:")
    print(f"  {p:.0%} probability of scam -> "
          f"{'do not send money' if p > 0.5 else 'probably fine'}\n")

    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [4, 6, 8, 12, None],
        "min_samples_leaf": randint(1, 20),
        "max_features": ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=0, n_jobs=-1),
        param_dist, n_iter=30, cv=5, scoring="roc_auc",
        random_state=0, n_jobs=-1)
    search.fit(X_tr, y_tr)

    print("RandomizedSearchCV (30 samples from the space):")
    print(f"  best CV AUC : {search.best_score_:.3f}")
    print(f"  test AUC    : {roc_auc_score(y_te, search.best_estimator_.predict_proba(X_te)[:,1]):.3f}")
    print(f"  params      : { {k: v for k, v in search.best_params_.items()} }\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    comparison.plot.barh(ax=ax[0])
    ax[0].set_title("Impurity vs permutation importance")
    ax[0].set_xlabel("importance"); ax[0].grid(alpha=.3, axis="x")

    sizes = [25, 50, 100, 200, 400]
    oobs = []
    for n in sizes:
        m = RandomForestClassifier(n_estimators=n, oob_score=True,
                                   random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        oobs.append(m.oob_score_)
    ax[1].plot(sizes, oobs, marker="o")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of trees"); ax[1].set_ylabel("OOB accuracy")
    ax[1].set_title("More trees never hurt — they just stop helping")
    ax[1].grid(alpha=.3)

    plt.tight_layout(); plt.savefig("plots/forest_analysis.png", dpi=110)
    print("  -> saved plots/forest_analysis.png")


if __name__ == "__main__":
    main()
