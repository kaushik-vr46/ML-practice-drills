import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load, FEATURES, TARGET


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]

    print(f"Class balance: {y.value_counts(normalize=True).round(2).to_dict()}")
    print("  (If one class dominates, accuracy becomes misleading.)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)

    model = build_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, pred,
                                target_names=["missed it", "made it"]))
    print("  precision = when it says yes, how often is it right")
    print("  recall    = of all the real yeses, how many did it catch")
    print("  f1        = harmonic mean of the two\n")

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:")
    print(f"                  predicted missed   predicted made")
    print(f"  actually missed  {cm[0,0]:^16} {cm[0,1]:^15}")
    print(f"  actually made    {cm[1,0]:^16} {cm[1,1]:^15}")
    print(f"\n  Top-right ({cm[0,1]}) = you took the bus and were late. The painful one.\n")

    auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC: {auc:.3f}")
    print("  Probability a random success is ranked above a random failure.")
    print("  0.5 = coin flip. Threshold-independent, unlike accuracy.\n")

    clf = model.named_steps["clf"]
    print("Drivers (on standardised features, positive = more likely to make it):")
    for name, w in sorted(zip(FEATURES, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:<18} {w:+.3f}")
    print()

    tonight = pd.DataFrame([[8, 1, 9, 1]], columns=FEATURES)
    p = model.predict_proba(tonight)[0, 1]
    print(f"8am, raining, 9 people waiting, weekday -> {p:.1%} chance\n")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    fpr, tpr, _ = roc_curve(y_test, proba)
    ax[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "r--", label="random")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC curve"); ax[0].legend(); ax[0].grid(alpha=.3)

    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax[1].plot(rec, prec)
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-recall"); ax[1].grid(alpha=.3)

    ConfusionMatrixDisplay(cm, display_labels=["missed", "made"]).plot(
        ax=ax[2], colorbar=False)
    ax[2].set_title("Confusion matrix")

    plt.tight_layout(); plt.savefig("plots/evaluation.png", dpi=110)
    print("  -> saved plots/evaluation.png\n")

    print("Sweeping the decision threshold:")
    print(f"  {'thresh':<9}{'accuracy':<11}{'bus taken':<12}{'late when taken'}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        p_ = (proba >= t).astype(int)
        taken = p_.sum()
        late = ((p_ == 1) & (y_test == 0)).sum()
        rate = f"{late / taken:.1%}" if taken else "  n/a"
        print(f"  {t:<9}{(p_ == y_test).mean():<11.1%}{taken:<12}{rate}")
    print("\n  sklearn defaults to 0.5. Nothing says 0.5 is right for YOU.")
    print("  If being late costs more than walking, raise it.")


if __name__ == "__main__":
    main()
