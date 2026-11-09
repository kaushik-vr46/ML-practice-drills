"""Verify the classifier is genuinely learning something."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

from data import load, FEATURES, TARGET
from solution import build_model


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)

    model = build_model().fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)

    # A DummyClassifier always predicts the majority class. ALWAYS compare
    # against this — it is the bar your model must clear to have earned its
    # existence, and plenty of published models do not.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    dummy_acc = dummy.score(X_te, y_te)
    model_acc = model.score(X_te, y_te)

    print(f"Dummy (majority class) accuracy: {dummy_acc:.1%}")
    print(f"Your model accuracy:             {model_acc:.1%}")
    print(f"ROC-AUC:                         {auc:.3f}")

    cv = cross_val_score(build_model(), X, y, cv=5, scoring="roc_auc")
    print(f"5-fold CV AUC: {cv.mean():.3f} (+/- {cv.std():.3f})")

    assert auc > 0.6, "AUC too low — the model is barely ranking"
    assert model_acc >= dummy_acc, "Your model does not beat always guessing the majority"
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
