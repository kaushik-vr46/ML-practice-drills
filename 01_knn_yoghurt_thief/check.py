"""Checks that you built the pipeline correctly, not just that it runs."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut

from data import load, tonight, FEATURES, TARGET
from solution import build_model


def main():
    df = load()
    model = build_model(3)

    assert isinstance(model, Pipeline), "build_model should return a Pipeline"
    steps = dict(model.named_steps)
    assert any(isinstance(s, StandardScaler) for s in steps.values()), \
        "Pipeline is missing a StandardScaler"
    assert any(isinstance(s, KNeighborsClassifier) for s in steps.values()), \
        "Pipeline is missing a KNeighborsClassifier"
    print("OK   Pipeline structure correct")

    model.fit(df[FEATURES], df[TARGET])
    pred = model.predict(tonight())[0]
    print(f"OK   Predicts: {pred}")

    acc = cross_val_score(build_model(3), df[FEATURES], df[TARGET],
                          cv=LeaveOneOut()).mean()
    print(f"OK   Leave-one-out accuracy: {acc:.1%}")
    assert acc > 0.7, "Accuracy is suspiciously low — check your pipeline order"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
