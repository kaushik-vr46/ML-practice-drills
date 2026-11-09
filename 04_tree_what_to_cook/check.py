"""Verify the tree learns the generating rules."""
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from data import load, FEATURES, TARGET, LABELS


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)

    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
    acc = tree.score(X_te, y_te)
    cv = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0),
                         X, y, cv=5).mean()

    print(f"Test accuracy : {acc:.1%}")
    print(f"5-fold CV     : {cv:.1%}")
    print(f"Baseline (most common class): {y.value_counts(normalize=True).max():.1%}")

    imp = dict(zip(FEATURES, tree.feature_importances_))
    print("\nTop features:", sorted(imp, key=imp.get, reverse=True)[:2])

    assert acc > 0.7, "Tree accuracy too low"
    assert imp["minutes_available"] > 0.05, "Tree ignored minutes_available"
    print("\nAll checks passed. The tree found the real structure.")


if __name__ == "__main__":
    main()
