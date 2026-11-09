"""Verify the model recovers the known truth."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from data import load, FEATURES, TARGET, TRUE_COEFFS, TRUE_INTERCEPT


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]
    model = LinearRegression().fit(X, y)

    print("Recovered coefficients:")
    ok = True
    for name, coef in zip(FEATURES, model.coef_):
        true = TRUE_COEFFS[name]
        close = abs(coef - true) < max(0.5, abs(true) * 0.15)
        ok &= close
        print(f"  {'OK  ' if close else 'FAIL'} {name:<20} {coef:6.2f} vs {true:6.2f}")

    print(f"  intercept {model.intercept_:.2f} vs {TRUE_INTERCEPT:.2f}")

    r2 = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean()
    print(f"\n5-fold CV R-squared: {r2:.3f}")

    assert ok, "Coefficients do not match the generating process"
    assert r2 > 0.85, "R-squared too low"
    print("\nAll checks passed. The model recovered the hidden relationship.")


if __name__ == "__main__":
    main()
