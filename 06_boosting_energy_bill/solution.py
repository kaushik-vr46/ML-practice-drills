import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import joblib
import lightgbm as lgb
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data import load, FEATURES, TARGET


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=0)

    print(f"train {len(X_train)}  val {len(X_val)}  test {len(X_test)}\n")

    models = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, random_state=0, verbose=-1),
    }

    print(f"{'model':<24}{'RMSE':<10}{'MAE':<10}{'R2':<9}{'fit (ms)'}")
    results = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        dt = (time.perf_counter() - t0) * 1000
        pred = m.predict(X_test)
        results[name] = rmse(y_test, pred)
        print(f"{name:<24}{results[name]:<10.2f}"
              f"{mean_absolute_error(y_test, pred):<10.2f}"
              f"{r2_score(y_test, pred):<9.3f}{dt:.0f}")

    print(f"\n  LinearRegression struggles because the relationship is")
    print(f"  non-linear (cold weather costs disproportionately more).")
    print(f"  Tree ensembles pick that up without being told.\n")

    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05,
                              random_state=0, verbose=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(0)],
    )
    print(f"Early stopping: asked for 2000 trees, stopped at "
          f"{model.best_iteration_}")
    print(f"  test RMSE: £{rmse(y_test, model.predict(X_test)):.2f}")
    print("  It found the turning point on its own. A random forest has no")
    print("  such point, which is the key difference between the two.\n")

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("LightGBM feature importance (split count):")
    for n, v in imp.items():
        print(f"  {n:<22} {v:>5.0f}  {'#' * int(v / imp.max() * 30)}")
    print()

    next_month = pd.DataFrame([[-1.0, 5, 1, 4]], columns=FEATURES)
    print("-1C week, 5 days WFH, heating on, 4 people:")
    print(f"  Predicted bill: £{model.predict(next_month)[0]:.2f}\n")

    joblib.dump(model, "energy_model.joblib")
    reloaded = joblib.load("energy_model.joblib")
    same = np.allclose(model.predict(X_test), reloaded.predict(X_test))
    print(f"Saved to energy_model.joblib, reloaded, identical: {same}")
    print("  Pickle-based, so it is NOT safe to load untrusted files, and it")
    print("  is tied to the library version. For production, pin your versions.\n")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    evals = model.evals_result_["valid_0"]["rmse"]
    ax[0].plot(evals)
    ax[0].axvline(model.best_iteration_, color="r", linestyle="--",
                  label=f"stopped at {model.best_iteration_}")
    ax[0].set_xlabel("boosting round"); ax[0].set_ylabel("validation RMSE")
    ax[0].set_title("Early stopping"); ax[0].legend(); ax[0].grid(alpha=.3)

    pred = model.predict(X_test)
    ax[1].scatter(pred, y_test, alpha=.6)
    lims = [min(pred.min(), y_test.min()), max(pred.max(), y_test.max())]
    ax[1].plot(lims, lims, "r--")
    ax[1].set_xlabel("predicted £"); ax[1].set_ylabel("actual £")
    ax[1].set_title("Predicted vs actual"); ax[1].grid(alpha=.3)

    names = list(results.keys()); vals = list(results.values())
    ax[2].barh(names, vals)
    ax[2].set_xlabel("test RMSE (£)"); ax[2].set_title("Model comparison")
    ax[2].grid(alpha=.3, axis="x")

    plt.tight_layout(); plt.savefig("plots/boosting_analysis.png", dpi=110)
    print("  -> saved plots/boosting_analysis.png\n")

    print("Test RMSE as trees accumulate (lr=0.3, no early stopping):")
    for n in (10, 50, 100, 300, 800, 1500):
        gb = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.3,
                               random_state=0, verbose=-1).fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=n, random_state=0,
                                   n_jobs=-1).fit(X_train, y_train)
        print(f"  {n:>5} trees   boosting £{rmse(y_test, gb.predict(X_test)):6.2f}"
              f"   forest £{rmse(y_test, rf.predict(X_test)):6.2f}")
    print("\n  Boosting degrades past its optimum. The forest plateaus and stays.")
    print("  That asymmetry is why early stopping exists for one and not the other.")


if __name__ == "__main__":
    main()
