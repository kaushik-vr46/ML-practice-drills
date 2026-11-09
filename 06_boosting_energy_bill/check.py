"""Verify boosting beats the baselines and early stopping works."""
import numpy as np
import lightgbm as lgb
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data import load, FEATURES, TARGET


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=0)

    dummy = DummyRegressor().fit(X_tr, y_tr)
    linear = LinearRegression().fit(X_tr, y_tr)
    gbm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05,
                            random_state=0, verbose=-1)
    gbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

    r_dummy = rmse(y_te, dummy.predict(X_te))
    r_lin = rmse(y_te, linear.predict(X_te))
    r_gbm = rmse(y_te, gbm.predict(X_te))

    print(f"Dummy (mean)      £{r_dummy:.2f}")
    print(f"LinearRegression  £{r_lin:.2f}")
    print(f"LightGBM          £{r_gbm:.2f}")
    print(f"\nEarly stopping halted at round {gbm.best_iteration_} of 2000")

    assert r_gbm < r_lin, "Boosting should beat linear on non-linear data"
    assert r_gbm < r_dummy * 0.5, "Boosting barely beats predicting the mean"
    assert gbm.best_iteration_ < 2000, "Early stopping never triggered"
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
