import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from data import load, FEATURES, TARGET, TRUE_COEFFS, TRUE_INTERCEPT

def main():
    df = load()

    print("Summary statistics:")
    print(df.describe().round(2).to_string(), "\n")

    print("Correlation with the target")
    print(df.corr()[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string(), "\n")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    model = LinearRegression().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"R_squared : {r2_score(y_test, y_pred):.3f}")
    print(f"MAE       : {mean_absolute_error(y_test,y_pred):.5f}minutes")
    print(f"MSE.      : {mean_squared_error(y_test,y_pred):.5f}minutes")

    promise = pd.DataFrame([[20,1,1]], columns= FEATURES)
    print(f"Predicted delay: {model.predict(promise)[0]:.0f} minutes \n")

    scores = cross_val_score(LinearRegression(), X,y, cv=5, scoring="r2")
    print(f"5 fold CV R2 score: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f" per fold: {np.round(scores, 3)}\n")


if __name__ == "__main__":
    main()


