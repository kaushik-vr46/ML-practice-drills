# 06 — What will my energy bill be? (the library way)

## Packages you will meet

| Package | What it does here |
|---|---|
| **scikit-learn** | `GradientBoostingRegressor`, `HistGradientBoostingRegressor` |
| **LightGBM** | The gradient boosting library actually used in industry |
| **joblib** | Saving a trained model to disk and loading it back |
| **matplotlib** | Early stopping curve, prediction diagnostics |

## The ideas worth taking away

**LightGBM is what people actually use.** It is faster than sklearn's version,
handles categoricals natively, and has better early stopping. On tabular data
it or XGBoost wins most of the time — deep learning rarely does.

**Early stopping is the whole game with boosting.** Unlike a random forest,
more trees eventually make boosting *worse*. Early stopping watches a
validation set and halts at the turning point automatically.

**`joblib.dump` is how a model leaves your laptop.** This is the first step of
deployment and it is one line.

## Time: 55 minutes
