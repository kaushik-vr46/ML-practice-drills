# The package map

Which library does what, and when you reach for it.

## The five you will use constantly

| Package | Import as | Use it for |
|---|---|---|
| **NumPy** | `np` | Arrays and maths. Everything else is built on it. |
| **pandas** | `pd` | Tables with named columns. Loading, cleaning, `groupby`. |
| **scikit-learn** | `sklearn` | Classical ML: models, preprocessing, metrics, validation. |
| **matplotlib** | `plt` | Plots. Ugly by default, works everywhere. |
| **LightGBM** | `lgb` | Gradient boosting. Wins most tabular problems. |

## Inside scikit-learn — the bits that matter

| Module | What is in it |
|---|---|
| `sklearn.pipeline` | **`Pipeline`** — chain preprocessing + model into one object |
| `sklearn.preprocessing` | `StandardScaler`, `OneHotEncoder` |
| `sklearn.model_selection` | `train_test_split`, `cross_val_score`, `GridSearchCV` |
| `sklearn.metrics` | `classification_report`, `roc_auc_score`, `confusion_matrix` |
| `sklearn.inspection` | **`permutation_importance`** — the honest importance |
| `sklearn.dummy` | `DummyClassifier` / `DummyRegressor` — the bar you must clear |

## What to reach for

| Situation | Reach for |
|---|---|
| Tabular data, predicting a number or class | **LightGBM**, then sklearn |
| You need to explain the model to someone | `DecisionTree` + `export_text` |
| Small data, want a fast baseline | `LinearRegression` / `LogisticRegression` |
| No labels at all | `KMeans`, `DBSCAN` |
| Too many dimensions to plot | `PCA` |
| Tuning hyperparameters | `GridSearchCV`, or `RandomizedSearchCV` when the space is big |
| Saving a fitted model | `joblib.dump` |
| Text, images, audio, sequences | **PyTorch** — Tier 2, not here |

## Three habits worth forming now

**Always use `Pipeline`.** It makes scaling bugs and train/test leakage
structurally impossible. Exercise 01 demonstrates exactly what it prevents.

**Always compare against `DummyClassifier` / `DummyRegressor`.** If your model
cannot beat "always guess the mean", it has not earned its existence. Plenty
of shipped models cannot.

**Always use `permutation_importance`, never `feature_importances_`.** The
built-in one is biased toward high-cardinality columns. Exercise 05 plants a
pure-noise column and shows the built-in method rating it as useful.

## Not here, and why

**PyTorch / TensorFlow** — for neural networks. None of these seven problems
needs one, and reaching for deep learning on small tabular data is a common
and expensive mistake. Tier 2.

**XGBoost / CatBoost** — excellent, and near-identical in use to LightGBM.
Learn one properly; the others take an afternoon.

**Optuna** — smarter hyperparameter search than `RandomizedSearchCV`. Worth it
once you have a model you actually care about tuning.
