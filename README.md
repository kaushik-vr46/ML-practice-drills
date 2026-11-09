# ML Practice: Tier 1 (scikit-learn and LightGBM)

Seven small machine learning exercises, each built around an everyday problem, to practise the main classical algorithms with Python libraries.

## How this was made

I asked Claude to write a problem statement and a dataset for each algorithm. I typed out the solutions myself, and used Claude to understand the concepts and clear up my doubts along the way.

## Exercises

| # | Problem | Algorithm | Also covers |
|---|---|---|---|
| 01 | Who ate my yoghurt? | k-nearest neighbours | `Pipeline`, scaling, leave-one-out validation |
| 02 | How late will my friend be? | Linear regression | Train/test split, cross-validation, R², MAE |
| 03 | Bus or walk? | Logistic regression | Confusion matrix, ROC curve, thresholds |
| 04 | What should I cook? | Decision tree | Tree plotting, `GridSearchCV`, overfitting |
| 05 | Is this seller dodgy? | Random forest | Permutation importance vs built-in importance |
| 06 | What's my energy bill? | LightGBM | Early stopping, train/validation/test split, saving models |
| 07 | Grouping my music | k-means | Silhouette score, PCA, where k-means fails |

Each folder has:

- `problem.md`: the problem statement
- `data.py`: generates the dataset
- `solution.py`: my solution
- `check.py`: checks the solution works
- `plots/`: charts produced by the solution

`PACKAGES.md` is a short reference on which library to use for what.

## Running it

```bash
pip install numpy pandas scikit-learn matplotlib lightgbm joblib scipy
cd 01_knn_yoghurt_thief
python solution.py
python check.py
```
