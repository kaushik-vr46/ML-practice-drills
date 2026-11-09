# 05 — Is this seller dodgy? (the library way)

## Packages you will meet

| Package | What it does here |
|---|---|
| **scikit-learn** | `RandomForestClassifier`, `permutation_importance`, `RandomizedSearchCV` |
| **scipy** | Random distributions to sample hyperparameters from |
| **matplotlib** | Importance comparison, OOB convergence |

## The ideas worth taking away

**`oob_score=True` gives free validation.** Each tree is tested on the rows
its bootstrap sample missed. No held-out set needed.

**Impurity importance lies; permutation importance does not.** The built-in
`feature_importances_` inflates features with many split points.
`permutation_importance` shuffles one column and measures how much
performance drops — slower, honest, and what you should actually report.

**`RandomizedSearchCV` beats `GridSearchCV` when the space is large.**
Sampling 30 random combinations usually finds something nearly as good as
exhausting 500.

## Time: 50 minutes
