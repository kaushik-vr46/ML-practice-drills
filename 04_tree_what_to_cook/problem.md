# 04 — What should I cook tonight? (the library way)

## Packages you will meet

| Package | What it does here |
|---|---|
| **scikit-learn** | `DecisionTreeClassifier`, `GridSearchCV`, `export_text`, `plot_tree` |
| **matplotlib** | Draws the actual tree |

## The ideas worth taking away

**`export_text` prints the tree as readable rules.** Trees are one of very
few models you can inspect directly. Use that.

**`GridSearchCV` replaces hand-tuning.** It tries every combination of
hyperparameters with cross-validation and reports the best. This is the
single biggest time-saver in the library.

**`feature_importances_` comes free** — though it is biased toward
high-cardinality features, which is why exercise 05 introduces permutation
importance as the more trustworthy version.

## Time: 45 minutes
