# 03 — Bus or walk? (the library way)

## Packages you will meet

| Package | What it does here |
|---|---|
| **scikit-learn** | `LogisticRegression`, and the whole `metrics` module |
| **matplotlib** | ROC curve, confusion matrix, threshold sweep |

## The ideas worth taking away

**Accuracy is a poor summary of a classifier.** `classification_report` gives
you precision, recall and F1 per class in one line, and they tell different
stories.

**The confusion matrix shows you which mistakes you make** — false alarms and
misses are not the same error and usually do not cost the same.

**ROC-AUC is threshold-free.** It measures how well the model *ranks*,
independent of where you draw the line.

**Where you draw that line is a business decision, not a modelling one.**

## Time: 50 minutes
