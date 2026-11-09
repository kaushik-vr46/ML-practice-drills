# 02 — How late will my friend be? (the library way)

## Packages you will meet

| Package | What it does here |
|---|---|
| **pandas** | The data table, plus `.describe()` and `.corr()` for a first look |
| **scikit-learn** | `LinearRegression`, `Ridge`, `cross_val_score`, metrics |
| **matplotlib** | Predicted-vs-actual plot, residual plot |

## The ideas worth taking away

**You no longer write a training loop.** `.fit()` uses the closed-form normal
equation, not gradient descent — exact, and instant at this size.

**`cross_val_score` replaces your manual train/test split.** One line, five
folds, an honest estimate with error bars.

**Residual plots are how you check a regression.** R-squared can look fine
while the model is systematically wrong in a way the plot makes obvious in
two seconds.

## Time: 45 minutes
