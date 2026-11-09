# 01 — Who ate my yoghurt? (the library way)

## Packages you will meet

| Package | What it does here |
|---|---|
| **pandas** | Holds the data as a labelled table, not a bare array |
| **scikit-learn** | `KNeighborsClassifier`, `StandardScaler`, `Pipeline` |
| **matplotlib** | Plots accuracy against k so you can see the trade-off |

## The one idea worth taking away

In the NumPy version you had to remember to scale your features, and you saw
what happened when you forgot: the accusation changed from Priya to Sam.

`Pipeline` makes that bug **structurally impossible**. It bundles the scaler
and the model into one object, so the scaler is always fitted on training
data only and always applied before predicting.

This is the single most useful class in scikit-learn and most beginners never
use it.

## Time: 35 minutes
