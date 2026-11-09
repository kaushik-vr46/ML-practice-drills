# 07 — Grouping my music library (the library way)

## Packages you will meet

| Package | What it does here |
|---|---|
| **scikit-learn** | `KMeans`, `silhouette_score`, `PCA`, `DBSCAN` |
| **pandas** | `groupby` to profile each cluster |
| **matplotlib / seaborn** | Elbow plot, PCA scatter |

## The ideas worth taking away

**Silhouette score beats the elbow method.** The elbow is a judgement call you
squint at. Silhouette gives you a number between -1 and 1 measuring how well
each point sits in its cluster versus the next nearest one.

**PCA lets you see 5 dimensions in 2.** You cannot plot 5 features. PCA
projects onto the two directions with the most variance so you can eyeball
whether the clusters are real.

**k-means assumes round, similar-sized clusters.** When that assumption
breaks, DBSCAN often works where k-means cannot. The final experiment shows a
shape k-means simply cannot handle.

## Time: 50 minutes
