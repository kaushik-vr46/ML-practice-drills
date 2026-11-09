"""Verify the clustering recovers the hidden structure."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from data import load, FEATURES


def main():
    df = load()
    Xs = StandardScaler().fit_transform(df[FEATURES])

    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)

    # Adjusted Rand Index compares two PARTITIONS while ignoring the arbitrary
    # cluster IDs. 1.0 = identical grouping, 0.0 = random agreement.
    ari = adjusted_rand_score(df["true_vibe"], km.labels_)

    print(f"Silhouette score      : {sil:.3f}")
    print(f"Adjusted Rand Index   : {ari:.3f}   (vs the hidden true groups)")

    sils = {k: silhouette_score(Xs, KMeans(n_clusters=k, n_init=10,
            random_state=0).fit_predict(Xs)) for k in range(2, 8)}
    best = max(sils, key=sils.get)
    print(f"\nSilhouette by k: { {k: round(v, 3) for k, v in sils.items()} }")
    print(f"Best k by silhouette: {best}")

    assert sil > 0.3, "Clusters are not well separated"
    assert ari > 0.6, "Clustering does not match the true groups"
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
