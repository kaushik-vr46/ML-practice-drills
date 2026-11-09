import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load, FEATURES


def main():
    df = load()
    X = df[FEATURES]

    print("Feature ranges — note tempo is on a completely different scale:")
    print(X.describe().loc[["min", "max"]].round(2).to_string())
    print("  Without scaling, distance would be almost entirely about tempo.\n")

    model = Pipeline([("scaler", StandardScaler()), ("km", KMeans(n_clusters=4, n_init=10, random_state=0))])
    df["cluster"] = model.fit_predict(X)
    Xs = model.named_steps["scaler"].transform(X)

    print(f"Silhouette score: {silhouette_score(Xs, df['cluster']):.3f}")
    print("  1 = tight and well separated, 0 = overlapping, <0 = misassigned\n")

    profile = df.groupby("cluster")[FEATURES].mean().round(2)
    profile["n"] = df.groupby("cluster").size()
    print("Cluster profiles — name these yourself:")
    print(profile.to_string(), "\n")

    print("Cross-tab against the true groups (you would not have this in reality):")
    print(pd.crosstab(df["cluster"], df["true_vibe"]).to_string(), "\n")

    print("Choosing k:")
    print(f"  {'k':<5}{'inertia':<12}{'silhouette'}")
    ks = range(2, 9)
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        inertias.append(km.inertia_)
        s = silhouette_score(Xs, km.labels_)
        sils.append(s)
        print(f"  {k:<5}{km.inertia_:<12.1f}{s:.3f}")
    best_k = list(ks)[int(np.argmax(sils))]
    print(f"\n  Silhouette picks k={best_k}. Inertia just keeps falling —")
    print("  it always does, which is why you cannot simply minimise it.\n")

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    ax[0, 0].plot(list(ks), inertias, marker="o")
    ax[0, 0].set_xlabel("k"); ax[0, 0].set_ylabel("inertia")
    ax[0, 0].set_title("Elbow method — squint for the bend"); ax[0, 0].grid(alpha=.3)

    ax[0, 1].plot(list(ks), sils, marker="s", color="darkorange")
    ax[0, 1].axvline(best_k, color="r", linestyle="--", label=f"best k={best_k}")
    ax[0, 1].set_xlabel("k"); ax[0, 1].set_ylabel("silhouette")
    ax[0, 1].set_title("Silhouette — an actual number"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

    coords = PCA(n_components=2, random_state=0).fit_transform(Xs)
    ax[1, 0].scatter(coords[:, 0], coords[:, 1], c=df["cluster"], cmap="viridis", alpha=.7, s=18)
    ax[1, 0].set_title("PCA projection, coloured by CLUSTER")
    ax[1, 0].set_xlabel("PC1"); ax[1, 0].set_ylabel("PC2")

    for vibe in df["true_vibe"].unique():
        m = df["true_vibe"] == vibe
        ax[1, 1].scatter(coords[m, 0], coords[m, 1], label=vibe, alpha=.7, s=18)
    ax[1, 1].set_title("Same projection, coloured by TRUE group")
    ax[1, 1].set_xlabel("PC1"); ax[1, 1].legend(fontsize=8)

    plt.tight_layout(); plt.savefig("plots/clustering.png", dpi=110)
    print("  -> saved plots/clustering.png")
    print("  The bottom two panels should look nearly identical. That is what")
    print("  a successful clustering looks like when you can check it.\n")

    moons, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
    km_moons = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(moons)
    db_moons = DBSCAN(eps=0.25, min_samples=5).fit_predict(moons)

    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.5))
    ax2[0].scatter(moons[:, 0], moons[:, 1], c=km_moons, cmap="coolwarm", s=18)
    ax2[0].set_title("k-means: cuts straight through both crescents")
    ax2[1].scatter(moons[:, 0], moons[:, 1], c=db_moons, cmap="coolwarm", s=18)
    ax2[1].set_title("DBSCAN: follows the density, gets it right")
    plt.tight_layout(); plt.savefig("plots/kmeans_limitation.png", dpi=110)

    print("Where k-means breaks:")
    print(f"  k-means silhouette on crescents: {silhouette_score(moons, km_moons):.3f}")
    print(f"  DBSCAN found {len(set(db_moons)) - (1 if -1 in db_moons else 0)} clusters"
          f" and {(db_moons == -1).sum()} noise points")
    print("  -> saved plots/kmeans_limitation.png")
    print("\n  k-means assumes round, similar-sized clusters. When that is wrong,")
    print("  no amount of tuning k will save it. Use a density method instead.")


if __name__ == "__main__":
    main()
