"""Your log of second-hand purchases."""
import numpy as np
import pandas as pd


def load(n=400, seed=5):
    rng = np.random.default_rng(seed)

    age = rng.integers(1, 2000, n).astype(float)
    reviews = rng.integers(0, 300, n).astype(float)
    below = rng.integers(0, 80, n).astype(float)
    offplatform = rng.integers(0, 2, n).astype(float)
    photos = rng.integers(1, 12, n).astype(float)

    risk = (1.8 * (age < 60) + 1.4 * (reviews < 5) + 1.6 * (below > 45)
            + 2.0 * offplatform + 0.9 * (photos <= 2))
    prob = 1 / (1 + np.exp(-(risk - 3.2)))

    df = pd.DataFrame({
        "account_age_days": age, "review_count": reviews,
        "pct_below_market": below, "wants_offplatform": offplatform,
        "photo_count": photos,
        "was_scam": (rng.random(n) < prob).astype(int),
    })

    # A pure-noise column. Watch how the two importance methods treat it.
    df["listing_id"] = rng.integers(100000, 999999, n).astype(float)
    return df


FEATURES = ["account_age_days", "review_count", "pct_below_market",
            "wants_offplatform", "photo_count", "listing_id"]
TARGET = "was_scam"
