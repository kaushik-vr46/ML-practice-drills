"""Months of logged friend-lateness, as a DataFrame."""
import numpy as np
import pandas as pd


def load(n=120, seed=0):
    rng = np.random.default_rng(seed)

    promised = rng.integers(5, 45, n).astype(float)
    raining = rng.integers(0, 2, n).astype(float)
    parking = rng.integers(0, 2, n).astype(float)

    # Hidden truth: delay = 0.6*promised + 8*rain + 12*parking + 4 + noise
    delay = (0.6 * promised + 8 * raining + 12 * parking + 4
             + rng.normal(0, 3, n))

    return pd.DataFrame({
        "minutes_promised": promised,
        "raining": raining,
        "needs_parking": parking,
        "actual_delay": delay,
    })


FEATURES = ["minutes_promised", "raining", "needs_parking"]
TARGET = "actual_delay"
TRUE_COEFFS = {"minutes_promised": 0.6, "raining": 8.0, "needs_parking": 12.0}
TRUE_INTERCEPT = 4.0
