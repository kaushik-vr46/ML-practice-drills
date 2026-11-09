"""Every time you have gambled on the bus."""
import numpy as np
import pandas as pd


def load(n=300, seed=1):
    rng = np.random.default_rng(seed)

    hour = rng.integers(6, 22, n).astype(float)
    raining = rng.integers(0, 2, n).astype(float)
    people = rng.integers(0, 15, n).astype(float)
    weekday = rng.integers(0, 2, n).astype(float)

    rush = ((hour >= 8) & (hour <= 9)) | ((hour >= 17) & (hour <= 18))
    z = (-1.5 * rush.astype(float) - 1.2 * raining
         + 0.12 * people + 0.9 * weekday + 0.3)
    prob = 1 / (1 + np.exp(-z))

    return pd.DataFrame({
        "hour_of_day": hour,
        "raining": raining,
        "people_at_stop": people,
        "is_weekday": weekday,
        "made_it": (rng.random(n) < prob).astype(int),
    })


FEATURES = ["hour_of_day", "raining", "people_at_stop", "is_weekday"]
TARGET = "made_it"
