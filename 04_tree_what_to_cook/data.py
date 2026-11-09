"""Six months of dinner decisions."""
import numpy as np
import pandas as pd


def load(n=250, seed=3):
    rng = np.random.default_rng(seed)

    energy = rng.integers(1, 11, n).astype(float)
    fridge = rng.integers(1, 11, n).astype(float)
    guests = rng.integers(0, 2, n).astype(float)
    minutes = rng.integers(10, 90, n).astype(float)

    y = np.zeros(n, dtype=int)
    for i in range(n):
        if energy[i] <= 3 and minutes[i] <= 25:
            y[i] = 0
        elif guests[i] == 1 and fridge[i] >= 5 and minutes[i] >= 45:
            y[i] = 3
        elif fridge[i] >= 6 and energy[i] >= 5:
            y[i] = 2
        else:
            y[i] = 1

    flip = rng.random(n) < 0.05      # 5% "I just fancied something else"
    y[flip] = rng.integers(0, 4, flip.sum())

    return pd.DataFrame({
        "energy": energy, "fridge_fullness": fridge,
        "guests": guests, "minutes_available": minutes, "dinner": y,
    })


FEATURES = ["energy", "fridge_fullness", "guests", "minutes_available"]
TARGET = "dinner"
LABELS = ["beans_on_toast", "pasta", "stir_fry", "curry"]
