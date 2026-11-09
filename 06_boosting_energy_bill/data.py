"""Smart-meter readings: weekly bills across several years and households."""
import numpy as np
import pandas as pd


def load(n=700, seed=7):
    """Features: [avg_outdoor_temp_c, days_wfh, heating_on, people_in_house]
    Target:   bill in pounds for the period

    The relationship is deliberately awkward for a straight line:

      1. NON-LINEAR  — cold weather costs disproportionately more,
                       not proportionally more
      2. INTERACTION — a cold week is far worse when the house is full,
                       because everyone is home wanting heat
      3. THRESHOLD   — working from home only costs you once the
                       heating is actually on

    Linear regression can represent none of these without being told about
    them explicitly. Tree ensembles find all three on their own. That is the
    entire reason this exercise exists.
    """
    rng = np.random.default_rng(seed)

    temp = rng.uniform(-4, 26, n)
    wfh = rng.integers(0, 8, n).astype(float)
    people = rng.integers(1, 6, n).astype(float)
    heating = (temp < 15).astype(float)

    cold = np.maximum(0, 16 - temp)

    bill = (
        22                                  # standing charge
        + 0.9 * cold ** 1.7                 # 1. non-linear in cold
        + 0.55 * cold * people              # 2. interaction
        + 6.0 * wfh * heating               # 3. threshold effect
        + 9.0 * people
        + rng.normal(0, 5, n)
    )

    return pd.DataFrame({
        "avg_outdoor_temp_c": temp,
        "days_wfh": wfh,
        "heating_on": heating,
        "people_in_house": people,
        "bill_gbp": bill,
    })


FEATURES = ["avg_outdoor_temp_c", "days_wfh", "heating_on", "people_in_house"]
TARGET = "bill_gbp"
