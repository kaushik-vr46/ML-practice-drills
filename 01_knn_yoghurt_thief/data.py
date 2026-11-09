"""Kitchen incident log — as a pandas DataFrame this time.

A DataFrame is a table with named columns. The practical benefit over a bare
NumPy array: you refer to `df["fridge_openings"]` instead of `X[:, 1]`, and
you stop miscounting column indices at 11pm.
"""
import pandas as pd


def load():
    """Returns a DataFrame. Features + the 'culprit' label."""
    rows = [
        (12, 3, 1, "Priya"), (15, 4, 1, "Priya"), (13, 3, 1, "Priya"), (16, 5, 1, "Priya"),
        (2, 1, 0, "Tom"),    (3, 1, 0, "Tom"),    (2, 2, 0, "Tom"),    (4, 1, 0, "Tom"),
        (25, 6, 1, "Alex"),  (30, 7, 1, "Alex"),  (28, 6, 1, "Alex"),  (26, 8, 1, "Alex"),
        (8, 2, 0, "Sam"),    (6, 2, 0, "Sam"),    (7, 3, 0, "Sam"),    (9, 2, 0, "Sam"),
    ]
    return pd.DataFrame(rows, columns=[
        "minutes_in_kitchen", "fridge_openings", "left_a_mess", "culprit"])


def tonight():
    """The incident. Deliberately borderline between Sam and Priya."""
    return pd.DataFrame([[10, 5, 1]], columns=[
        "minutes_in_kitchen", "fridge_openings", "left_a_mess"])


FEATURES = ["minutes_in_kitchen", "fridge_openings", "left_a_mess"]
TARGET = "culprit"
