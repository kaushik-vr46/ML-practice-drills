"""Your saved tracks as Spotify-style audio features."""
import numpy as np
import pandas as pd


def load(n_per_group=75, seed=11):
    """Four real groups are hidden in here. The algorithm never sees them —
    `true_vibe` is returned only so you can check afterwards."""
    rng = np.random.default_rng(seed)

    def group(tempo, energy, dance, acoustic, speech):
        return pd.DataFrame({
            "tempo_bpm": rng.normal(tempo, 12, n_per_group),
            "energy": rng.normal(energy, .09, n_per_group),
            "danceability": rng.normal(dance, .09, n_per_group),
            "acousticness": rng.normal(acoustic, .09, n_per_group),
            "speechiness": rng.normal(speech, .04, n_per_group),
        })

    parts, labels = [], []
    for name, args in [("late_night", (128, .82, .80, .08, .07)),
                       ("focus",      (92, .28, .35, .72, .04)),
                       ("gym",        (150, .93, .68, .05, .14)),
                       ("sunday",     (80, .42, .55, .62, .06))]:
        parts.append(group(*args))
        labels += [name] * n_per_group

    df = pd.concat(parts, ignore_index=True)
    for c in df.columns[1:]:
        df[c] = df[c].clip(0, 1)
    df["true_vibe"] = labels

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


FEATURES = ["tempo_bpm", "energy", "danceability", "acousticness", "speechiness"]
