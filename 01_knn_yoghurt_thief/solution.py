import matplotlib
matplotlib.use("Agg")          # write files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load, tonight, FEATURES, TARGET

def build_model(k=3):

    return Pipeline([ ("scaler", StandardScaler()),           # step 1: standardise
                     ("knn", KNeighborsClassifier(n_neighbors=k)),   # step 2: classify
                     ])

def main():
    df = load()
    print(df.head(), "\n")
    print("Incidents per person:")
    print(df[TARGET].value_counts().to_string(), "\n")

    X = df[FEATURES]
    y = df[TARGET]

    model = build_model(k=3)
    model.fit(X,y)
    suspect = model.predict(tonight())[0]
    print(f"Prime suspect: {suspect}")

    probs = model.predict_proba(tonight())[0]
    print("Vote split:")
    for name, p in sorted(zip(model.classes_,probs), key = lambda t : -t[1]):
        if p > 0:
            print(f"  {name:<8}{p:.0%}")
    print()

    # 1 - cross-validation

    scores = cross_val_score(build_model(3), X, y, cv = LeaveOneOut())
    print(f"Leave-one-out accuracy: {scores.mean():.1%} \n")

    # 2 - pick k by cross validation

    print("Picking k by cross-validation, not by guessing")
    ks , accs = list(range(1,12)) , []
    for k in ks:
        acc = cross_val_score(build_model(k), X , y, cv = LeaveOneOut()).mean()
        accs.append(acc)
        print(f"  k = {k:<2}  accuracy = {acc:.1%}")

    plt.figure(figsize=(5,3))
    plt.plot(ks, accs, marker = "o")
    plt.xlabel("k (number of neighbours)")
    plt.ylabel("leave-one-out accuracy")
    plt.title("Picking k by cross-validation, not by guessing")
    plt.grid(alpha = .3)
    plt.tight_layout()
    plt.savefig("plots/k_selection_self.png", dpi = 110)
    print("\n -> saved plots/k_selection_self.png\n")

    # 3 - pipeline and rescaling

    df_sec = df.copy()
    df_sec["minutes_in_kitchen"] *=60
    tonight_sec = tonight().copy()
    tonight_sec["minutes_in_kitchen"] *= 60

    no_scaler_sec = KNeighborsClassifier(n_neighbors = 3)
    no_scaler_sec.fit(df_sec[FEATURES], df_sec[TARGET])
    a = no_scaler_sec.predict(tonight_sec)[0]

    no_scaler = KNeighborsClassifier(n_neighbors = 3)
    no_scaler.fit(df[FEATURES], df[TARGET])
    b = no_scaler.predict(tonight())[0]

    with_scaler = build_model(3)
    with_scaler.fit(df_sec[FEATURES], df_sec[TARGET])
    c = with_scaler.predict(tonight_sec)[0]

    print(f"Without scaler, in seconds: {a}")
    print(f"Without scaler, in minutes: {b}")
    print(f"With scaler, in seconds: {c}")


if __name__ == "__main__":
    main()





     
