import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from data import load, FEATURES, TARGET, LABELS

def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]

    print("Dinner frequency:")
    counts = y.value_counts().sort_index()
    for i,n in counts.items():
        print(f" {LABELS[i]:<16} {n:>3} ({n/len(y):.1%})")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y)

    tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)

    tree.fit(X_train, y_train)
    score = tree.score(X_test, y_test)
    print(f"Test accuracy: {score:.1%}\n")

    print("The learned tree is as follows:")
    print(export_text (tree, feature_names = FEATURES, class_names = LABELS,
                       max_depth = 4))

    imp = pd.Series(tree.feature_importances_, index = FEATURES).sort_values(ascending = False)
    print("Feature importance (impurity-based):")
    for name,v in imp.items():
        bar = "#" * int(v*40)
        print(f". {name:<20} {v:.3f} {bar}")
    print()


    tonight = pd.DataFrame([[8, 8, 1, 60]], columns=FEATURES)
    pred = tree.predict(tonight)[0]
    probability = tree.predict_proba(tonight)[0]
    print(f"-> {LABELS[pred]} ({probability[pred]:.0%} of that leaf's training rows)\n ")




    






    

if __name__ == "__main__":
    main()
    