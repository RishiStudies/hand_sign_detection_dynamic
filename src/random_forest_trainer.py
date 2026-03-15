import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("../data/hand_alphabet_data.csv", header=None, on_bad_lines="skip")
data = data.dropna().reset_index(drop=True)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
print(f"Model trained. Test Accuracy: {accuracy:.2f}")

joblib.dump(clf, "../models/hand_alphabet_model.pkl")
np.save("../models/class_labels.npy", clf.classes_)

print("Model and class labels saved successfully.")
