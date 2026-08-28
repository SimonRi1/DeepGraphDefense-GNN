import ember
import numpy as np
import pandas as pd

# load training set
X_train, y_train, X_test, y_test = ember.read_vectorized_features("/home/simone/UNI/Master/Tesi/thesis-project/data/raw/ember2018")

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")

# Distribuzione label
unique, counts = np.unique(y_train, return_counts=True)
for val, count in zip(unique, counts):
    label = {-1: "unlabeled", 0: "benigno", 1: "maligno"}.get(int(val), "?")
    print(f"  {label}: {count}")