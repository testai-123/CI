import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# Generate sample structural damage dataset
X, y = make_classification(n_samples=200, n_features=5, n_informative=3, 
                           n_redundant=0, n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define simple Clonal Selection Algorithm
def affinity(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def clone_and_mutate(antibody, rate=0.1):
    return [gene + random.uniform(-rate, rate) for gene in antibody]

def train_ais(X_train, y_train, n_clones=5):
    memory_set = []
    for idx, antigen in enumerate(X_train):
        label = y_train[idx]
        antibody = antigen.copy()
        clones = [clone_and_mutate(antibody) for _ in range(n_clones)]
        best_clone = min(clones, key=lambda x: affinity(x, antigen))
        memory_set.append((best_clone, label))
    return memory_set

def predict_ais(memory_set, X_test):
    predictions = []
    for antigen in X_test:
        dists = [affinity(antigen, mem[0]) for mem in memory_set]
        nearest = memory_set[np.argmin(dists)]
        predictions.append(nearest[1])
    return predictions

# Train and predict
memory = train_ais(X_train, y_train)
y_pred = predict_ais(memory, X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
