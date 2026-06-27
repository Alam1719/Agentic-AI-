import pickle
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)



iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(
    X_train_scaled,
    y_train
)


y_pred = model.predict(
    X_test_scaled
)



accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


with open(
    "../models/knn_model.pkl",
    "wb"
) as file:
    pickle.dump(
        model,
        file
    )



with open(
    "../models/scaler.pkl",
    "wb"
) as file:
    pickle.dump(
        scaler,
        file
    )


print("\nModel and scaler saved successfully.")