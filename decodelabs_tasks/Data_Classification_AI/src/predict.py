import pickle
import numpy as np

with open("../models/knn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("../models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

import pandas as pd

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}
print("Prediction:", species[prediction[0]])