import streamlit as st
import pickle
import pandas as pd

with open("models/knn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("🌸 Iris Flower Classification App")
st.write(
    "Enter the flower measurements below and click Predict."
)

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)

if st.button("Predict"):

    new_flower = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    new_flower_scaled = scaler.transform(
        new_flower
    )

    prediction = model.predict(
        new_flower_scaled
    )

    species = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(
        f"🌸 Predicted Species: {species[prediction[0]]}"
    )