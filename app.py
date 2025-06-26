import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the model
model = pickle.load(open("flower_model.pkl", "rb"))

# Page title
st.title("🌸 Flower Species Predictor by ADIJAT OYETOKE")
st.write("Enter the flower's features to predict its species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Class labels and image links
species_map = {
    0: ("Setosa", "/content/Setosa.jpeg"),
    1: ("Versicolor", "/content/Versicolor.jpeg"),
    2: ("Virginica", "/content/Verginica.jpeg")
}

# Predict on button click
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    species_name, image_url = species_map.get(prediction, ("Unknown", None))

    # Show prediction
    st.success(f"Predicted species: **{species_name}** 🌼")

    # Show flower image
    if image_url:
        st.image(image_url, caption=species_name, width=300)

    # Show prediction probabilities as a bar chart
    st.subheader("🔍 Prediction Probabilities")
    labels = ["Setosa", "Versicolor", "Virginica"]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, probabilities, color=["#86C5D8", "#F4A261", "#9C89B8"])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    plt.grid(True)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    st.pyplot(fig)
