import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# App Title
st.title("💎 Diamond Price Prediction")
st.write(
    "Predict diamond prices using Multiple Linear Regression."
)

st.divider()

# Load Trained Model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.pkl"

    with open(model_path, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "model.pkl not found! Please place model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

# User Inputs
st.subheader("Enter Diamond Details")

carat = st.number_input(
    "Carat (Weight)",
    min_value=0.01,
    max_value=5.0,
    value=1.0,
    step=0.1
)

x = st.number_input(
    "Length (mm)",
    min_value=0.1,
    max_value=15.0,
    value=6.5,
    step=0.1
)

y = st.number_input(
    "Width (mm)",
    min_value=0.1,
    max_value=15.0,
    value=6.5,
    step=0.1
)

z = st.number_input(
    "Depth (mm)",
    min_value=0.1,
    max_value=10.0,
    value=4.0,
    step=0.1
)

st.divider()

# Prediction Button
if st.button("💰 Predict Diamond Price", use_container_width=True):

    # Prepare Input Data
    new_diamond = pd.DataFrame({
        "carat": [carat],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    # Predict Price
    predicted_price = model.predict(new_diamond)[0]

    # Display Result
    if predicted_price < 0:
        st.warning(
            f"Predicted price is negative: {predicted_price:.2f}. "
            "Please check the model and input values."
        )
    else:
        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Diamond Price",
            value=f"{predicted_price:,.2f}"
        )

        st.write("### Your Diamond Details")
        st.dataframe(new_diamond, hide_index=True)

# Footer
st.divider()

st.caption(
    "Machine Learning Project | Multiple Linear Regression"
)
