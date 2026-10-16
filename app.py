import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

st.title("💎 Diamond Price Prediction")
st.write(
    "Predict diamond prices using Multiple Linear Regression."
)

st.divider()

# Load trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.pkl"

    with open(model_path, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found! Please save your trained model as model.pkl."
    )
    st.stop()

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

if st.button("💰 Predict Diamond Price", use_container_width=True):

    if x <= 0 or y <= 0 or z <= 0 or carat <= 0:
        st.error("Please enter valid positive measurements.")
    else:
        input_data = pd.DataFrame(
            [[carat, x, y, z]],
            columns=["carat", "x", "y", "z"]
        )

        prediction = model.predict(input_data)[0]

        if prediction < 0:
            st.warning(
                f"Predicted price is negative ({prediction:.2f}). "
                "Check the model and input values."
            )
        else:
            st.success("Prediction completed!")
            st.metric(
                label="Predicted Diamond Price",
                value=f"{prediction:,.2f}"
            )

st.divider()
st.caption(
    "Machine Learning Project | Multiple Linear Regression"
)
