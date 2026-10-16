import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("diamond_price_model (1).pkl")

# Page configuration
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="centered"
)

st.title("💎 Diamond Price Prediction")
st.write("Enter diamond details to predict its estimated price.")

st.divider()

# User inputs
carat = st.number_input(
    "Carat (Diamond Weight)",
    min_value=0.01,
    max_value=5.00,
    value=0.50,
    step=0.01
)

x = st.number_input(
    "Length (x) in mm",
    min_value=0.01,
    max_value=15.0,
    value=4.5,
    step=0.1
)

y = st.number_input(
    "Width (y) in mm",
    min_value=0.01,
    max_value=15.0,
    value=4.5,
    step=0.1
)

z = st.number_input(
    "Depth (z) in mm",
    min_value=0.01,
    max_value=10.0,
    value=2.8,
    step=0.1
)

# Prediction button
if st.button("Predict Diamond Price", type="primary"):
    if x <= 0 or y <= 0 or z <= 0 or carat <= 0:
        st.error("Please enter valid positive values.")
    else:
        input_data = np.array([[carat, x, y, z]])

        prediction = model.predict(input_data)[0]

        st.success("Prediction Completed!")
        st.metric(
            label="Predicted Diamond Price",
            value=f"{prediction:,.2f}"
        )

st.caption("Note: Prediction accuracy depends on the model's training data.")
