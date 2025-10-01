# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

st.title("House Price Predictor (Demo)")

# load artifacts
preprocessor = joblib.load("artifacts/preprocessor.joblib")
model = tf.keras.models.load_model("artifacts/house_price_model.keras")

st.write("Enter features for prediction (or upload CSV).")

# Create inputs automatically from preprocessor column names (for this example we hardcode typical features)
# In real use you'd programmatically read required columns.
cols = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude',
        'RoomsPerHousehold','BedroomsPerRoom']

inputs = {}
for c in cols:
    inputs[c] = st.number_input(c, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    Xp = preprocessor.transform(df)
    pred = model.predict(Xp).squeeze()
    st.metric("Predicted Median House Value", f"{pred:.3f}")
