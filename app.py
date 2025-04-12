import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the model
try:
    rf_regressor = joblib.load(r'D:\project\Weather-Prediction\rf_regressor_model (1).pkl')
except FileNotFoundError:
    st.error("Model file not found. Please check the path.")
    st.stop()

# Define input features
st.title("Temperature Prediction with Random Forest")

cloudcover = st.number_input("Cloud Cover (%)", min_value=0, max_value=100, step=1)
winddir = st.number_input("Wind Direction (Degrees)", min_value=0, max_value=360, step=1)
sealevelpressure = st.number_input("Sea Level Pressure (hPa)", min_value=900.0, max_value=1100.0, step=0.1)
windgust = st.number_input("Wind Gust (km/h)", min_value=0.0, max_value=200.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, step=0.1)
precip = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, step=0.1)
uvindex = st.slider("UV Index", min_value=0, max_value=11, step=1)
dew = st.number_input("Dew Point (°C)", min_value=-20.0, max_value=30.0, step=0.1)

# Create a DataFrame from the input features
input_data = pd.DataFrame({
    'cloudcover': [cloudcover],
    'winddir': [winddir],
    'sealevelpressure': [sealevelpressure],
    'windgust': [windgust],
    'humidity': [humidity],
    'windspeed': [windspeed],
    'precip': [precip],
    'uvindex': [uvindex],
    'dew': [dew]
})

if st.button("Predict Temperature"):
    prediction = rf_regressor.predict(input_data)[0]
    st.write(f"## Predicted Temperature: {prediction:.2f} °C")

    st.subheader("Input Feature Values")
    st.dataframe(input_data)

    # Feature Importance
    if hasattr(rf_regressor, 'feature_importances_'):
        st.subheader("Feature Importance")
        feature_importances = pd.Series(rf_regressor.feature_importances_, index=input_data.columns)
        feature_importances_sorted = feature_importances.sort_values(ascending=False)
        fig_importance, ax_importance = plt.subplots()
        sns.barplot(x=feature_importances_sorted, y=feature_importances_sorted.index, ax=ax_importance)
        ax_importance.set_xlabel("Importance Score")
        ax_importance.set_ylabel("Feature")
        ax_importance.set_title("Random Forest Feature Importance")
        st.pyplot(fig_importance)
    else:
        st.warning("Feature importance is not available for this model.")

    
   

