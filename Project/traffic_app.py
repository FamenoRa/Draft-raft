import os
import streamlit as st
from data_clean import load_and_clean
from data_training import train_congestion_model
from QAOA_algo import setup_qaoa_optimizer
from sms_alerts import send_sms

# Constants
DATA_PATH = os.getenv('TRAFFIC_DATA_PATH', 'traffic_datas.csv')
SMS_RECIPIENT = os.getenv('SMS_RECIPIENT', '+00000000000')
GIF_PATH = os.getenv('GIF_PATH', 'vehicle_progression.gif')

# Cached resources
@st.cache_data
def load_data():
    return load_and_clean(DATA_PATH)

@st.cache_resource
def get_model(df):
    return train_congestion_model(df)

@st.cache_resource
def get_optimizer():
    return setup_qaoa_optimizer()

# Main UI
st.title("🚦 Quantum-AI Traffic Optimizer")

df = load_data()
model = get_model(df)
optimizer = get_optimizer()

# Sidebar inputs
st.sidebar.header("Traffic Input")
hour = st.sidebar.slider("Hour of day", 0, 23, 10)
vehicles = st.sidebar.slider("Vehicle count", int(df['vehicle_count'].min()), int(df['vehicle_count'].max()), int(df['vehicle_count'].median()))

# Prediction
congestion = {0:"Low",1:"Medium",2:"High"}[model.predict([[hour, vehicles]])[0]]
st.subheader(f"Predicted congestion: {congestion}")

# Optimization
if st.button("Run Quantum Optimization"):
    qp = QuadraticProgram()
    qp.binary_var('x0', name='MainRoadGreen')
    qp.binary_var('x1', name='SideRoadGreen')
    qp.minimize(linear={'x0':2,'x1':3})
    result = optimizer.solve(qp)
    st.json(result.variables_dict)

    if st.button("Send SMS Alert"):
        msg = f"Update at {hour:02d}:00 - Congestion: {congestion}, Lights: {result.variables_dict}"
        if send_sms(msg, SMS_RECIPIENT):
            st.success("SMS sent!")

# Show GIF
if os.path.exists(GIF_PATH):
    st.image(GIF_PATH, caption="Traffic progression", use_column_width=True)

# Data preview
st.markdown("---")
st.dataframe(df.head())