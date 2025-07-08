import os
import pygame
import streamlit as st
import numpy as np
from data_clean import load_and_clean
from data_training import train_congestion_model
from QAOA_algo import optimize_light_cycle
from qiskit_optimization import QuadraticProgram
from sms_alerts import send_sms

# Configuration from environment
DATA_PATH = os.getenv('TRAFFIC_DATA_PATH','traffic_datas.csv')
SMS_DEFAULT = os.getenv('SMS_RECIPIENT','+00000000000')

# Cache resources
@st.cache_data
def get_data():
    return load_and_clean(DATA_PATH)

@st.cache_resource
def get_model(df):
    return train_congestion_model(df)
    
# Load data and model
df = get_data()
model = get_model(df)

# Streamlit Layout
st.title("🚦 Quantum-AI Traffic Optimizer")



# Display sample
#st.subheader("Data Sample")
#st.dataframe(df.head())

# Sidebar controls
st.sidebar.header("Parameters and visualization")
hour = st.sidebar.slider("Hour of day",6,19,10)

# vehicle counts in 4 junctions
hour_data = df[df['hour'] == hour]
veh_counts = hour_data.groupby('junction_id')['vehicle_count'].sum().reindex([1,2,3,4], fill_value=0).to_dict()
cycle = st.sidebar.slider("Total cycle time (s)",30,120,60)

# Button to launch Pygame animation
# st.sidebar.header("Visualization")
#if st.sidebar.button("Animate Intersection"):
    # This will open a separate Pygame window
#    animate_intersection(veh_counts, duration=10, fps=30)


# Visualization of vehicle distribution at intersection
st.divider()
st.subheader("Intersection Vehicle Visualization")
# Prepare data for the selected hour across junctions
# Assuming junction_ids 1-4 are mapped to intersection corners
hour_df = df[df['hour'] == hour]
# Sum vehicles per junction
veh_counts = hour_df.groupby('junction_id')['vehicle_count'].sum().reindex([1,2,3,4], fill_value=0)

# Coordinates for visualization
junction_coords = {
    1: (0.2, 0.8),  # NW
    2: (0.8, 0.8),  # NE
    3: (0.2, 0.2),  # SW
    4: (0.8, 0.2)   # SE
}

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4,4))
# Plot roads
ax.plot([0,1],[0.5,0.5], color='gray', linewidth=10)
ax.plot([0.5,0.5],[0,1], color='gray', linewidth=10)

# Plot junctions with circle sizes based on vehicle count
total = veh_counts.max() if veh_counts.max()>0 else 1
for j, count in veh_counts.items():
    x,y = junction_coords[j]
    # Scale marker size
    size = 300 * (count / total)
    ax.scatter(x, y, s=size, alpha=0.9, edgecolor='black')
    ax.text(x, y, f"J{j}{int(count)}", ha='center', va='center', color='white', weight='bold')

ax.axis('off')
ax.set_aspect('equal')
st.pyplot(fig)

# Predict congestion
# compute total vehicles at the intersection for prediction
import builtins 
total_veh = hour_data['vehicle_count'].sum()
features = np.array([[hour, total_veh]])
pred = model.predict(features)[0] # Predict congestion level based on hour and vehicle count.
labels = {0:"Low",1:"Medium",2:"High"}
st.subheader(f"Predicted congestion: {labels[pred]}")

# Run quantum optimization
if st.button("Optimize Traffic Lights ⚛️"):
    with st.spinner("Running quantum optimization..."):
        greens = optimize_light_cycle(veh_counts, cycle_time=cycle)
        #res = optimize_light_cycle(vehicles, ratio, cycle)
    st.success("Optimization complete!")
    cols = st.columns(4)
    for i,col in enumerate(cols,start=1):
        col.metric(f"J{i} Green (s)", greens.get(i,0))
    status = greens.pop('status')
    st.write(f"Status: {status}")

    # Store results for SMS
    st.session_state.optimization_result = greens
    st.session_state.optimization_params = {
        "hour": hour_df['hour'].iloc[0],
        "vehicles": veh_counts,
        "prediction": pred
    }


    # SMS
    phone=st.text_input("SMS recipient",SMS_DEFAULT)
    if st.button("Send SMS Alert 📱"):
        if phone:
            opt = st.session_state.optimization_result
            params = st.session_state.optimization_params
            msg = f"Green times: " + ", ".join([f"J{j}:{t}s" for j,t in greens.items()])
            if send_sms(msg,phone): st.success("SMS sent!")




# Explanation
st.divider()
with st.expander("How it works"):
    st.write(
        "1. predicts congestion based on hour & vehicle count\n"
        "2. QAOA with integer QUBO solves green time allocation\n in 4 junctions"
        "3. send results via SMS using Twilio"
    )