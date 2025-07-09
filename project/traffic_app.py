import os
import streamlit as st
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

# UI Layout
st.title("🚦 Quantum-AI Traffic Optimizer")

# Load data and model
df = get_data()
model = get_model(df)

# Display sample
st.subheader("Data Sample")
st.dataframe(df.head())

# Sidebar controls
hour = st.sidebar.slider("Hour of day",6,19,10)
vehicles = st.sidebar.slider("Vehicle count",100,int(df['vehicle_count'].max()),300)
ratio = st.sidebar.slider("Main road ratio",0.5,0.9,0.7,step=0.1)
cycle = st.sidebar.slider("Total cycle time (s)",30,120,60)

# Predict congestion
pred = model.predict([[hour,vehicles]])[0] # Predict congestion level based on hour and vehicle count.
labels = {0:"Low",1:"Medium",2:"High"}
st.subheader(f"Predicted congestion: {labels[pred]}")

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

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect('equal')

st.pyplot(fig)

# Run quantum optimization
if st.button("Optimize Traffic Lights ⚛️"):
    with st.spinner("Running quantum optimization..."):
        res = optimize_light_cycle(vehicles, ratio, cycle)
    st.success("Optimization complete!")
    st.metric("Main road green time", f"{res['main_green']} s")
    st.metric("Side road green time", f"{res['side_green']} s")
    status = res['status']

    # Store results for SMS
    st.session_state.optimization_result = res
    st.session_state.optimization_params = {
        "hour": hour,
        "vehicles": vehicles,
        "prediction": pred
    }
    print(res)
    print({
        "hour": hour,
        "vehicles": vehicles,
        "prediction": pred
    })

if 'optimization_result' in st.session_state:
    st.divider()
    st.subheader("Alert System")
    phone_number = st.text_input("Enter recipient phone number", "+393713521395")
    custom_message = st.text_area("Custom message", 
                                f"Traffic update: Congestion level at {hour}:00 is {labels[pred]}.")
    
    if st.button("Send SMS Alert 📱"):
        if phone_number:
            opt = st.session_state.optimization_result
            params = st.session_state.optimization_params
            
            message = (f"{custom_message}\n\n"
                    f"Optimal light timing:\n"
                    f"- Main Road: {opt['main_green']}s green\n"
                    f"- Side Road: {opt['side_green']}s green\n"
                    f"Cycle time: {cycle}s")
            
            try:
                success = send_sms(message, phone_number)
                if success:
                    st.success("SMS alert sent!")
                else:
                    st.error("Failed to send SMS. Please check the phone number.")
            except Exception as e:
                st.error(f"SMS sending failed: {str(e)}")
        else:
            st.warning("Please enter a valid phone number")


# Explanation
st.divider()
with st.expander("How it works"):
    st.write(
        "1. ML predicts congestion based on hour & vehicle count\n"
        "2. QAOA with integer QUBO solves green time allocation\n"
        "3. send results via SMS using Twilio"
    )