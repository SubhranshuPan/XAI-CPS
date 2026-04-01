import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Parameters
num_samples = 1000
start_time = datetime(2026, 3, 1, 0, 0)

# Generate Base Data (Normal Operations)
timestamps = [start_time + timedelta(minutes=15 * i) for i in range(num_samples)]
pressure = np.random.normal(50, 1.5, num_samples) # Normal pressure ~ 50 psi
vibration = np.random.normal(2, 0.3, num_samples) # Normal vibration ~ 2 mm/s
weather = ['Clear'] * num_samples
latency = np.random.normal(20, 5, num_samples)    # Normal latency ~ 20 ms
anomaly_type = ['None'] * num_samples

# Inject Contextual Anomalies (e.g., Storms causing network lag, heatwaves causing thermal expansion, maintenance causing routing disruptions)
# Let's inject 3 diverse contextual anomalies, each lasting about 10 samples (2.5 hours)
contextual_events = [
    (150, 'Heavy Storm', 'Contextual (Storm)'),
    (500, 'Severe Heat Wave', 'Contextual (Thermal Stress)'),
    (850, 'Scheduled Network Maintenance', 'Contextual (Routing Disruption)')
]
for idx, ext_ctx, anomaly_lbl in contextual_events:
    for i in range(10):
        weather[idx + i] = ext_ctx
        latency[idx + i] = np.random.normal(200, 30) # High latency
        pressure[idx + i] = np.random.normal(28, 2)  # Pressure drops due to lag/environmental stress
        vibration[idx + i] = np.random.normal(6.5, 0.5) # Pump works harder -> high vibration
        anomaly_type[idx + i] = anomaly_lbl

# Inject Real Mechanical Anomalies (No storm, pump just breaks)
# Let's inject 2 real failures
mech_indices = [350, 720]
for idx in mech_indices:
    for i in range(5):
        pressure[idx + i] = np.random.normal(25, 2) # Pressure drops
        vibration[idx + i] = np.random.normal(8.0, 0.8) # Severe vibration
        anomaly_type[idx + i] = 'Mechanical Failure'

# Create DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Water_Pressure_psi': pressure,
    'Pump_Vibration_mms': vibration,
    'External_Context': weather,
    'Network_Latency_ms': latency,
    'Ground_Truth_State': anomaly_type
})

# Save to CSV
df.to_csv('smart_water_telemetry_1000.csv', index=False)
print("Successfully generated 'smart_water_telemetry_1000.csv' with 1000 samples!")