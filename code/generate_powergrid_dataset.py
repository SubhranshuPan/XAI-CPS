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
voltage = np.random.normal(132, 2, num_samples)    # Normal voltage ~132 kV (transmission line)
current = np.random.normal(400, 20, num_samples)   # Normal current ~400 A
frequency = np.random.normal(50, 0.1, num_samples) # Normal frequency ~50 Hz (tight tolerance)
external_context = ['Clear'] * num_samples
latency = np.random.normal(20, 5, num_samples)     # Normal latency ~20 ms
anomaly_type = ['None'] * num_samples

# Inject Contextual Anomalies (external conditions cause sensor deviation)
contextual_events = [
    (150, 'Lightning Storm', 'Contextual (Lightning Storm)'),
    (500, 'Extreme Heat Wave', 'Contextual (Thermal Stress)'),
    (850, 'Scheduled Grid Maintenance', 'Contextual (Routing Disruption)')
]
for idx, ext_ctx, anomaly_lbl in contextual_events:
    for i in range(10):
        external_context[idx + i] = ext_ctx
        latency[idx + i] = np.random.normal(200, 30)   # High latency
        voltage[idx + i] = np.random.normal(108, 4)    # Voltage sag due to stress
        current[idx + i] = np.random.normal(650, 40)   # Current surge
        frequency[idx + i] = np.random.normal(49.2, 0.2)  # Frequency dip
        anomaly_type[idx + i] = anomaly_lbl

# Inject Physical Failures (no external cause — equipment breaks)
mech_events = [
    (350, 'Transformer Failure'),
    (720, 'Line Fault')
]
for idx, anomaly_lbl in mech_events:
    for i in range(5):
        voltage[idx + i] = np.random.normal(62, 5)    # Voltage collapse
        current[idx + i] = np.random.normal(900, 50)  # Current spike
        frequency[idx + i] = np.random.normal(49.0, 0.3)  # Frequency deviation
        anomaly_type[idx + i] = anomaly_lbl

# Create DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Voltage_kV': voltage,
    'Current_A': current,
    'Frequency_Hz': frequency,
    'External_Context': external_context,
    'Network_Latency_ms': latency,
    'Ground_Truth_State': anomaly_type
})

# Save to CSV
df.to_csv('smart_powergrid_telemetry_1000.csv', index=False)
print("Successfully generated 'smart_powergrid_telemetry_1000.csv' with 1000 samples!")
