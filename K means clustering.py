import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. Synthetic Operator Data ---
# Creating data for 70 operators
np.random.seed(101)
operators = pd.DataFrame({
    'Operator_ID': range(1, 71),
    'Avg_Speed_Units_Per_Hour': np.random.normal(60, 10, 70), # Speed
    'Error_Rate_Percent': np.random.uniform(0.1, 5.0, 70),    # Quality (lower is better)
    'Consistency_Score': np.random.uniform(70, 99, 70)        # Stability
})

# --- 2. Preprocessing ---
# K-Means requires scaling as features have different units (Speed vs %)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(operators])

# --- 3. K-Means Clustering (Scikit-Learn) ---
# We define 3 clusters: High Performers, Precision Specialists, and Trainees
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
operators['Cluster'] = kmeans.fit_predict(features_scaled)

# Naming the clusters based on centroids (Logic would vary by run results)
# Assumed mapping:
# 0: High Speed, High Error -> "Pace Setters" (Good for initial assembly)
# 1: Low Speed, Low Error -> "Precision Craftsmen" (Good for IP68 sealing)
# 2: Low Speed, High Error -> "Needs Training"

# --- 4. Visualization (Seaborn) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=operators, 
    x='Avg_Speed_Units_Per_Hour', 
    y='Error_Rate_Percent', 
    hue='Cluster', 
    palette='viridis', 
    s=100
)
plt.title("Strategic Workforce Allocation Clusters")
plt.xlabel("Speed (Units/Hour)")
plt.ylabel("Error Rate (%)")
plt.axhline(y=2.5, color='r', linestyle='--', label='Max Acceptable Error')
plt.legend()
plt.show()

# --- 5. Strategic Allocation Output ---
print(operators.groupby('Cluster')].mean())
