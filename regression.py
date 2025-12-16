import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Synthetic Data Generation ---
# Simulating 100 days of manufacturing data
np.random.seed(42)
data_size = 100
data = {
    'Staff_Count': np.random.randint(65, 75, data_size),  # Staff fluctuating around 70
    'Avg_Cycle_Time_Sec': np.random.normal(12, 1.5, data_size), # Target ~12s
    'First_Pass_Yield': np.random.uniform(0.85, 0.98, data_size), # Yield %
    'Machine_Uptime_Hrs': np.random.uniform(7.0, 8.0, data_size) # Out of 8 hr shift
}
df = pd.DataFrame(data)

# Simulating Profitability (Target Variable) based on a linear relationship with noise
# Formula: Profit increases with Yield & Uptime, decreases with Cycle Time & Over-staffing
df = (
    50 * df + 
    20 * df['Machine_Uptime_Hrs'] - 
    10 * df - 
    2 * abs(df - 70) + # Penalty for deviation from optimal 70 staff
    np.random.normal(0, 2, data_size) # Random noise
)

# --- 2. Data Visualization (Seaborn) ---
# Correlation Heatmap to understand drivers
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix: Manufacturing Drivers vs Profitability")
plt.show()

# --- 3. Regression Model Building (Scikit-Learn) ---
X = df]
y = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Evaluation & Forecasting ---
predictions = model.predict(X_test)
print(f"Model R-Squared: {r2_score(y_test, predictions):.2f}")
print(f"Coefficients: {model.coef_}")

# Example Forecast: What if we have 70 staff, 95% yield, and 10s cycle time?
new_scenario = pd.DataFrame([[70, 10.0, 0.95, 8.0]], columns=X.columns)
forecasted_profit = model.predict(new_scenario)
print(f"Forecasted Profit Index for Optimal Scenario: {forecasted_profit:.2f}")
