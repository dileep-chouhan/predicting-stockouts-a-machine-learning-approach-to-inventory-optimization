import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# --- 1. Synthetic Data Generation ---
# Generate synthetic data for product demand
np.random.seed(42)  # for reproducibility
num_days = 365
dates = pd.date_range(start='2022-01-01', periods=num_days)
demand = 100 + 50 * np.sin(2 * np.pi * np.arange(num_days) / 30) + np.random.normal(0, 20, num_days) #Seasonal trend + noise
lead_time = np.random.randint(1, 5, num_days) #Lead time in days
data = {'Date': dates, 'Demand': demand, 'Lead_Time': lead_time}
df = pd.DataFrame(data)
# --- 2. Feature Engineering ---
df['DayofWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Rolling_Demand_7'] = df['Demand'].rolling(window=7).mean()
df['Rolling_Demand_30'] = df['Demand'].rolling(window=30).mean()
df = df.dropna() #Handle NaN values from rolling mean
# --- 3. Model Training (Linear Regression) ---
X = df[['DayofWeek', 'Month', 'Rolling_Demand_7', 'Rolling_Demand_30', 'Lead_Time']]
y = df['Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# --- 4. Prediction and Evaluation ---
predictions = model.predict(X_test)
# Add evaluation metrics here (e.g., RMSE, MAE) if needed.
# --- 5. Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Demand')
plt.plot(predictions, label='Predicted Demand')
plt.xlabel('Days')
plt.ylabel('Demand')
plt.title('Actual vs. Predicted Demand')
plt.legend()
plt.grid(True)
plt.tight_layout()
output_filename = 'demand_prediction.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 6. Reorder Point Calculation (Example) ---
# A simple reorder point calculation (replace with a more sophisticated method if needed)
safety_stock = 20 # Adjust based on desired service level
reorder_point = predictions[-1] + safety_stock
print(f"\nExample Reorder Point: {reorder_point:.2f}")