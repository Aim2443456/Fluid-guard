import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import datetime

# Load data
df = pd.read_csv('data/fluid_guard_corrosion_data.csv')

# Convert corrosion_start_date to datetime
df['corrosion_start_date'] = pd.to_datetime(df['corrosion_start_date'])

# Use today's date as reference
today = datetime.datetime.today()

# Create target: days_to_corrosion
df['days_to_corrosion'] = (df['corrosion_start_date'] - today).dt.days
df = df[df['days_to_corrosion'] > 0]  # remove negative or zero

# Drop the original date column
df = df.drop(columns=['corrosion_start_date'])

# Separate target before encoding
target = 'days_to_corrosion'
y = df[target]
X = df.drop(columns=[target])

# Convert categorical features to numeric
X = pd.get_dummies(X)

# Now X and y are ready
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

# Save model
joblib.dump(model, 'corrosion_model.pkl')
print("✅ Model saved as corrosion_model.pkl")
