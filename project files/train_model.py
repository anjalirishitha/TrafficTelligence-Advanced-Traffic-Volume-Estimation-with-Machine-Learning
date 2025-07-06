import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("traffic volume.csv")

# Drop NA and parse datetime
df.dropna(inplace=True)
df["date_time"] = pd.to_datetime(df["date"] + " " + df["Time"], dayfirst=True)



# Create time features
df["year"] = df["date_time"].dt.year
df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.day
df["hours"] = df["date_time"].dt.hour
df["minutes"] = df["date_time"].dt.minute
df["seconds"] = df["date_time"].dt.second

# Encode categorical features
df["holiday"] = df["holiday"].astype("category").cat.codes
df["weather"] = df["weather"].astype("category").cat.codes

# Rename target
target = "traffic_volume"
features = ["holiday", "temp", "rain", "snow", "weather", "year", "month", "day", "hours", "minutes", "seconds"]

# Ensure all columns exist
for col in features + [target]:
    if col not in df.columns:
        raise ValueError(f"Missing column in dataset: {col}")

X = df[features]
y = df[target]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scale.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and Scaler saved as model.pkl and scale.pkl")
