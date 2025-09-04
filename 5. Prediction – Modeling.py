import pandas as pd

#1️⃣ Load Feature-Engineered Data
# Load processed dataset
df = pd.read_csv("feature_engineered_taxi_data.csv")

# Define target and features
target = "fare_amount"
features = [
    "distance_km", "log_distance",
    "hour", "day_of_week", "month",
    "is_weekend", "is_night",
    "passenger_count", "distance_passenger"
]
X = df[features]
y = df[target]
print("✅ Data loaded:", df.shape)

##2️⃣ Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Train/Test split:", X_train.shape, X_test.shape)

##3️⃣ Baseline Model – Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("📊 Linear Regression")
print(f"MAE: {mae_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")



##4️⃣ Random Forest Model

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n🌲 Random Forest")
print(f"MAE: {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")

##5️⃣ Gradient Boosting (Optional Stronger Model)

from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print("\n⚡ XGBoost")
print(f"MAE: {mae_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")


##6️⃣ Compare Results

results = {
    "Linear Regression": [mae_lr, rmse_lr],
    "Random Forest": [mae_rf, rmse_rf],
    "XGBoost": [mae_xgb, rmse_xgb],
}

results_df = pd.DataFrame(results, index=["MAE", "RMSE"]).T
print("\n🔎 Model Comparison")
print(results_df)


##7️⃣ Visualization – Predicted vs Actual

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Random Forest: Predicted vs Actual")
plt.show()



###
##8️⃣ Export for Power BI

# Create a single file with features + actual + predicted
test_df = X_test.copy().reset_index(drop=True)    # features from test split
test_df['actual'] = y_test.reset_index(drop=True)
test_df['predicted_rf'] = y_pred_rf
test_df['predicted_xgb'] = y_pred_xgb
test_df['error_rf'] = test_df['actual'] - test_df['predicted_rf']
test_df['error_xgb'] = test_df['actual'] - test_df['predicted_xgb']

# Optional: distance buckets (useful in Power BI too)
bins = [0, 2, 5, 10, 20, 9999]
labels = ['0-2', '2-5', '5-10', '10-20', '20+']
test_df['distance_bucket'] = pd.cut(test_df['distance_km'], bins=bins, labels=labels)

# Save to CSV
output_file = "powerbi_data.csv"
test_df.to_csv(output_file, index=False)

print(f"✅ Export complete. File saved as {output_file}")
