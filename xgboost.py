import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Features:
# [Size in sqft]

X = [
    [1000],
    [1500],
    [2000],
    [2500],
    [3000]
]

# House prices
y = [100, 150, 200, 250, 300]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create XGBoost model
model = xgb.XGBRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE:", rmse)

# Predict new house price
new_house = [[2200]]

price = model.predict(new_house)

print("Predicted Price:", price[0])