"""
Linear Regression — predict a continuous target from features.
"""

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


def main() -> None:
    X, y = make_regression(
        n_samples=500,
        n_features=3,
        noise=15.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("=== Linear Regression ===")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept:    {model.intercept_:.4f}")
    print(f"R² score:     {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE:         {rmse:.4f}")


if __name__ == "__main__":
    main()
