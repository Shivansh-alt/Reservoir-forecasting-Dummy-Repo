import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Data
# Note: For demonstration without the full dataset, we generate synthetic proxy data 
# Replace this section with: df = pd.read_csv('reservoir_dataset.csv')
np.random.seed(42)
time_steps = np.linspace(0, 10, 100)
target_values = 2 * time_steps + np.sin(time_steps) * 2 + np.random.normal(0, 1, 100)
df = pd.DataFrame({'feature_1': time_steps, 'target': target_values})

# 2. Create Lag Features
df['lag_1'] = df['target'].shift(1)
df['lag_2'] = df['target'].shift(2)
df.dropna(inplace=True)

X = df[['feature_1', 'lag_1', 'lag_2']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Evaluation Function
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}\n")

# 4. Model Training & Evaluation
# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
evaluate_model("Linear Regression", y_test, lr.predict(X_test))

# Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_lr = LinearRegression().fit(poly.fit_transform(X_train), y_train)
evaluate_model("Polynomial Regression", y_test, poly_lr.predict(poly.transform(X_test)))

# Ridge Regression
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
evaluate_model("Ridge Regression", y_test, ridge.predict(X_test))

# k-NN Regressor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
knn = KNeighborsRegressor(n_neighbors=5).fit(X_train_scaled, y_train)
evaluate_model("k-NN Regressor", y_test, knn.predict(scaler.transform(X_test)))
