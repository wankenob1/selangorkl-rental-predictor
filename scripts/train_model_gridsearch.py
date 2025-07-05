import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ========== Load data ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "mudah-apartment-kl-selangor.csv")
df = pd.read_csv(data_path)

# ========== Cleaning ==========
# Clean 'monthly_rent'
df['monthly_rent'] = (
    df['monthly_rent']
    .str.replace('RM', '', regex=False)
    .str.replace('per month', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
df['monthly_rent'] = pd.to_numeric(df['monthly_rent'], errors='coerce')

# Clean 'size'
df['size'] = (
    df['size']
    .str.replace('sq.ft.', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
df['size'] = pd.to_numeric(df['size'], errors='coerce')

# Clean 'rooms'
df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')

# Drop invalids and outliers
df_clean = df.dropna(subset=['monthly_rent', 'rooms', 'size', 'furnished', 'region'])
df_clean = df_clean[df_clean['size'] < 5000]

# Encode categorical features
for col in ['furnished', 'region', 'property_type', 'location']:
    df_clean[col] = df_clean[col].astype('category').cat.codes

# ========== Features ==========
X = df_clean[['rooms', 'size', 'furnished', 'region', 'property_type', 'location']]
y = df_clean['monthly_rent']

# ========== Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Grid Search ==========
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# ========== Evaluation ==========
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Best Parameters:", grid.best_params_)
print("📉 RMSE:", round(rmse, 2))
print("📈 R² Score:", round(r2, 4))

# ========== Save model ==========
models_path = os.path.join(script_dir, "..", "models")
os.makedirs(models_path, exist_ok=True)
joblib.dump(best_model, os.path.join(models_path, "rent_predictor_tuned.joblib"))
print("💾 Tuned model saved to models/rent_predictor_tuned.joblib")
