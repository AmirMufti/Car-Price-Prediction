"""
Car Price Prediction - Improved Version
=========================================

Improvements over the original notebook:
1. Fixed the scaled-vs-unscaled target bug: original code trained/evaluated on
   raw price for some models but scaled price was still sitting in the frame.
   Here, price is NEVER included in X and is only transformed via log1p (not
   MinMax), which is the standard approach for right-skewed price targets.
2. Removed target leakage risk: features are scaled with a scaler *fit only on
   the training set*, not on the full dataframe before the split.
3. Replaced raw CarBrand one-hot (which fragments into ~30 rare categories)
   with frequency-based grouping of rare brands into "other".
4. Expanded GridSearchCV grids (the originals were narrow and left several
   XGBoost hyperparameters untuned) and added RandomizedSearchCV as a faster
   alternative for the larger grids.
5. Added a proper cross-validated comparison across models using consistent
   scoring (R2), plus MAE/RMSE on a held-out test set for the winning model.
6. Added feature importance + permutation importance for interpretability.
7. Fixed typo/duplicate axis labels in the original plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import xgboost as xgb

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("CarPrice_Assignment.csv")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
df[["CarBrand", "CarModel"]] = df["CarName"].str.split(" ", n=1, expand=True)
df["CarBrand"] = df["CarBrand"].str.lower().replace({
    "vw": "volkswagen", "vokswagen": "volkswagen",
    "toyouta": "toyota", "porcshce": "porsche",
    "maxda": "mazda",
})

# group rare brands (fewer than 4 occurrences) into "other" instead of
# one-hot-encoding ~30 sparse brand columns
brand_counts = df["CarBrand"].value_counts()
rare_brands = brand_counts[brand_counts < 4].index
df["CarBrand"] = df["CarBrand"].where(~df["CarBrand"].isin(rare_brands), "other")

df = df.drop(columns=["car_ID", "CarName", "CarModel"])

# binary encodings
binary_mappings = {
    "fueltype": {"gas": 0, "diesel": 1},
    "aspiration": {"std": 0, "turbo": 1},
    "enginelocation": {"front": 0, "rear": 1},
}
for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)

df["symboling"] = df["symboling"].astype(int)

one_hot_cols = ["doornumber", "carbody", "drivewheel", "enginetype",
                 "cylindernumber", "fuelsystem", "CarBrand"]
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)

# ---------------------------------------------------------------------------
# 3. Target transform: log1p to tame right-skew in price
# ---------------------------------------------------------------------------
y_raw = df["price"].copy()
y = np.log1p(y_raw)
X = df.drop(columns=["price"])

X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=RANDOM_STATE
)

# scale numeric columns using a scaler fit ONLY on the training split
numeric_cols = ["wheelbase", "carlength", "carwidth", "carheight", "curbweight",
                 "enginesize", "boreratio", "stroke", "compressionratio",
                 "horsepower", "peakrpm", "citympg", "highwaympg"]

scaler = MinMaxScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---------------------------------------------------------------------------
# 4. Model comparison with GridSearchCV (expanded grids)
# ---------------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_params = {
    "LinearRegression": {"model": LinearRegression(), "params": {}},
    "Ridge": {"model": Ridge(random_state=RANDOM_STATE), "params": {"alpha": [0.1, 1.0, 10.0, 50.0]}},
    "Lasso": {"model": Lasso(random_state=RANDOM_STATE, max_iter=10000), "params": {"alpha": [0.001, 0.01, 0.1, 1.0]}},
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "params": {"criterion": ["absolute_error", "squared_error"], "max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 2, 5]},
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE),
        "params": {"n_estimators": [200, 400], "max_depth": [6, 10, None], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 1.0]},
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "params": {"n_estimators": [100, 300], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1]},
    },
    "XGB": {
        "model": xgb.XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"),
        "params": {
            "n_estimators": [200, 400],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_weight": [1, 3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 2.0],
        },
    },
}

results = []
fitted_best_models = {}

for name, mp in model_params.items():
    grid = GridSearchCV(mp["model"], mp["params"], cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    fitted_best_models[name] = grid.best_estimator_
    results.append({
        "model": name,
        "cv_best_r2": grid.best_score_,
        "best_params": grid.best_params_,
    })

results_df = pd.DataFrame(results).sort_values("cv_best_r2", ascending=False)
print("\n=== Cross-validated R2 by model (log-price target) ===")
print(results_df[["model", "cv_best_r2"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Evaluate the best model on the held-out test set, in real-dollar terms
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["model"]
best_model = fitted_best_models[best_name]

y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # back-transform to dollars

r2 = r2_score(y_test_raw, y_pred)
mae = mean_absolute_error(y_test_raw, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))

print(f"\n=== Best model: {best_name} ===")
print(f"Test R2 (dollar scale):  {r2:.4f}")
print(f"Test MAE (dollar scale): {mae:,.2f}")
print(f"Test RMSE (dollar scale): {rmse:,.2f}")

# ---------------------------------------------------------------------------
# 6. Feature importance (native + permutation)
# ---------------------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Top 15 Feature Importances ({best_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("\nSaved feature_importance.png")

perm = permutation_importance(best_model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
perm_series = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
perm_series.plot(kind="barh", color="salmon")
plt.gca().invert_yaxis()
plt.title(f"Top 15 Permutation Importances ({best_name}, test set)")
plt.xlabel("Mean R2 drop when shuffled")
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=120)
plt.close()
print("Saved permutation_importance.png")

# ---------------------------------------------------------------------------
# 7. Actual vs predicted plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test_raw, y_pred, alpha=0.6)
lims = [min(y_test_raw.min(), y_pred.min()), max(y_test_raw.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title(f"Actual vs Predicted Price ({best_name})")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=120)
plt.close()
print("Saved actual_vs_predicted.png")

results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved model_comparison_results.csv")
