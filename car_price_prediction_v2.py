"""
Car Price Prediction - v3 (post code-review fixes)
=====================================================

This revision fixes the issues raised in a code review of the previous
version (see REVIEW_FINDINGS.md). Summary of what changed and why:

1. Category encoding no longer leaks test data into feature construction.
   Rare-brand grouping, one-hot encoding, and numeric scaling are now all
   sklearn Transformers living INSIDE a Pipeline, fit only on whatever data
   GridSearchCV/train_test_split hands them at fit time. Concretely this
   means: (a) the "which brands count as rare" decision is made from
   training folds only, and (b) OneHotEncoder(handle_unknown="ignore") means
   a genuinely new car with an unseen brand/category no longer crashes
   prediction — it just gets zeroed-out dummy columns instead.

2. Model selection and the final reported metric are now on the SAME scale.
   Every model is wrapped in TransformedTargetRegressor(log1p/expm1), so
   GridSearchCV's R2 is computed after back-transforming to dollars — the
   CV ranking and the headline number are answering the same question now,
   instead of ranking models in log-space and reporting dollar-space.

3. A Duan's smearing correction is applied to the back-transformed
   predictions to correct the systematic underestimation bias that log1p/
   expm1 round-trips introduce (Jensen's inequality on the convex exp()).

4. The reported test metrics are no longer a single lucky/unlucky 80/20
   split. The winning model+hyperparameters are refit across 30 repeated
   random train/test splits, and R2/MAE/RMSE are reported as mean +/- std
   with a 95% interval, alongside the single canonical split (seed=42) used
   for the plots.

5. Hyperparameter grids were trimmed. The previous XGBoost grid alone
   searched 288 combinations against ~164 training rows, which mostly adds
   overfitting risk to the model-selection step itself on a 205-row dataset,
   without meaningfully changing which model wins.

Caveat that still applies: with only 205 rows, even a 30-split confidence
interval is an approximation, not a guarantee. Treat these numbers as
"this approach is clearly reasonable and Ridge is a believable winner",
not as a precise benchmark to quote to three decimal places.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
import joblib

RANDOM_STATE = 42
N_REPEATS = 30  # repeated random splits used for the final confidence interval

# ---------------------------------------------------------------------------
# 1. Load data + light feature engineering (safe to do before the split:
#    these only derive brand/model strings from CarName, no target/no
#    train-only statistics are involved yet)
# ---------------------------------------------------------------------------
raw = pd.read_csv("CarPrice_Assignment.csv")

raw[["CarBrand", "CarModel"]] = raw["CarName"].str.split(" ", n=1, expand=True)
raw["CarBrand"] = raw["CarBrand"].str.lower().replace({
    "vw": "volkswagen", "vokswagen": "volkswagen",
    "toyouta": "toyota", "porcshce": "porsche",
    "maxda": "mazda",
})
raw = raw.drop(columns=["car_ID", "CarName", "CarModel"])

binary_mappings = {
    "fueltype": {"gas": 0, "diesel": 1},
    "aspiration": {"std": 0, "turbo": 1},
    "enginelocation": {"front": 0, "rear": 1},
}
for col, mapping in binary_mappings.items():
    raw[col] = raw[col].map(mapping)
raw["symboling"] = raw["symboling"].astype(int)

y = raw["price"].copy()
X = raw.drop(columns=["price"])

numeric_cols = ["wheelbase", "carlength", "carwidth", "carheight", "curbweight",
                 "enginesize", "boreratio", "stroke", "compressionratio",
                 "horsepower", "peakrpm", "citympg", "highwaympg"]
categorical_cols = ["doornumber", "carbody", "drivewheel", "enginetype",
                     "cylindernumber", "fuelsystem", "CarBrand"]
passthrough_cols = ["symboling", "fueltype", "aspiration", "enginelocation"]


class RareBrandGrouper(BaseEstimator, TransformerMixin):
    """Groups CarBrand values seen fewer than `min_count` times into "other".

    Fit-time only looks at whatever data it is given, so when this sits
    inside a Pipeline used by GridSearchCV or train_test_split, the set of
    "common" brands is always derived from the training fold only — never
    from data outside it. A brand never seen during fit (or too rare) simply
    maps to "other" at transform time, so it can't create a new one-hot
    column that didn't exist during training.
    """

    def __init__(self, min_count=4):
        self.min_count = min_count

    def fit(self, X, y=None):
        counts = X["CarBrand"].value_counts()
        self.common_brands_ = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        X = X.copy()
        X["CarBrand"] = X["CarBrand"].where(X["CarBrand"].isin(self.common_brands_), "other")
        return X


def build_pipeline(estimator):
    """Full leakage-safe pipeline: brand grouping -> encode/scale -> model.

    Every step is fit only on the data passed to .fit(), so this is safe to
    use directly inside GridSearchCV (each CV fold gets its own brand
    grouping/encoding/scaling) and to refit from scratch across the repeated
    train/test splits used for the confidence interval below.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
        ("bin", "passthrough", passthrough_cols),
    ])
    target_model = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([
        ("brand_group", RareBrandGrouper(min_count=4)),
        ("prep", preprocessor),
        ("model", target_model),
    ])


def smearing_correct(pipeline, X_train, y_train, X_new):
    """Duan's smearing estimator: corrects the systematic underestimation
    that a plain expm1(predicted log-price) introduces (Jensen's inequality
    on the convex exp function biases naive back-transformed point
    predictions downward on average)."""
    log_pred_train = np.log1p(pipeline.predict(X_train))
    log_resid_train = np.log1p(y_train.to_numpy()) - log_pred_train
    smear_factor = np.mean(np.exp(log_resid_train))
    return pipeline.predict(X_new) * smear_factor


# ---------------------------------------------------------------------------
# 2. Canonical 80/20 split (seed=42) used for model selection + plots
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Model comparison with GridSearchCV, scored in DOLLAR-space R2 (fixes
#    the log-space-vs-dollar-space mismatch from the previous version), with
#    trimmed grids appropriate for a 205-row dataset
# ---------------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_params = {
    "LinearRegression": {"estimator": LinearRegression(), "params": {}},
    "Ridge": {
        "estimator": Ridge(random_state=RANDOM_STATE),
        "params": {"model__regressor__alpha": [0.1, 1.0, 10.0]},
    },
    "Lasso": {
        "estimator": Lasso(random_state=RANDOM_STATE, max_iter=10000),
        "params": {"model__regressor__alpha": [0.001, 0.01, 0.1]},
    },
    "DecisionTree": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "params": {
            "model__regressor__max_depth": [4, 6, None],
            "model__regressor__min_samples_leaf": [2, 5],
        },
    },
    "RandomForest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
        "params": {
            "model__regressor__n_estimators": [200, 400],
            "model__regressor__max_depth": [6, None],
            "model__regressor__min_samples_leaf": [1, 2],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "params": {
            "model__regressor__n_estimators": [100, 200],
            "model__regressor__max_depth": [2, 3],
            "model__regressor__learning_rate": [0.05, 0.1],
        },
    },
    "XGB": {
        "estimator": xgb.XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"),
        "params": {
            "model__regressor__n_estimators": [200, 300],
            "model__regressor__max_depth": [2, 3],
            "model__regressor__learning_rate": [0.05, 0.1],
            "model__regressor__subsample": [0.8, 1.0],
        },
    },
}

results = []
fitted_pipelines = {}

for name, mp in model_params.items():
    pipe = build_pipeline(mp["estimator"])
    grid = GridSearchCV(pipe, mp["params"], cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    fitted_pipelines[name] = grid.best_estimator_
    results.append({
        "model": name,
        "cv_best_r2_dollar_scale": grid.best_score_,
        "best_params": grid.best_params_,
    })

results_df = pd.DataFrame(results).sort_values("cv_best_r2_dollar_scale", ascending=False)
print("\n=== Cross-validated R2 by model (dollar-scale, matches final metric) ===")
print(results_df[["model", "cv_best_r2_dollar_scale"]].to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_pipeline = fitted_pipelines[best_name]
best_params_row = results_df.iloc[0]["best_params"]

prefix = "model__regressor__"
best_estimator_params = {k[len(prefix):]: v for k, v in best_params_row.items() if k.startswith(prefix)}
best_estimator_template = clone(model_params[best_name]["estimator"]).set_params(**best_estimator_params)

# ---------------------------------------------------------------------------
# 4. Metrics on the canonical split (seed=42), with smearing correction
# ---------------------------------------------------------------------------
y_pred = smearing_correct(best_pipeline, X_train, y_train, X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== Best model: {best_name} (single canonical split, seed={RANDOM_STATE}) ===")
print(f"Test R2:   {r2:.4f}")
print(f"Test MAE:  {mae:,.2f}")
print(f"Test RMSE: {rmse:,.2f}")

# ---------------------------------------------------------------------------
# 5. Confidence interval from 30 repeated random splits, refitting the
#    winning model+hyperparameters from scratch each time (brand grouping,
#    encoding, and scaling are all refit fresh per split -> no leakage)
# ---------------------------------------------------------------------------
r2_scores, mae_scores, rmse_scores = [], [], []

for i in range(N_REPEATS):
    Xtr_i, Xte_i, ytr_i, yte_i = train_test_split(X, y, test_size=0.2, random_state=1000 + i)
    pipe_i = build_pipeline(clone(best_estimator_template))
    pipe_i.fit(Xtr_i, ytr_i)
    pred_i = smearing_correct(pipe_i, Xtr_i, ytr_i, Xte_i)
    r2_scores.append(r2_score(yte_i, pred_i))
    mae_scores.append(mean_absolute_error(yte_i, pred_i))
    rmse_scores.append(np.sqrt(mean_squared_error(yte_i, pred_i)))

r2_scores = np.array(r2_scores)
mae_scores = np.array(mae_scores)
rmse_scores = np.array(rmse_scores)

print(f"\n=== {best_name}: {N_REPEATS}-repeat holdout confidence interval ===")
print(f"R2:   {r2_scores.mean():.4f} +/- {r2_scores.std():.4f}  "
      f"(95% range [{np.percentile(r2_scores, 2.5):.4f}, {np.percentile(r2_scores, 97.5):.4f}])")
print(f"MAE:  {mae_scores.mean():,.2f} +/- {mae_scores.std():,.2f}")
print(f"RMSE: {rmse_scores.mean():,.2f} +/- {rmse_scores.std():,.2f}")

pd.DataFrame({"r2": r2_scores, "mae": mae_scores, "rmse": rmse_scores}).to_csv(
    "repeated_split_scores.csv", index=False
)
print("Saved repeated_split_scores.csv")

# ---------------------------------------------------------------------------
# 6. Persist the fitted pipeline so it can be reused for real predictions
#    without needing to re-run this whole script (fixes the "no way to score
#    a new car" gap flagged in review — handle_unknown='ignore' means an
#    unseen brand/category degrades gracefully instead of crashing)
# ---------------------------------------------------------------------------
joblib.dump(best_pipeline, "best_model_pipeline.joblib")
print("Saved best_model_pipeline.joblib")

# ---------------------------------------------------------------------------
# 7. Feature importance (native + permutation), using the canonical split
# ---------------------------------------------------------------------------
fitted_regressor = best_pipeline.named_steps["model"].regressor_
if hasattr(fitted_regressor, "feature_importances_"):
    feature_names = best_pipeline.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(fitted_regressor.feature_importances_, index=feature_names)
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

perm = permutation_importance(best_pipeline, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
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
# 8. Actual vs predicted plot (canonical split)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
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
