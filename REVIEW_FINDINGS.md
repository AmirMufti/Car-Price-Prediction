# Code Review: `car_price_prediction_v2.py`

A sub-agent reviewed the model-comparison script for correctness bugs,
methodology issues, and whether the reported test R2 (0.9226) could be
trusted as-is. Findings below, with the fix applied in this revision noted
under each one.

## Findings

### 1. Model-selection bias in the winning score — no error bar
Seven models were each hyperparameter-tuned via `GridSearchCV`, then the
single best one was evaluated once on a 41-row test set, with no correction
for having picked the best of seven independently-optimized candidates and
no confidence interval around the reported R2.

**Fix applied:** the winning model+hyperparameters are now refit from
scratch across 30 repeated random train/test splits. R2/MAE/RMSE are
reported as mean +/- std with a 95% range, in addition to the single
canonical split used for the plots. See `repeated_split_scores.csv`.

### 2. CV comparison and final test metric were on different scales
Model selection ranked candidates by R2 on `log1p(price)`, but the final
reported R2/MAE/RMSE were computed in dollar space after `expm1`
back-transform. R2 is not scale-invariant under a nonlinear transform, so
the model that won on log-scale CV wasn't guaranteed to be the model that
would win on dollar-scale R2. There was also an uncorrected bias from
`expm1(log-prediction)` systematically underestimating price on average
(Jensen's inequality on the convex exp function).

**Fix applied:** every model is now wrapped in
`TransformedTargetRegressor(func=log1p, inverse_func=expm1)`, so
`GridSearchCV`'s R2 is computed in dollar space directly — the same scale
as the final reported metric. A Duan's smearing correction is also applied
to the back-transformed predictions to correct the systematic
underestimation bias.

### 3. Category encoding was built on the full dataset, before the split
Rare-brand grouping and one-hot encoding were computed over all 205 rows
before `train_test_split`, which is a mild form of leakage (test-set rows
influenced which brands got grouped into "other"), and a latent production
bug: there was no persisted encoder with `handle_unknown="ignore"`, so a
genuinely new car with an unseen brand/category would break prediction with
a column mismatch rather than degrade gracefully.

**Fix applied:** rare-brand grouping is now a custom
`RareBrandGrouper` transformer, and encoding/scaling now live in a
`ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")`, all
inside one `Pipeline`. This pipeline is fit fresh on whatever data
`GridSearchCV` or `train_test_split` hands it — including inside every CV
fold — so there's no leakage anywhere, and an unseen category at inference
time now just produces zeroed dummy columns instead of crashing. The fitted
pipeline is persisted to `best_model_pipeline.joblib` for reuse.

### 4. Hyperparameter grids were large relative to 205 rows
The original XGBoost grid alone searched 288 combinations against ~164
training rows, adding overfitting risk to the model-selection step itself
— likely part of why the more flexible tree ensembles didn't beat Ridge.

**Fix applied:** grids were trimmed to the hyperparameters that matter most
for each model family, keeping the search meaningfully smaller relative to
the dataset size.

### 5. Single 80/20 holdout drove all reported dollar-scale metrics
Same underlying issue as #1, called out separately because it affects
MAE/RMSE too, not just the model-selection R2.

**Fix applied:** covered by the repeated-split confidence interval in #1.

## What was already correct (confirmed by review)
- The log1p/expm1 round trip was applied consistently between training,
  CV scoring, and final evaluation — no mixed-scale comparison bug.
- `MinMaxScaler` was fit only on the training split.
- All models shared identical CV folds for a fair comparison.
- `train_test_split` kept train/test indices aligned across the log and
  raw targets — no index-mismatch bug.

## Verdict
The original 0.9226 test R2 was a real, honestly-computed number (no
leakage bug produced it), but it was a single point estimate that
shouldn't have been quoted without a caveat about sample size and
selection bias. Ridge beating XGBoost/Random Forest on this 205-row
dataset is a sane, non-suspicious result — this revision doesn't change
which model wins, it makes the reported confidence in that result honest.
