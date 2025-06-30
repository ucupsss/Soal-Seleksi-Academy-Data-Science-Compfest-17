import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, DMatrix, train as xgb_train
from xgboost.callback import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ======== 1. LOAD DATA ========
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ======== 2. FEATURE ENGINEERING ========
def add_data_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df = df.drop(columns=["date", "ID"])
    return df

train = add_data_features(train)
test = add_data_features(test)

# ======== 3. SPLIT FEATURE & TARGET ========
X = train.drop(columns=["electricity_consumption"])
y = train["electricity_consumption"]
X_test = test.copy()

# ======== 4. ONE-HOT ENCODING =========
X = pd.get_dummies(X, columns=["cluster_id"])
X_test = pd.get_dummies(X_test, columns=["cluster_id"])

X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# ======== 5. TRAIN-VAL SPLIT ========
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ======== 6. GRID SEARCH CV =========
print("\nTuning hyperparameter dengan GridSearchCV...")
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train)
best_params = grid.best_params_
print("Best parameters:", best_params)

# ======== 7. TRAINING XGBOOST DENGAN early stopping =========
print("\nTraining XGBoost dengan best parameters + early stopping...")

# buat DMatrix untuk training dan validation
dtrain = DMatrix(X_train, label=y_train)
dval = DMatrix(X_val, label=y_val)

params = {
    "objective": "reg:squarederror",
    "learning_rate": best_params['learning_rate'],
    "max_depth": best_params['max_depth'],
    "subsample": best_params['subsample'],
    "colsample_bytree": best_params['colsample_bytree']
}

evals_result = {}

booster = xgb_train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_params['n_estimators'],
    evals=[(dval, "validation")],
    callbacks=[EarlyStopping(rounds=10)],
    evals_result=evals_result,
    verbose_eval=True
)


# ======== 8. EVALUASI TRAINING & VALIDATION ========
y_pred_train = booster.predict(dtrain)
y_pred_val = booster.predict(dval)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"\nTraining RMSE: {rmse_train:.4f}")
print(f"Validation RMSE: {rmse_val:.4f}")

# ======== 9. PREDIKSI TEST SET ========
dtest = DMatrix(X_test)
y_pred_test = booster.predict(dtest)

# ======== 10. SIMPAN SUBMISSION ========
submission = pd.read_csv("submission.csv")
submission["electricity_consumption"] = y_pred_test
submission.to_csv("submission_xgboost_tuned.csv", index=False)

print("\nsubmission_xgboost_tuned.csv berhasil dibuat!")

# ======== 11. PLOT VALIDATION RMSE PER BOOSTING ROUND =========
# plotting validation RMSE per boosting round
import matplotlib.pyplot as plt

val_rmse = evals_result['validation']['rmse']

plt.figure(figsize=(10, 5))
plt.plot(val_rmse, label='Validation RMSE')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.title('Validation RMSE per Boosting Round')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()