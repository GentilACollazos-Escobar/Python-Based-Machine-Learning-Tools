# Import and load packages and functions in Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    max_error,
    r2_score,
    explained_variance_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

# Load dataset on water adsorption isotherms
data = pd.read_excel("WaterAdsorptionIsotherms_DriedRoasted_Cocoa.xlsx")

# Declare variables from dataset
Type = data.iloc[:, 0]         # Define the type of cocoa
Temperature = data.iloc[:, 2]  # Define the temperature
aw = data.iloc[:, 3]           # Define the water activity cocoa
Moisture = data.iloc[:, 5]     # Define the equilibrium moisture content of cocoa

# Splitting observations for training (75%) and validation (25%)
X = pd.DataFrame({
    'Type': Type,
    'aw': aw,
    'T': Temperature
})
Y = Moisture

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=1234
)

# Preprocessing: encode categorical variable "Type"
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type']),
        ('num', 'passthrough', ['aw', 'T'])
    ]
)

# XGBoost regressor
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Build pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb)
])

# Hyperparameter grid for XGBoost
param_grid = {
    'regressor__n_estimators': [50, 100, 200, 500],
    'regressor__max_depth': [3, 5, 7, 10],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__subsample': [0.7, 0.8, 1.0],
    'regressor__colsample_bytree': [0.7, 0.8, 1.0],
    'regressor__min_child_weight': [1, 3, 5]
}

# Grid Search-Cross Validation (GridSearchCV) to find the best hyperparameters
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, Y_train)

# Best model and predictions
best_xgb = grid_search.best_estimator_
Y_pred_train = best_xgb.predict(X_train)
Y_pred_test = best_xgb.predict(X_test)

# Extract trained XGBoost model
xgb_model = best_xgb.named_steps['regressor']

# Get feature importances
importances = xgb_model.feature_importances_

# Get feature names after preprocessing
preprocessor = best_xgb.named_steps['preprocessor']

# Get names from OneHotEncoder + numerical features
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['Type'])
num_features = ['aw', 'T']

feature_names = list(cat_features) + num_features

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig("XGBoost_FeatureImportance.tif", dpi=300, format="tif", bbox_inches="tight")
plt.show()

from xgboost import plot_importance
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, importance_type='gain')
plt.title("Feature Importance (Gain)")
plt.tight_layout()
plt.show()


import shap
explainer = shap.Explainer(xgb_model)
X_transformed = best_xgb.named_steps['preprocessor'].transform(X_train)
shap_values = explainer(X_transformed)
shap.summary_plot(shap_values, feature_names=feature_names)


# For MSLE, predictions must be non-negative
Y_pred_train_msle = np.clip(Y_pred_train, 0, None)
Y_pred_test_msle = np.clip(Y_pred_test, 0, None)

# Goodness of fit optimized model
# Mean absolute error
mae_train = mean_absolute_error(Y_train, Y_pred_train)
mae_test = mean_absolute_error(Y_test, Y_pred_test)

# Mean square error
mse_train = mean_squared_error(Y_train, Y_pred_train)
mse_test = mean_squared_error(Y_test, Y_pred_test)

# Mean square log error
mseLog_train = mean_squared_log_error(Y_train, Y_pred_train_msle)
mseLog_test = mean_squared_log_error(Y_test, Y_pred_test_msle)

# Median absolute error
medae_train = median_absolute_error(Y_train, Y_pred_train)
medae_test = median_absolute_error(Y_test, Y_pred_test)

# Max error
me_train = max_error(Y_train, Y_pred_train)
me_test = max_error(Y_test, Y_pred_test)

# Coefficient of determination
r2_train = r2_score(Y_train, Y_pred_train) * 100
r2_test = r2_score(Y_test, Y_pred_test) * 100

# Explained variance
var_train = explained_variance_score(Y_train, Y_pred_train) * 100
var_test = explained_variance_score(Y_test, Y_pred_test) * 100

# Print the best parameters and performance metrics
print(f"Best parameters: {grid_search.best_params_}")
print(f"Training Mean absolute error: {mae_train:.4f}")
print(f"Testing Mean absolute error: {mae_test:.4f}")
print(f"Training Mean squared error: {mse_train:.4f}")
print(f"Testing Mean squared error: {mse_test:.4f}")
print(f"Training Mean squared log error: {mseLog_train:.4f}")
print(f"Testing Mean squared log error: {mseLog_test:.4f}")
print(f"Training Median absolute error: {medae_train:.4f}")
print(f"Testing Median absolute error: {medae_test:.4f}")
print(f"Training Max error: {me_train:.4f}")
print(f"Testing Max error: {me_test:.4f}")
print(f"Training R² Score: {r2_train:.4f}")
print(f"Testing R² Score: {r2_test:.4f}")
print(f"Training explained variance: {var_train:.4f}")
print(f"Testing explained variance: {var_test:.4f}")

# Save metrics table
metrics_table = pd.DataFrame({
    "Metric": ["MAE", "MSE", "MSELog", "MEDAE", "ME", "R² (%)", "Var (%)"],
    "Training (75%)": [mae_train, mse_train, mseLog_train, medae_train, me_train, r2_train, var_train],
    "Testing (25%)": [mae_test, mse_test, mseLog_test, medae_test, me_test, r2_test, var_test]
})

metrics_table.to_excel("XGBoost_modeling_performance.xlsx", index=False)

# Plotting the results
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Isotherm (aw vs moisture content) for Training data
axs[0, 0].scatter(X_train['aw'], Y_train, color=(234/255, 123/255, 123/255), label='Experimental')
axs[0, 0].scatter(X_train['aw'], Y_pred_train, color=(54/255, 1/255, 133/255), label='Predicted')
axs[0, 0].set_title('Training (75%) data: Experimental vs Predicted water adsorption isotherms')
axs[0, 0].set_xlabel('Water activity')
axs[0, 0].set_ylabel('Equilibrium moisture content (%d.b.)')
axs[0, 0].set_xlim(0.06, 0.9)
axs[0, 0].set_ylim(0, 10)
axs[0, 0].legend()

# Plot 2: Isotherm (aw vs Moisture Content) for Testing data
axs[0, 1].scatter(X_test['aw'], Y_test, color=(133/255, 25/255, 60/255), label='Experimental')
axs[0, 1].scatter(X_test['aw'], Y_pred_test, color=(117/255, 176/255, 111/255), label='Predicted')
axs[0, 1].set_title('Testing (25%) data: Experimental vs Predicted water adsorption isotherms')
axs[0, 1].set_xlabel('Water activity')
axs[0, 1].set_ylabel('Equilibrium moisture content (%d.b.)')
axs[0, 1].set_xlim(0.06, 0.9)
axs[0, 1].set_ylim(0, 10)
axs[0, 1].legend()

# Plot 3: Experimental vs Calculated (Training Data)
axs[1, 0].scatter(Y_train, Y_pred_train, color='green', label='Experimental vs predicted')
axs[1, 0].plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'k--', lw=2)
axs[1, 0].set_title('Training 75%')
axs[1, 0].set_xlabel('Experimental equilibrium moisture content (%d.b.)')
axs[1, 0].set_ylabel('Predicted equilibrium moisture content (%d.b.)')
axs[1, 0].set_xlim(0, 10)
axs[1, 0].set_ylim(0, 10)
axs[1, 0].legend()

# Plot 4: Experimental vs Calculated (Testing Data)
axs[1, 1].scatter(Y_test, Y_pred_test, color='blue', label='Experimental vs predicted')
axs[1, 1].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
axs[1, 1].set_title('Testing 25%')
axs[1, 1].set_xlabel('Experimental equilibrium moisture content (%d.b.)')
axs[1, 1].set_ylabel('Predicted equilibrium moisture content (%d.b.)')
axs[1, 1].set_xlim(0, 10)
axs[1, 1].set_ylim(0, 10)
axs[1, 1].legend()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

labels = ['(A)', '(B)', '(C)', '(D)']
for ax, label in zip(axs.flatten(), labels):
    ax.text(
        0.02, 0.95, label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        va='top'
    )

plt.tight_layout()
plt.savefig("XGBoostResults.tif", dpi=300, format="tif", bbox_inches="tight")
plt.show()


xgb.plot_importance(best_xgb)
