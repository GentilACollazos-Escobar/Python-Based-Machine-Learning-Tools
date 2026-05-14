import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, max_error, r2_score, explained_variance_score 
from sklearn.preprocessing import StandardScaler

# Load dataset on water adsorption isotherms
data=pd.read_excel("WaterAdsorptionIsotherms_DriedRoasted_Cocoa.xlsx") 

# Declare variables from dataset
Type=data.iloc[:,0] ##Define the type of cocoa
Temperature=data.iloc[:,2]##Define the temperature
aw=data.iloc[:,3] ##Define the water activity cocoa
Moisture=data.iloc[:,5] ##Define the equilibrium moisture content of cocoa

# Splitting observations for training (75%) and validation (25%)
X=pd.DataFrame({'Type': Type, 'aw': aw,'T': Temperature})
Y=Moisture
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)

# Standardizing the data (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

###SVM Regression ###
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm = SVR()
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
svm_grid_search.fit(X_train_scaled, Y_train)

# Best model and predictions
best_svm = svm_grid_search.best_estimator_
Y_pred_train = best_svm.predict(X_train_scaled)
Y_pred_test = best_svm.predict(X_test_scaled)

# Goodness of fit optimized model
# Mean absolute error
mae_train = mean_absolute_error(Y_train, Y_pred_train)
mae_test = mean_absolute_error(Y_test, Y_pred_test)
# Mean square error
mse_train = mean_squared_error(Y_train, Y_pred_train)
mse_test = mean_squared_error(Y_test, Y_pred_test)
# Mean square log error
mseLog_train = mean_squared_log_error(Y_train, Y_pred_train)
mseLog_test = mean_squared_log_error(Y_test, Y_pred_test)
# Median absolute error
medae_train = median_absolute_error(Y_train, Y_pred_train)
medae_test = median_absolute_error(Y_test, Y_pred_test)
# Max error
me_train = max_error(Y_train, Y_pred_train)
me_test = max_error(Y_test, Y_pred_test)
# Coefficient of determination
r2_train = (r2_score(Y_train, Y_pred_train)) *100
r2_test = (r2_score(Y_test, Y_pred_test)) *100
# Explained variance
var_train = (explained_variance_score(Y_train, Y_pred_train)) *100
var_test = (explained_variance_score(Y_test, Y_pred_test))*100  

# Print the best parameters and performance metrics
print(f"Best parameters: {best_svm}")
print(f"Training Mean absolute error: {mae_train:.4f}")
print(f"Testing Mean absolute error: {mae_test:.4f}")
print(f"Training Mean squared error: {mse_train:.4f}")
print(f"Testing Mean Squared Error: {mse_test:.4f}")
print(f"Training Mean squared log error: {mseLog_train:.4f}")
print(f"Testing Mean Squared Error: {mseLog_test:.4f}")
print(f"Training Median absolute error: {medae_train:.4f}")
print(f"Testing Median absolute error: {medae_test:.4f}")
print(f"Training Max error: {me_train:.4f}")
print(f"Testing Max error: {me_test:.4f}")
print(f"Training R² Score: {r2_train:.4f}")
print(f"Testing R² Score: {r2_test:.4f}")
print(f"Training explained variance: {var_train:.4f}")
print(f"Testing explained variance: {var_test:.4f}")


metrics_table = pd.DataFrame({
    "Metric": ["MAE","MSE","MSELog","MEDAE","ME","R² (%)","Var (%)"],
    "Training (75%)": [mae_train,mse_train,mseLog_train,medae_train,me_train,r2_train,var_train    ],
    "Testing (25%)": [mae_test,mse_test,mseLog_test,medae_test,me_test,r2_test,var_test]})

metrics_table.to_excel("Support Vector Machine modeling performance.xlsx")

# Plotting the results
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# Plot 1: Isotherm (aw vs moisture Content) for Training data
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
plt.rcParams.update({"font.family": "sans-serif","font.size": 14,"axes.titlesize": 14,
    "axes.labelsize": 14,"xtick.labelsize": 14,"ytick.labelsize": 14,
    "legend.fontsize": 14})
labels = ['(A)', '(B)', '(C)', '(D)']
for ax, label in zip(axs.flatten(), labels):
    ax.text(0.02, 0.95, label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top')
plt.tight_layout()
plt.savefig("SupportVectorMachineResults.tif", dpi=300, format="tif", bbox_inches="tight")
plt.show()