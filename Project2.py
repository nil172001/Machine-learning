import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the dataset
df = pd.read_csv("LAozone.data")  # Ensure your dataset is in the correct path

# Separate predictors (X) and target variable (y)
X = df.drop(columns=['ozone']).values  # Drop 'ozone' (target) and keep predictors
y = df['ozone'].values  # Target variable (ozone levels)

# Define a range of lambda values (log scale)
lambda_values = np.logspace(-4, 4, 20)  # 20 values from 10^-4 to 10^4

# Prepare 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store generalization errors for each fold and each lambda
fold_errors = []

# Loop over lambda values
for lambda_ in lambda_values:
    model = make_pipeline(StandardScaler(), Ridge(alpha=lambda_))  # Ridge Regression with standardization
    fold_error = []  # Store errors for each fold

    # Perform cross-validation and collect errors for each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)  # Train model
        y_pred = model.predict(X_test)  # Predict on test set
        
        mse = np.mean((y_pred - y_test) ** 2)  # Calculate MSE for this fold
        fold_error.append(mse)  # Append error for this fold

    fold_errors.append(fold_error)  # Store errors for all folds for this lambda

# Plotting the results
plt.figure(figsize=(8, 6))

# Loop to plot each fold's error for each lambda
for i, lambda_ in enumerate(lambda_values):
    plt.plot([lambda_] * len(fold_errors[i]), fold_errors[i], 'o', color='blue', alpha=0.5)

# Plot the mean error for each lambda
mean_errors = [np.mean(errors) for errors in fold_errors]
plt.plot(lambda_values, mean_errors, marker='o', color='r', linestyle='-', markersize=6, label='Mean Error')

# Format plot
plt.xscale('log')  # Use a logarithmic scale for lambda
plt.xlabel('Lambda (Regularization Parameter)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Generalization Error vs. Lambda for Ridge Regression', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
