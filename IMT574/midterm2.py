# Trying with StandardSclaer scaling features
step = 'Problem1.1-FeatScale'  # model1
model_type = 'GradientDescent'
# Set hyperparameters
learning_rate = 0.1
iterations = 1000
features = ['Age']
features_scaled = True
predictor = 'Disease'
print(f"{step}: {model_type} Prediction of {predictor} using {features} with {learning_rate=} {iterations=} Age=0 filtered out; {features_scaled=}")

random_state=42
# --- Split data into training and testing sets BEFORE scaling ---
X_train_age, X_test_age, y_train, y_test = train_test_split(
    df_derm['Age'].values.reshape(-1, 1), # Age feature reshaped for scaler
    df_derm['Disease'].values.reshape(-1, 1), # Disease target reshaped for scaler
    test_size=0.2, # Use 20% for testing
    random_state=random_state
)

# --- Standardize X_train and y_train for GD training ---
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_age)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

# Add bias column to scaled training features
X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]

# --- Assume gradient_descent function is defined and trained the model ---
theta_final, cost_history, theta_history = gradient_descent(X_train_b, y_train_scaled, theta, learning_rate, iterations)

# Replace with your actual theta_final from your GD training
theta_final = np.array([[slope], [intercept]])

# --- Recover coefficients in original scale ---
# Get mean and std from the scalers fitted on training data
X_mean_train = scaler_X.mean_[0]
X_std_train = scaler_X.scale_[0]
y_mean_train = scaler_y.mean_[0]
y_std_train = scaler_y.scale_[0]

# Calculate intercept and slope in original scale
slope_new = (y_std_train / X_std_train) * theta_final[1, 0]
intercept_new = y_mean_train - slope_new * X_mean_train

print("\nGradient Descent Model Coefficients (original units):")
print(f"Intercept: {intercept_new:.4f}")
print(f"Slope (X): {slope_new:.4f}")

# --- Prepare test data for evaluation ---
# Scale the test Age data using the *same scaler fitted on training data*
X_test_scaled = scaler_X.transform(X_test_age)

# --- 1. Make predictions on the TEST data using original coefficients ---
# Prediction function: y_hat = intercept + slope * Age (original scale)
y_pred_test = intercept + slope * X_test_age.flatten() # Ensure X_test_age is 1D for calculation

# --- 2. Calculate Mean Squared Error on the TEST data ---
# Use the mean_squared_error function with original scale y_test
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"\nMean Squared Error (MSE) on Test Data: {mse_test:.4f}")

# Calculate RMSE
rmse_test = np.sqrt(mse_test)
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.4f}")

results[step] = {'model_type': model_type, 'learning_rate': learning_rate, 'iterations': iterations, 'features': features, 'predictor': predictor,
                 'slope': slope, 'intercept': intercept, 'dataset': 'filtered, age>0', 'features_scaled': features_scaled}Template-Lab-X_x_x-Lab_Name-Dataset-SGonzales.ipynb