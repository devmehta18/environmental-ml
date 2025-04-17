# ## Machine Learning Models

# In[26]:


X = df[['fire','secveg','crop','urban','port','river','road','mining','dgfor','defor','precp','precn','tempp']]
y = df['refor']


# ### *Linear Regression*

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression

get_ipython().run_line_magic('reload_ext', 'memory_profiler')

get_ipython().run_line_magic('memit', '')
  # Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=45)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=45)

  # Create and train the model
model = LinearRegression()
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
time_taken_1 = end_time - start_time

  # Predict on the validation set
y_test_pred = model.predict(X_test)

import sys

# Determine memory usage of objects
X_train_size = sys.getsizeof(X_train)
X_val_size = sys.getsizeof(X_val)
X_test_size = sys.getsizeof(X_test)
y_train_size = sys.getsizeof(y_train)
y_val_size = sys.getsizeof(y_val)
y_test_size = sys.getsizeof(y_test)

y_test_pred_size = sys.getsizeof(y_test_pred)
total_size = X_train_size + X_val_size + X_test_size + y_train_size + y_val_size + y_test_size + y_test_pred_size
model_size = sys.getsizeof(model)

# Calculate metrics on the validation set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"Time taken: {time_taken_1} seconds")
print("Test Mean Squared Error (MSE):", mse_test)
print("Test Root Mean Squared Error (RMSE):", rmse_test)
print("Test R-squared:", r2_test)
print("Test Mean Absolute Error (MAE):", mae_test)
print("Basic Data Storage:", total_size)
print("Model Memory:", model_size)

# Calculate residuals
residuals = y_test - y_test_pred

# Create residual plot
plt.figure(figsize=(10,6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red')
plt.title('Residuals vs Predicted Values for Linear Regression on Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[29]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# Apply Information Gain
ig = mutual_info_regression(X, y)

# Create a dictionary of feature importance scores
feature_scores = {}
for i in range(len(X.columns)):
    feature_scores[X.columns[i]] = ig[i]

# Sort the features by importance score in descending order
sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

# Print the feature importance scores and the sorted features
for feature, score in sorted_features:
    print("Feature:", feature, "Score:", score)

# Plot a horizontal bar chart of the feature importance scores
fig, ax = plt.subplots()
y_pos = np.arange(len(sorted_features))
ax.barh(y_pos, [score for feature, score in sorted_features], align="center")
ax.set_yticks(y_pos)
ax.set_yticklabels([feature for feature, score in sorted_features])
ax.invert_yaxis()  # Labels read top-to-bottom
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance Scores (Information Gain)")

# Add importance scores as labels on the horizontal bar chart
for i, v in enumerate([score for feature, score in sorted_features]):
    ax.text(v + 0.01, i, str(round(v, 3)), color="black", fontweight="bold")

plt.show()


# In[30]:


from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)  # We negate because learning_curve returns negative values for MSE
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)  # We negate because learning_curve returns negative values for MSE
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    
    return plt

# Defining the model
model = LinearRegression()

# Plotting learning curve for LinearRegression
title = "Learning Curves (LinearRegression)"
cv = 10  # Define the number of folds for cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


# In[31]:


from sklearn.linear_model import Ridge, Lasso

# Create and train the Ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict on the validation set using the Ridge regression model
y_test_pred_ridge = ridge_model.predict(X_test)

# Calculate metrics on the validation set for the Ridge regression model
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)
rmse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge, squared=False)
r2_test_ridge = r2_score(y_test, y_test_pred_ridge)

print("Ridge Validation Mean Squared Error (MSE):", mse_test_ridge)
print("Ridge Validation Root Mean Squared Error (RMSE):", rmse_test_ridge)
print("Ridge Validation R-squared:", r2_test_ridge)

# Create and train the Lasso regression model
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

# Predict on the validation set using the Lasso regression model
y_test_pred_lasso = lasso_model.predict(X_test)

# Calculate metrics on the validation set for the Lasso regression model
mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
rmse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso, squared=False)
r2_test_lasso = r2_score(y_test, y_test_pred_lasso)

print("Lasso Validation Mean Squared Error (MSE):", mse_test_lasso)
print("Lasso Validation Root Mean Squared Error (RMSE):", rmse_test_lasso)
print("Lasso Validation R-squared:", r2_test_lasso)


# ### *Decision Tree*

# In[32]:


from sklearn.tree import DecisionTreeRegressor

get_ipython().run_line_magic('reload_ext', 'memory_profiler')

get_ipython().run_line_magic('memit', '')
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train the model
dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, min_samples_split=40)  # Adjust the max_depth parameter if needed
start_time = time.time()
dt.fit(X_train, y_train)
end_time = time.time()

# Predict on the validation set
y_test_pred = dt.predict(X_test)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(dt, X_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Print the mean cross-validation score
print("Mean cross-validation score:", cv_scores.mean())

# Calculate metrics on the validation set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"Time taken: {end_time - start_time} seconds")
print("Test Mean Squared Error (MSE):", mse_test)
print("Test Root Mean Squared Error (RMSE):", rmse_test)
print("Test R-squared:", r2_test)
print("Test Mean Absolute Error (MAE):", mae_test)

# Calculate residuals for the validation set
residuals_test = y_test - y_test_pred

# Create residual plot for the validation set
plt.figure(figsize=(10,6))
plt.scatter(y_test_pred, residuals_test, alpha=0.5)
plt.axhline(y=0, color='red')
plt.title('Residuals vs Predicted Values for Decision Tree on Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[33]:


from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)  # We negate because learning_curve returns negative values for MSE
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)  # We negate because learning_curve returns negative values for MSE
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    
    return plt

# Defining the model
dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, min_samples_split=40)

# Plotting learning curve for DecisionTreeRegressor
title = "Learning Curves (DecisionTreeRegressor)"
cv = 10  # Define the number of folds for cross-validation
plot_learning_curve(dt, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


# In[34]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and the values you want to try
param_grid = {
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_leaf': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 30, 40]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='r2')

# Fit the model to the data and find the best hyperparameters
grid_search.fit(X_train, y_train)

# Print the best parameters it found
print(grid_search.best_params_)


# In[35]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the hyperparameters and the distributions to sample from
param_dist = {
    'max_depth': randint(5, 25),
    'min_samples_leaf': randint(5, 20),
    'min_samples_split': randint(10, 40)
}

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(dt, param_distributions=param_dist, 
                                   n_iter=20, cv=5, scoring='r2')

# Fit the model to the data and find the best hyperparameters
random_search.fit(X_train, y_train)

# Print the best parameters it found
print(random_search.best_params_)


# ### *Random Forest*

# In[36]:


from sklearn.ensemble import RandomForestRegressor

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=1, max_features=1.0)  # Adjust the number of estimators if needed
start_time = time.time()
rf.fit(X_train, y_train)
end_time = time.time()
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns[indices]
sorted_importances = importances[indices]
for i in range(len(sorted_importances)):
    print(f"{feature_names[i]}: {sorted_importances[i]}")
    
time_taken_2 = end_time - start_time

# Perform 5-fold cross validation
scores = cross_val_score(rf, X_train, y_train, cv=25, scoring='r2')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Make predictions using the Random Forest model
y_pred = rf.predict(X_test)

# Calculate evaluation metrics
# Print cross-validated scores
print("Cross-validated scores:", scores)

# Print the mean R-squared score
print("Mean cross-validated score: ", scores.mean())
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Time taken: {time_taken_2} seconds")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2_2)
print("Mean Absolute Error (MAE):", mae)

n = len(y)  # number of observations

# Calculate residuals
residuals = y_test - y_pred

# Create residual plot
plt.figure(figsize=(10,6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red')
plt.title('Residuals vs Predicted Values for Random Forest on Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[37]:


from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)  # We negate because learning_curve returns negative values for MSE
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)  # We negate because learning_curve returns negative values for MSE
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    
    return plt

# Defining the model
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=1, max_features=1.0)

# Plotting learning curve for RandomForestRegressor
title = "Learning Curves (RandomForestRegressor)"
cv = 10  # Define the number of folds for cross-validation
plot_learning_curve(rf, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


# In[38]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [0.5, 1.0, 'log2', 'sqrt']
}

# Create a base model
rf = RandomForestRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)


# In[39]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(low=100, high=500),
    'max_depth': randint(low=5, high=20),
    'min_samples_leaf': randint(low=1, high=4),
    'max_features': [0.5, 1.0, 'log2', 'sqrt']
}

# Create a base model
rf = RandomForestRegressor()

# Instantiate the random search model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                   n_iter=100, cv=3, n_jobs=-1, verbose=2)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters
print(random_search.best_params_)


# In[40]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# Already defined and trained linear regression model
y_test_pred_lr = model.predict(X_test)
test_error_lr = mean_squared_error(y_test, y_test_pred_lr)
print(f"Test error of Linear Regression: {test_error_lr}")

# Train a RandomForestRegressor
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

# Calculate validation error of RandomForestRegressor
y_test_pred_rf = model_rf.predict(X_test)
test_error_rf = mean_squared_error(y_test, y_test_pred_rf)
print(f"Test error of Random Forest: {test_error_rf}")

# Compare and select the model with lowest validation error
best_model = 'Linear Regression' if test_error_lr < test_error_rf else 'Random Forest'
print(f"Best model based on test error is: {best_model}")


# ### *Gradient Boosting*

# In[41]:


from sklearn.ensemble import GradientBoostingRegressor

get_ipython().run_line_magic('reload_ext', 'memory_profiler')

get_ipython().run_line_magic('memit', '')
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train the model
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8)  # Adjust the number of estimators and learning rate if needed
start_time = time.time()
gb.fit(X_train, y_train)
end_time = time.time()

# Predict on the validation set
y_test_pred = gb.predict(X_test)

# Perform 5-fold cross validation
cv_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring='r2')

# Print cross-validated scores
print("Cross-validated scores:", cv_scores)

# Print the mean cross-validated score
print("Mean cross-validated score:", cv_scores.mean())

# Calculate metrics on the validation set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"Time taken: {end_time - start_time} seconds")
print("Test Mean Squared Error (MSE):", mse_test)
print("Test Root Mean Squared Error (RMSE):", rmse_test)
print("Test R-squared:", r2_test)
print("Test Mean Absolute Error (MAE):", mae_test)

# Calculate residuals for the validation set
residuals_test = y_test - y_test_pred

# Create residual plot for the validation set
plt.figure(figsize=(10,6))
plt.scatter(y_test_pred, residuals_test, alpha=0.5)
plt.axhline(y=0, color='red')
plt.title('Residuals vs Predicted Values for Gradient Boosting on Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[39]:


from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingRegressor

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)  # We negate because learning_curve returns negative values for MSE
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)  # We negate because learning_curve returns negative values for MSE
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    
    return plt

# Defining the model
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8)

# Plotting learning curve for GradientBoostingRegressor
title = "Learning Curves (GradientBoostingRegressor)"
cv = 10  # Define the number of folds for cross-validation
plot_learning_curve(gb, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


# In[76]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1]
}

# Create a GradientBoostingRegressor object
gb = GradientBoostingRegressor()

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring='r2')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the optimal parameters
best_params = grid_search.best_params_

print("Best parameters:", best_params)


# ### *Support Vector Regression*

# In[42]:


from sklearn.svm import SVR

get_ipython().run_line_magic('load_ext', 'memory_profiler')

get_ipython().run_line_magic('memit', '')
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train the SVR model
svm = SVR(kernel='rbf', C=10, epsilon=0.1)
start_time = time.time()
svm.fit(X_train, y_train)  
end_time = time.time()

# Predict on the validation set
y_test_pred = svm.predict(X_test)

# Calculate metrics on the validation set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"Time taken: {end_time - start_time} seconds")
print("Test Mean Squared Error (MSE):", mse_test)
print("Test Root Mean Squared Error (RMSE):", rmse_test)
print("Test R-squared:", r2_test)
print("Test Mean Absolute Error (MAE):", mae_test)

# Calculate residuals for the validation set
residuals_test = y_test - y_test_pred

# Create residual plot for the validation set
plt.figure(figsize=(10,6))
plt.scatter(y_test_pred, residuals_test, alpha=0.5)
plt.axhline(y=0, color='red')
plt.title('Residuals vs Predicted Values for SVR on Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[41]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Define function to plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    
    return plt

# Plotting learning curve for your SVR model
title = "Learning Curves (SVR)"
cv = 10  # Define the number of folds for cross-validation, for instance, 10-fold
plot_learning_curve(svm, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


# In[79]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Create a scorer
scorer = make_scorer(r2_score)

# Create the GridSearchCV object
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring=scorer)

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters:", best_params)

# Fit the model with the best parameters to the training data
best_svr = SVR(**best_params)
best_svr.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = best_svr.predict(X_val)

# Calculate metrics on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)

print("Validation Mean Squared Error (MSE):", mse_val)
print("Validation Root Mean Squared Error (RMSE):", rmse_val)
print("Validation R-squared:", r2_val)
print("Validation Mean Absolute Error (MAE):", mae_val)


# ## Neural Network

# ### *Multi-Layer Perceptron*

# In[43]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score

get_ipython().run_line_magic('reload_ext', 'memory_profiler')

get_ipython().run_line_magic('memit', '')
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% of data for training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% for validation, 15% for testing

# Create and train the model
model = MLPRegressor(hidden_layer_sizes=(100, 50, 50, 25), max_iter=1000, batch_size=250, activation='relu', solver='adam', alpha=0.01, early_stopping=True, n_iter_no_change=25, learning_rate='adaptive')
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

time_taken = end_time - start_time

# Make predictions on the validation set
y_test_pred = model.predict(X_test)

# Perform 5-fold cross-validation
scores = cross_val_score(model, X_train, y_train, cv=25)

# Print the cross-validation scores
print("Cross-validation scores:", scores)

# Print the mean cross-validation score
print("Mean cross-validation score:", scores.mean())

# Calculate metrics on the validation set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"Time taken: {time_taken} seconds")
print("Test Mean Squared Error (MSE):", mse_test)
print("Test Root Mean Squared Error (RMSE):", rmse_test)
print("Test R-squared:", r2_test)
print("Test Mean Absolute Error (MAE):", mae_test)

# Calculate residuals for the validation set
residuals_test = y_test - y_test_pred

# Create residual plot for the validation set
plt.figure(figsize=(10,6))
plt.scatter(y_test_pred, residuals_test, alpha=0.5)
plt.axhline(y=0, color='red')
plt.title('Residuals vs Predicted Values for MLP on Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# In[43]:


from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)  # Negate because learning_curve returns negative values for MSE
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)  # Negate because learning_curve returns negative values for MSE
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    
    return plt

# Defining the model
model = MLPRegressor(hidden_layer_sizes=(100, 50, 50, 25), max_iter=1000, batch_size=250, activation='relu', solver='adam', alpha=0.01, early_stopping=True, n_iter_no_change=25, learning_rate='adaptive')

# Plotting learning curve for MLPRegressor
title = "Learning Curves (MLPRegressor)"
cv = 5  # Define the number of folds for cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


# In[43]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their possible values
param_grid = {
    'hidden_layer_sizes': [(50, 50, 25), (100, 50, 50, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

# Create the MLPRegressor object
mlp = MLPRegressor(max_iter=1000, early_stopping=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='r2', n_jobs=-1)

# Train the model
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best hyperparameters:", grid_search.best_params_)

# Predict on the validation set using the best model
y_val_pred = grid_search.best_estimator_.predict(X_val)


# In[96]:


from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'hidden_layer_sizes': [(50, 50, 25), (100, 50, 50, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

mlp = MLPRegressor(max_iter=1000, early_stopping=True, random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)

# Train the model
random_search.fit(X_train, y_train)

# Print best parameters
print("Best hyperparameters:", random_search.best_params_)

# Predict on the validation set using the best model
y_val_pred = random_search.best_estimator_.predict(X_val)


# In[97]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the neural network architecture
model = Sequential([
    Dense(100, activation='tanh', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Example of dropout layer with 20% dropout rate
    Dense(50, activation='tanh'),
    Dropout(0.2),
    Dense(50, activation='tanh'),
    Dropout(0.2),
    Dense(25, activation='tanh'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model using the training set
model.fit(X_train, y_train, batch_size=50, epochs=150, validation_data=(X_val, y_val))

# Make predictions on the validation set
y_val_pred = model.predict(X_val).flatten()

# Calculate metrics on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)

print("Validation Mean Squared Error (MSE):", mse_val)
print("Validation Root Mean Squared Error (RMSE):", rmse_val)
print("Validation R-squared:", r2_val)
print("Validation Mean Absolute Error (MAE):", mae_val)

# ... (continue with the residuals plot)


# In[ ]: