# Importing necessary libraries
from sklearn.model_selection import GridSearchCV # Hyperparameter fine tuning

# Defining function to find optimised hyperparameters from a grid using cross validation
def tuneHyperparameters(regressor, param_grid, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5):
    print(f"Optimising parameters for {type(regressor).__name__}...") # Signaling function start

    grid_search = GridSearchCV(regressor, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1) # Setting up grid search
    grid_search.fit(X_train, y_train) # Running cross validated grid search

    print("Grid search complete") # Signaling function end
    print(f"Best parameters: {grid_search.best_params_}") # Displaying best hyperparameters
    print(f"Best {scoring} score: {grid_search.best_score_:.5f}") # Displaying best score

    return grid_search.best_estimator_, grid_search.best_params_ # Returning best model and parameters