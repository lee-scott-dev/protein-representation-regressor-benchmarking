# Importing necessary libraries and functions
import numpy as np # Numerical operations
import pandas as pd # Data manipulation
from sklearn.model_selection import KFold # Cross-validation
from sklearn.metrics import r2_score, mean_squared_error # Regression scoring metrics

# Importing from created .py files
import data_loader # Loads data

# Defining function to run a naive positional-mean baseline, imputing the mean known score per position
def runNaiveBaseline(dataset, verbose=False):
    print("="*50 + "\nNaive baseline details:")
    print(f"\tDataset: {dataset}")
    print(f"\tMethod: mean functional score per position\n" + "-"*50)

    wild_type, scores_df = data_loader.loadDataset(dataset, verbose) # Loading scores data

    known_df = scores_df[scores_df["score"].notna()].copy() # Known variants with scores
    known_df["position"] = known_df["variant"].str[1:-1].astype(int) # Parsing mutation position

    positions = known_df["position"].values # Position of each variant
    y_known = known_df["score"].values # Scores of each variant

    kfold = KFold(n_splits=5, shuffle=True, random_state=42) # 5-fold split as with tuning
    r2_scores = [] # Initialising list to store fold R-squared scores
    rmse_scores = [] # Initialising list to store fold RMSE scores

    # Iterating through folds
    for fold, (train_idx, test_idx) in enumerate(kfold.split(y_known), start=1):
        print(f"Fold {fold}:") # Signalling current fold
        pos_train, pos_test = positions[train_idx], positions[test_idx] # Splitting positions for fold
        y_train, y_test = y_known[train_idx], y_known[test_idx] # Splitting targets for fold

        # Building per-position mean score from training data so test data doesn't leak
        train_df = pd.DataFrame({"position": pos_train, "score": y_train}) # Training positions and scores
        position_means = train_df.groupby("position")["score"].mean() # Mean score per position
        global_mean = y_train.mean() # Fallback to global mean for positions unseen in training

        # Predicting each test variant as the mean score of its position with fallback set
        y_pred = np.array([position_means.get(p, global_mean) for p in pos_test]) # Positional-mean predictions

        r2 = r2_score(y_test, y_pred) # Calculating R-squared score
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Calculating RMSE

        print(f"Naive R2 score: {r2:.5f}") # Displaying fold R-squared
        print(f"Naive RMSE: {rmse:.5f}") # Displaying fold RMSE
        print("\n" + "="*50) # Results separator

        r2_scores.append(float(r2)) # Storing fold R-squared
        rmse_scores.append(float(rmse)) # Storing fold RMSE

    # Displaying averaged cross-validation results
    print(f"Mean R2 score: {np.mean(r2_scores):.5f} (+/- {np.std(r2_scores):.5f})")
    print(f"Mean RMSE: {np.mean(rmse_scores):.5f} (+/- {np.std(rmse_scores):.5f})")
    print("\n" + "="*50) # Results separator

    return {"dataset": dataset,
            "encoder": "None",
            "regressor": "NaivePositionalMean",
            "mean_r2": np.mean(r2_scores),
            "std_r2": np.std(r2_scores),
            "mean_rmse": np.mean(rmse_scores),
            "std_rmse": np.std(rmse_scores),
            "encode_time": 0.0,
            "train_time": 0.0,
            "folds_r2": r2_scores,
            "folds_rmse": rmse_scores}
