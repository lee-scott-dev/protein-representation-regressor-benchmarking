# Importing necessary libraries and py files
import os # Utilities
import pandas as pd # Data manipulation
import numpy as np # Numerical operations

# Importing from created .py files
import data_loader # Loads data

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY") # The 20 standard amino acids

# Defining function to generate and save one-hot features from variants, alongside variant and score
def generateEmbeddingDataset(dataset, verbose=True):
    model_name = "onehot" # Fixed name for this encoding
    save_path = f"{dataset}_{model_name}_embeddings.pkl" # Setting save file name
    wild_type, scores_df = data_loader.loadDataset(dataset, verbose) # Loading scores data

    # Checking if dataset already generated
    if os.path.exists(save_path):
        print(f"Embedding dataset already exists at {save_path}, loading instead") # Signalling skip

        full_df = pd.read_pickle(save_path) # Loading dataframe

        print("Embeddings loaded\n") # Signalling function end

        return full_df

    # Generating features and creating dataset if not
    else:
        print("Generating one-hot feature dataset...") # Signalling generation start

        variants = scores_df["variant"].values # All variant labels

        # Parsing wild-type residue, position, and mutant residue from each variant label
        wt_residues = [v[0] for v in variants] # First character is wild-type residue
        positions = [int(v[1:-1]) for v in variants] # Middle characters are position
        mut_residues = [v[-1] for v in variants] # Last character is mutant residue

        # Building feature space from full dataset
        unique_positions = sorted(set(positions)) # All positions present in the dataset
        position_index = {p: i for i, p in enumerate(unique_positions)} # Mapping position to column index
        residue_index = {a: i for i, a in enumerate(AMINO_ACIDS)} # Mapping residue to column index

        n_positions = len(unique_positions) # Number of position columns
        n_residues = len(AMINO_ACIDS) # Number of residue columns (i.e. 20)
        n_features = n_residues + n_positions + n_residues # WT residue + position + mutant residue

        # Building one-hot feature matrix per variant
        rows = [] # Initialising rows
        for i, variant in enumerate(variants):
            features = np.zeros(n_features) # Initialising zero feature vector

            features[residue_index[wt_residues[i]]] = 1 # One-hot wild-type residue
            features[n_residues + position_index[positions[i]]] = 1 # One-hot position
            features[n_residues + n_positions + residue_index[mut_residues[i]]] = 1 # One-hot mutant residue

            row = {"variant": variant} # Initialising row with variant label
            for j in range(n_features):
                row[f"feature_{j}"] = features[j] # Adding feature values
            rows.append(row) # Saving full row

        full_df = pd.DataFrame(rows) # Building dataframe
        full_df = full_df.merge(scores_df[["variant", "score"]], on="variant", how="left") # Merging in scores

        full_df.to_pickle(save_path) # Saving features so they don't need to be remade

        print(f"One-hot features generated, loaded, and saved to {save_path}") # Signalling function end

        return full_df
