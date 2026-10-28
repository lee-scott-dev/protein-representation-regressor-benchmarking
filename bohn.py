# Importing necessary libraries and py files
import os # Utilities
import pandas as pd # Data manipulation
import numpy as np # Numerical operations
from Bio.Align import substitution_matrices # Functional score substitutions

# Importing from created .py files
import onehot_encoder # Generates/loads one-hot features

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Defining function to create and save BOHN representation dataset
def generateEmbeddingDataset(dataset, model_name="bohn", verbose=True):
    save_path = f"{dataset}_{model_name}_embeddings.pkl" # Combined output file

    # Checking if combined dataset already generated
    if os.path.exists(save_path):
        print(f"Embedding dataset already exists at {save_path}, loading instead")
        full_df = pd.read_pickle(save_path)
        print("Embeddings loaded\n")
        return full_df

    print("Generating BOHN base dataset...")

    # Getting one-hot variant features (generates if not already cached)
    onehot_path = f"{dataset}_onehot_embeddings.pkl"
    if os.path.exists(onehot_path):
        onehot_df = pd.read_pickle(onehot_path) # Loading existing one-hot features
    else:
        onehot_df = onehot_encoder.generateEmbeddingDataset(dataset, verbose=verbose) # Generating one-hot variant features

    # Existing one-hot feature columns
    onehot_features = [c for c in onehot_df.columns if c.startswith("feature_")]
    n_onehot = len(onehot_features)

    # Building BLOSUM features per-variant
    blosum_rows = []
    for variant in onehot_df["variant"].values:
        wt, mut = variant[0], variant[-1]
        blosum_values = [BLOSUM62[wt, mut]]  # Substitution score for this mutation
        blosum_values += [BLOSUM62[wt, aa] for aa in AMINO_ACIDS]  # BLOSUM row for wild-type residue
        blosum_values += [BLOSUM62[mut, aa] for aa in AMINO_ACIDS]  # BLOSUM row for mutant residue
        blosum_rows.append(blosum_values)

    blosum_array = np.array(blosum_rows)

    # Appending BLOSUM features as new feature columns after the one-hot ones
    combined_df = onehot_df.copy()
    for j in range(blosum_array.shape[1]):
        combined_df[f"feature_{n_onehot + j}"] = blosum_array[:, j]  # Continuing feature_ numbering

    # Reordering so BLOSUM stays at the end
    feature_cols = [c for c in combined_df.columns if c.startswith("feature_")]
    combined_df = combined_df[["variant"] + feature_cols + ["score"]]

    combined_df.to_pickle(save_path)  # Saving combined features
    print(f"Combined one-hot + BLOSUM features saved to {save_path} to create BOHN base (positional means are in-fold)")
    return combined_df