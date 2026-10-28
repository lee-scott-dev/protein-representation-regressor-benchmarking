# Importing necessary libraries and py files
from Bio.Align import substitution_matrices # BLOSUM substitution matrices
from scipy.stats import spearmanr # Rank correlation

# Importing from created .py files
import data_loader # Loads data

BLOSUM62 = substitution_matrices.load("BLOSUM62") # Loading BLOSUM62 substitution matrix

# Defining function to assess the BLOSUM substitution scores against functional scores
def assessBlosum(dataset, verbose=False):
    wild_type, scores_df = data_loader.loadDataset(dataset, verbose) # Loading scores data

    known_df = scores_df[scores_df["score"].notna()].copy() # Known variants with scores

    # Getting the BLOSUM substitution score for each variant
    known_df["blosum"] = known_df["variant"].apply(lambda v: BLOSUM62[v[0], v[-1]]) # Score for WT to mutant

    y_true = known_df["score"].values # Actual functional scores
    blosum_scores = known_df["blosum"].values # BLOSUM substitution scores

    # Spearman rank correlation (as BLOSUM and functional scores sit on different scales)
    rho, p_value = spearmanr(blosum_scores, y_true)

    # Displaying statistics
    print(f"BLOSUM substitution vs functional score ({dataset}):")
    print(f"Spearman rho: {rho:.5f}")
    print(f"p-value: {p_value:.3e}")

    return rho, p_value