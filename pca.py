# Importing necessary libraries
import pandas as pd # Data manipulation
import matplotlib.pyplot as plt # Plotting
from sklearn.decomposition import PCA # Principle component analysis

# Defining function to plot PCA of ESM embeddings for a dataset, coloured by score
def plotEmbeddingPCA(dataset, protein_language_model):
    full_df = pd.read_pickle(f"{dataset}_{protein_language_model}_embeddings.pkl") # Loading cached embeddings
    feature_columns = [col for col in full_df.columns if col.startswith("feature_")] # Getting feature columns
    known_df = full_df[full_df["score"].notna()] # Getting scored variants
    X = known_df[feature_columns].values
    scores = known_df["score"].values

    pca = PCA(n_components=2) # Setting up PCA
    X_pca = pca.fit_transform(X) # Reducing to 2 dimensions

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=scores, cmap="coolwarm", vmin=-1, vmax=1, s=8, alpha=0.7) # Capped scatter plot
    plt.colorbar(scatter, ax=ax, label="Score")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title(f"{dataset} {protein_language_model} Embedding PCA")
    plt.show()