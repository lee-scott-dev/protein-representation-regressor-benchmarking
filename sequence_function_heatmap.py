# Importing necessary libraries and functions
import numpy as np # Numerical operations
import matplotlib.pyplot as plt # Plotting
from matplotlib.colors import ListedColormap # Block colour plotting
import seaborn as sns # Plotting
from Bio.Data.IUPACData import protein_letters # Amino acid letters

amino_acids = list(protein_letters) # Amino acids

# Defining function to plot sequence-function heatmap for a dataset
def plotHeatmap(title, wild_type, scores_df):
    protein_length = len(wild_type) # Getting length of wild type protein
    positions = list(range(1, protein_length + 1)) # Building list of amino acid positions (sequence starts at 1)

    pivot = scores_df.pivot(index="mutant_residue", columns="position", values="score") # Reshaping scores into a grid
    pivot = pivot.reindex(index=amino_acids, columns=positions) # Filling in missing amino acids and positions

    unchanged = np.zeros((len(amino_acids), len(positions)), dtype=bool) # Setting up mask for unchanged amino acids
    for position in positions:
        row_idx = amino_acids.index(wild_type[position - 1]) # Finding wild type residue's row
        unchanged[row_idx, position - 1] = 1 # Marking cell as unchanged
        
    cmap = plt.get_cmap("coolwarm").copy() # Setting colour map
    cmap.set_bad("grey") # Showing missing data as grey
    
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.heatmap(pivot, cmap=cmap, ax=ax, cbar_kws={"label": "Score"}) # Plotting heatmap on each dataset's own score range
    
    # NOTE: Alternative commented version here uses capped colour scale as it can sometimes be difficult to see scores without it,
    # this isn't representative of the effect scale however, and should just be used for overviews
    #sns.heatmap(pivot, cmap=cmap, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Score"})
    ax.imshow(np.where(unchanged, 1, np.nan), extent=(0, len(positions), len(amino_acids), 0), cmap=ListedColormap(["yellow"]), aspect="auto", zorder=2) # Overlaying yellow on unchanged amino acids
    ax.set_title(title) # Setting title
    ax.set_xlabel("Position") # Setting x label
    ax.set_ylabel("Mutant Amino-Acid") # Setting y label
    plt.show()