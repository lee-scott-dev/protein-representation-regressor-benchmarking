# Importing necessary libraries
import matplotlib.pyplot as plt # Plotting
import seaborn as sns # Plotting
from Bio.Data.IUPACData import protein_letters # Amino acid letters

amino_acids = list(protein_letters) # Amino acids

# Defining function to print summary statistics for a dataset
def printSummary(dataset, wild_type, scores_df):
    protein_length = len(wild_type) # Getting length of wild type protein
    present_count = len(scores_df) # Calculating number of mutations scored
    n_positions = scores_df["position"].nunique() # Calculating number of positions scored
    expected_count = protein_length * 19 # Calculating expected number of mutations (based on 19 other amino acids)
    coverage = 100 * present_count / expected_count

    single_variant_positions = (scores_df.groupby("position")["score"].count() == 1).sum() # Positions with only one scored variant (no within-position SD)

    print(f"Dataset: {dataset}")
    print(f"Protein length: {protein_length}")
    print(f"Positions scored: {n_positions} / {protein_length}")
    print(f"Coverage: {coverage:.2f}% ({present_count}/{expected_count})")
    print(f"Positions with a single scored variant: {single_variant_positions}")
    print(scores_df["score"].describe())

# Defining function to identify and print outlier scores for a dataset
def printOutliers(dataset, scores_df):
    scores = scores_df["score"]

    Q1 = scores.quantile(0.25) # Calculating first quartile
    Q3 = scores.quantile(0.75) # Calculating third quartile
    IQR = Q3 - Q1 # Calculating interquartile range
    lower_fence = Q1 - 1.5 * IQR # Calculating lower outlier fence
    upper_fence = Q3 + 1.5 * IQR # Calculating upper outlier fence

    outliers = scores_df[(scores < lower_fence) | (scores > upper_fence)] # Identifying outliers

    print(f"\n{dataset} outliers: {len(outliers)} - IQR fences: {lower_fence:.4f} to {upper_fence:.4f}:")
    for _, row in outliers.sort_values("score", ascending=False).iterrows():
        print(f"{row['variant']} (Effect: {row['score']:.4f})")

# Defining function to plot score distribution and score by mutant residue for a dataset
def plotScores(dataset, scores_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.hist(scores_df["score"], bins=75, edgecolor="black") # Scores histogram
    ax1.set_title(f"{dataset} Effect Distribution") # Setting title
    ax1.set_xlabel("Effect") # Setting x label
    ax1.set_ylabel("Count") # Setting y label

    sns.boxplot(data=scores_df, x="mutant_residue", y="score", order=amino_acids, ax=ax2) # Mutant scores boxplot
    ax2.set_title(f"{dataset} Effect by Mutant Amino-Acid") # Setting title
    ax2.set_xlabel("Mutant Amino-Acid") # Setting x label
    ax2.set_ylabel("Effect") # Setting y label

    plt.tight_layout()
    plt.show()

# Defining function to plot mean score per wild-type to mutant amino acid pair
def plotSubstitutionHeatmap(dataset, scores_df):
    pivot = scores_df.pivot_table(index="wild_type_residue", columns="mutant_residue", values="score", aggfunc="mean") # Averaging scores
    pivot = pivot.reindex(index=amino_acids, columns=amino_acids) # Setting ordering

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white") # Self-substitutions and missing pairs shown as white

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap=cmap, center=0, ax=ax, cbar_kws={"label": "Mean Effect"})
    ax.set_title(f"{dataset} Substitution Heat-Map")
    ax.set_xlabel("Mutant Amino-Acid")
    ax.set_ylabel("Wild-Type Amino-Acid")
    plt.tight_layout()
    plt.show()

# Defining function to plot dataset vs per-position score standard deviation for a dataset
def plotStandardDeviations(dataset, scores_df):
    overall_std = scores_df["score"].std() # Calculating overall dataset standard deviation
    std_per_position = scores_df.groupby("position")["score"].std().dropna() # Calculating standard deviation per position (positions with one variant give NaN, so are dropped)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.hist(std_per_position.values, bins=50, edgecolor="black") # Distribution of per-position stds
    ax1.axvline(overall_std, color="black", label=f"Overall SD={overall_std:.4f}")
    ax1.axvline(std_per_position.mean(), color="orange", label=f"Mean per-Position SD = {std_per_position.mean():.4f}")
    ax1.set_xlabel("SD")
    ax1.set_ylabel("Count")
    ax1.set_title(f"{dataset} Distribution of Per-Position Effect SDs ({len(std_per_position)} positions)")
    ax1.legend()

    ax2.bar(std_per_position.index, std_per_position.values) # Per-position standard deviations
    ax2.axhline(overall_std, color="black", label=f"Overall SD = {overall_std:.4f}")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("SD")
    ax2.set_title(f"{dataset} Per-Position Effect SD")
    ax2.legend()

    plt.tight_layout()
    plt.show()