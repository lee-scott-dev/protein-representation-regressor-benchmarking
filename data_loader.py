# Importing necessary libraries
import pandas as pd # Data manipulation
from Bio import SeqIO # Parsing FASTA files

# Defining function to load a dataset's wild-type sequence and parse its scores dataset
def loadDataset(dataset, verbose=True):
    # NOTE: loadDataset is used as standard for loading scores.csv throughout this project, although
    # not all attributes of the amended scores_df are used all the time, the entire function takes
    # milliseconds to run each time, and keeps things cleaner, which is why things have been done like this

    if verbose: print(f"Loading {dataset} wild-type and scores data...") # Signalling function start

    wild_type_raw = SeqIO.read(f"datasets/{dataset}/wild_type.fasta", "fasta") # Reading wild-type fasta
    wild_type = wild_type_raw.seq # Saving sequence

    scores_df = pd.read_csv(f"datasets/{dataset}/scores.csv") # Reading scores csv
    scores_df["wild_type_residue"] = scores_df["variant"].str[0] # Parsing wild-type residue
    scores_df["mutant_residue"] = scores_df["variant"].str[-1] # Parsing mutant residue
    scores_df["position"] = scores_df["variant"].str[1:-1].astype(int) # Parsing mutation position
    
    if verbose: print("Data loaded\n") # Signalling function end

    return wild_type, scores_df

# Defining function to load variant sequences
def loadVariantSequences(dataset, verbose=True):
    if verbose: print(f"Loading {dataset}'s all variant data...") # Signalling function start

    records = SeqIO.parse(f"datasets/{dataset}/all_variants.fasta", "fasta") # Getting variant sequence records
    sequences = [] # Initialising empty list to store variants and sequences

    # Appending IDs and sequences to list
    for record in records:
        sequences.append((record.id, str(record.seq)))

    if verbose: print("Data loaded\n") # Signalling function end

    return sequences