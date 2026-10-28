# Importing necessary libraries and py files
import os # Utilities
import pandas as pd # Plotting
import torch # PyTorch machine learning
from esm.models.esmc import ESMC # ESM Cambrian (ESM-C)
from esm.sdk.api import ESMProtein, LogitsConfig # ESM-C input

# Importing from created .py files
import data_loader # Loads data

# Defining function to load and set up an ESM-C model
def loadEncoder(model_name, device, verbose=True):
    if verbose: print(f"Loading {model_name}...") # Signalling function start

    model = ESMC.from_pretrained(model_name).to(device) # Getting model and moving to device
    model.eval() # Disables dropout for deterministic results

    if verbose: print("Model loaded\n") # Signalling function end

    return model

# Defining function to generate and save embeddings from sequences, alongside variant and score
def generateEmbeddingDataset(dataset, model_name, model, device, verbose=True):
    save_path = f"{dataset}_{model_name}_embeddings.pkl" # Setting save file name
    wild_type, scores_df = data_loader.loadDataset(dataset, verbose) # Loading scores data

    # Checking if dataset already generated
    if os.path.exists(save_path):
        print(f"Embedding dataset already exists at {save_path}, loading instead") # Signalling skip

        full_df = pd.read_pickle(save_path) # Loading dataframe

        print("Embeddings loaded\n") # Signalling function end

        return full_df

    # Generating embeddings and creating dataset if not
    else:
        print("Generating embedding dataset...") # Signalling generation start

        sequences = data_loader.loadVariantSequences(dataset, verbose) # Getting sequences
        embeddings = {} # Initialising dictionary to store embeddings

        # Iterating through sequences
        for label, sequence in sequences:
            protein = ESMProtein(sequence=sequence) # Setting object for sequence with ESMProtein

            # Extracting per-residue representations
            with torch.no_grad():
                protein_tensor = model.encode(protein) # Encoding sequence to tokens
                logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)) # Getting embeddings
            
            token_representations = logits_output.embeddings # Taking per-residue representations

            # Generating per-sequence representation via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1
            embeddings[label] = token_representations[0, 1:-1].float().mean(0).cpu().numpy() # CPU averaging over residues, as numpy can't operate on GPU tensors

        # Building dataset with variant label, ESM-C embeddings, and score (NA if missing)
        rows = [] # Initialising rows
        for label, embedding in embeddings.items():
            row = {"variant": label} # Initialising rows with variant label
            for i in range(len(embedding)):
                row[f"feature_{i}"] = embedding[i] # Adding feature values
            rows.append(row) # Saving full row
        
        full_df = pd.DataFrame(rows) # Building dataframe
        full_df = full_df.merge(scores_df[["variant", "score"]], on="variant", how="left") # Merging in scores

        full_df.to_pickle(save_path) # Saving embeddings so they don't need to be remade

        print(f"Embeddings generated, loaded, and saved to {save_path}") # Signalling function end
        
        return full_df