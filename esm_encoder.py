# Importing necessary libraries and py files
import os # Utilities
import pandas as pd # Plotting
import torch # PyTorch machine learning
import esm # Protein language models

# Importing from created .py files
import data_loader # Loads data

# Defining function to load and set up an ESM model
def loadEncoder(model_name, device, verbose=True, half=False):
    if verbose: print(f"Loading {model_name}...") # Signalling function start

    model, alphabet = getattr(esm.pretrained, model_name)() # Getting model and alphabet
    if half and device.type == "cuda":
        model = model.half() # Loading weights in fp16 to halve GPU memory (CUDA only)
    model = model.to(device) # Moving model parameters to device (GPU if available)
    batch_converter = alphabet.get_batch_converter()
    model.eval() # Disables dropout for deterministic results

    if verbose: print("Model loaded\n") # Signalling function end

    return model, alphabet, batch_converter

# Defining function to generate and save embeddings from sequences, alongside variant and score
def generateEmbeddingDataset(dataset, model_name, model, alphabet, batch_converter, device, batch_size=16, verbose=True):
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
        last_layer = model.num_layers # Defining last layer to extract final representation later
        embeddings = {} # Initialising dictionary to store embeddings

        # Iterating through batches of sequences
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size] # Setting current batch
            batch_labels, batch_strs, batch_tokens = batch_converter(batch) # Converting batch sequences into tokens
            batch_tokens = batch_tokens.to(device) # Moving tokens to device
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1) # Finding true length of each sequence

            # Extracting per-residue representations
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[last_layer])

            token_representations = results["representations"][last_layer] # Taking representation from final layer

            # Generating per-sequence representations via averaging for batch
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1
            for j, label in enumerate(batch_labels):
                # Saving embeddings as plain numpy array
                embeddings[label] = token_representations[j, 1:batch_lens[j] - 1].float().mean(0).cpu().numpy() # CPU averaging over residues, numpy can't operate on GPU tensors

        # Building dataset with variant label, ESM embeddings, and score (NA if missing)
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