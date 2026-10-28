# Importing necessary libraries and py files
import os # Utilities
import re # Regular expressions for sequence cleaning
import pandas as pd # Data manipulation
import torch # PyTorch machine learning
from transformers import T5Tokenizer, T5EncoderModel # ProtT5 model and tokenizer

# Importing from created .py files
import data_loader # Loads data

# Defining function to load and set up a ProtT5 model
def loadEncoder(model_name, device, verbose=True):
    if verbose: print(f"Loading {model_name}...") # Signalling function start

    hf_name = f"Rostlab/{model_name}" # Prepending Rostlab org for the HuggingFace path
    tokenizer = T5Tokenizer.from_pretrained(hf_name, do_lower_case=False) # Loading tokenizer
    model = T5EncoderModel.from_pretrained(hf_name) # Loading encoder-only model

    if device.type == "cpu":
        model = model.to(torch.float32) # Full precision on CPU (half only supported on GPU)
    model = model.to(device) # Moving model to device
    model.eval() # Disables dropout for deterministic results

    if verbose: print("Model loaded") # Signalling function end

    return model, tokenizer

# Defining function to generate and save embeddings from sequences, alongside variant and score
def generateEmbeddingDataset(dataset, model_name, model, tokenizer, device, verbose=True):
    save_path = f"{dataset}_{model_name}_embeddings.pkl" # Setting save file name
    wild_type, scores_df = data_loader.loadDataset(dataset, verbose) # Loading scores data

    # Checking if dataset already generated
    if os.path.exists(save_path):
        print(f"Embedding dataset already exists at {save_path}, loading instead") # Signalling skip

        full_df = pd.read_pickle(save_path) # Loading dataframe

        print("Embeddings loaded") # Signalling function end

        return full_df

    # Generating embeddings and creating dataset if not
    else:
        print("Generating embedding dataset...") # Signalling generation start

        sequences = data_loader.loadVariantSequences(dataset, verbose) # Getting sequences
        embeddings = {} # Initialising dictionary to store embeddings

        # Iterating through sequences
        for label, sequence in sequences:
            seq_len = len(sequence) # True residue length before adding spaces/tokens
            clean_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence))) # Mapping rare residues to X and spacing residues

            # Tokenising and encoding sequence
            ids = tokenizer(clean_seq, add_special_tokens=True, return_tensors="pt") # Tokenising sequence
            input_ids = ids["input_ids"].to(device) # Moving token ids to device
            attention_mask = ids["attention_mask"].to(device) # Moving attention mask to device

            # Extracting per-residue representations
            with torch.no_grad():
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask) # Running model

            # Generating per-sequence representation via averaging
            # NOTE: ProtT5 adds an end-of-sequence token at the end, so real residues are [:seq_len]
            embeddings[label] = embedding_repr.last_hidden_state[0, :seq_len].float().mean(0).cpu().numpy() # CPU averaging over residues

        # Building dataset with variant label, ProtT5 embeddings, and score (NA if missing)
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