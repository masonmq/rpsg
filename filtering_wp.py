import torch
import pandas as pd
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
#import faiss
from sklearn.metrics.pairwise import cosine_similarity
import collections

def get_logprobs(pairs, model, tokenizer, device, batch_size=32):
    model.eval()
    all_log_probs = []

    for i in range(0, len(pairs), batch_size):
        batch_texts = [t[0] for t in pairs[i:i + batch_size]]
        batch_refs = [t[1] for t in pairs[i:i + batch_size]]

        encoding = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(input_ids=encoding.input_ids,
                            attention_mask=encoding.attention_mask,
                            labels=encoding.input_ids)
            loss = outputs.loss
            if loss.numel() == 1:
                losses = [loss.item()] * len(batch_texts)
            else:
                losses = loss.tolist()

        log_probs = [-l for l in losses]
        all_log_probs.extend(log_probs)

    return all_log_probs


def get_embedding_batched(texts, model, tokenizer, device, batch_size=64):
    all_embeddings = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoding = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.bert(**encoding)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(emb)

    return np.vstack(all_embeddings)



def main(args):
    num_bins = 250
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForMaskedLM.from_pretrained(args.model_dir).to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)

    df_synthetic = pd.read_csv(args.input_csv)
    synthetic_texts = df_synthetic["text"].tolist()

    df_private = pd.read_csv(args.train_data_file)
    private_texts = df_private["text"].tolist()

    # Step 1: Compute log probs for private data and select 250 representatives evenly
    private_log_probs = np.array([
        get_logprobs([(text, text)], model, tokenizer, device, args.batch_size)[0]
        for text in tqdm(private_texts, desc="Private LogProbs")
    ])
    
    sorted_indices = np.argsort(private_log_probs)
    bin_size = len(private_texts) // num_bins
    
    representative_private_texts = []
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < 249 else len(private_texts)
        bin_indices = sorted_indices[start:end]
        bin_log_probs = private_log_probs[bin_indices]
    
        # Pick the index closest to the mean in this bin
        mean_val = np.mean(bin_log_probs)
        chosen_idx_in_bin = np.argmin(np.abs(bin_log_probs - mean_val))
        chosen_idx = bin_indices[chosen_idx_in_bin]
        representative_private_texts.append(private_texts[chosen_idx])

    print(f"====================== Selected {len(representative_private_texts)} representative private data points.")

    # Step 2: Compute cosine similarity between synthetic and private embeddings
    synthetic_emb = get_embedding_batched(synthetic_texts, model, tokenizer, device, batch_size=64)
    private_emb = get_embedding_batched(private_texts, model, tokenizer, device, batch_size=64)

    similarity_matrix = cosine_similarity(synthetic_emb, private_emb)  
    max_similarities = similarity_matrix.max(axis=1)  

    retain_fraction = 0.6 
    
    retain_count = int(len(synthetic_texts) * retain_fraction)
    bottom_percent_idx = max_similarities.argsort()[:retain_count]
    filtered_synthetic_texts = [synthetic_texts[i] for i in bottom_percent_idx]

    print(f"===================== Retained {len(filtered_synthetic_texts)} synthetic points after embedding filtering.")

    # Step 3: Compute log probabilities in batch (final comparison step)
    final_log_probs = []
    for i in tqdm(range(0, len(filtered_synthetic_texts), args.batch_size), desc="Final LogProbs"):
        batch = filtered_synthetic_texts[i:i + args.batch_size]
        batch_scores = []
        for syn_text in batch:
            pairs = [(syn_text, priv_text) for priv_text in representative_private_texts]
            log_probs = get_logprobs(pairs, model, tokenizer, device, args.batch_size)
            batch_scores.append(max(log_probs))
        final_log_probs.extend(batch_scores)

    # Step 4: Final filtering (retain the bottom%)
    final_log_probs = np.array(final_log_probs)
    cutoff = np.percentile(final_log_probs, 50)
    final_filtered_texts = [text for text, lp in zip(filtered_synthetic_texts, final_log_probs) if lp <= cutoff]

    print(f"=========================Retained {len(final_filtered_texts)} synthetic points after final log filtering.")
    pd.DataFrame({"text": final_filtered_texts}).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Efficient Hybrid Filtering Pipeline with LogProbs")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args)