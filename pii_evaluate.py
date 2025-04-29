import argparse
import os
import pandas as pd
import torch
from transformers import pipeline
from huggingface_hub import login
import torchvision

torchvision.disable_beta_transforms_warning()

HF_TOKEN = "hf_your_token_here"  # Replace with your Hugging Face token
login(token=HF_TOKEN)

def load_ner_model(token):
    return pipeline(
        "token-classification",
        model="bigcode/starpii",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
        token=token,
    )

def load_dataset(file_path):
    return pd.read_csv(file_path)

def chunk_text(text, max_tokens=100, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + max_tokens]))
        i += max_tokens - overlap
    return chunks

def evaluate_pii_extraction(dataset, ner_classifier, csv_fname):
    total_successful_extractions = 0
    total_pii_count = 0

    for text in dataset["text"]:
        extracted_pii = []
        chunks = chunk_text(text)

        for chunk in chunks:
            ner_results = ner_classifier(chunk)
            extracted_pii.extend([res["word"] for res in ner_results])

        if extracted_pii:
            total_successful_extractions += 1

    results = {
        "round": 1,
        "successful_extractions": total_successful_extractions,
    }

    if os.path.isfile(csv_fname):
        df_results = pd.read_csv(csv_fname)
        df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index=True)
    else:
        df_results = pd.DataFrame([results])

    df_results.to_csv(csv_fname, index=False)

    print(f"Results saved to {csv_fname}")
    print(f"Total successful extractions: {total_successful_extractions}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    csv_fname = os.path.join(args.output_dir, "eval_pii.csv")

    ner_classifier = load_ner_model(HF_TOKEN)
    dataset = load_dataset(args.synthetic_file)

    evaluate_pii_extraction(dataset, ner_classifier, csv_fname)

if __name__ == "__main__":
    main()
