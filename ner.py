import pandas as pd
import spacy
import torch
import re
import os
from multiprocessing import Pool, cpu_count

nlp = spacy.load("en_core_web_sm")


SENSITIVE_PATTERNS = {
    "SSN": (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),  
    "Phone": (r"\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),  
    "ZIP": (r"\b\d{5}(?:-\d{4})?\b", "[ZIPCODE]"),  
    "Currency": (r"\$\d+(?:,\d{3})*(?:\.\d{2})?", "[CURRENCY]"),  
    "Email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]"), 
    "Name": (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NAME]"),  
    "Username": (r"@[A-Za-z0-9_]+", "[USERNAME]"),  
}

CURRENCY_GENERALIZATION = [
    (r"\$([1-9][0-9]{2})", "$100-$999"),  
    (r"\$([1-9][0-9]{3})", "$1K-$9K"),  
    (r"\$([1-9][0-9]{4})", "$10K-$99K"),  
    (r"\$([1-9][0-9]{5})", "$100K-$999K"),  
    (r"\$([1-9][0-9]{6})", "$1M+"),  
]

def apply_stricter_redaction(text):
    
    for pattern_name, (pattern, replacement) in SENSITIVE_PATTERNS.items():
        text = re.sub(pattern, replacement, text)

    for pattern, replacement in CURRENCY_GENERALIZATION:
        text = re.sub(pattern, replacement, text)

    return text


def batch_process_redaction(text_list):
    return [apply_stricter_redaction(text) for text in text_list]


def process_texts_parallel(text_list, num_workers=4, batch_size=500):
    num_workers = min(cpu_count(), num_workers)
    text_batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
    
    with Pool(num_workers) as pool:
        results = pool.map(batch_process_redaction, text_batches)
    
    return [item for sublist in results for item in sublist]


def main(args):
    df = pd.read_csv(args.input_csv)
    if "text" in df.columns:
        syn_texts = df["text"].dropna().tolist()
    else:
        syn_texts = df.iloc[:, 0].dropna().tolist()  

    print(f"Found {len(syn_texts)} synthetic samples.")

    sanitized_texts = process_texts_parallel(syn_texts, num_workers=8, batch_size=1000)

    os.makedirs(args.output_dir, exist_ok=True)

    out_df = pd.DataFrame({"text": sanitized_texts})
    out_df.to_csv(args.output_csv, index=False)

    print(f"Sanitized dataset saved to {args.output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_data_file", type=str, required=True,
    #                     help="Path to the private dataset")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the input CSV file containing synthetic data.")
    parser.add_argument("--output_csv", type=str, default="",
                        help="output csv.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Path to save the ner results.")

    args = parser.parse_args()
    main(args)

