import pandas as pd
import re
import os
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import torch
from tqdm import tqdm

nlp_spacy = spacy.load("en_core_web_lg")
model_name = "bigcode/starpii"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1

pii_detector = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device
)

# Updated regex patterns
# SENSITIVE_PATTERNS = {
#     "Email": (r"<[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}>|\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b", "[EMAIL]"),
#     "Phone": (r"\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}", "[PHONE]"),
#     "ZIP": (r"\\b\\d{5}(?:-\\d{4})?\\b", "[ZIPCODE]"),
#     "NameAndID": (r"[^\\n()]{2,100}\\([a-zA-Z0-9_]{3,15}\\)", "[NAME_AND_ID]"),
#     "CamelCaseName": (r"\\b[A-Z][a-z]+[A-Z][a-z]+\\b", "[NAME]"),
#     "TitleName": (r"\\b(?:Dr|Mr|Ms|Mrs|Prof)\\.?\\s+[A-Z][a-z]+\\b", "[NAME]"),
#     "UserID": (r"\\b[a-z]{3}\\d{1,3}\\b", "[USERID]"),
#     "AccountID": (r"\\b[a-z]{2,6}\\d{2,4}_[a-z0-9]+\\b", "[ACCOUNT]"),
#     "DoctorName": (r"\\bdr\\.?\\s?[a-z]+(?:'[a-z]+)?\\b", "[NAME]"),
#     "TildeURL": (r"https?://[^\\s]*~[a-z]{2,20}|www\\.[^\\s]*~[a-z]{2,20}", "[URL_USER]"),
# }

SENSITIVE_PATTERNS = {
    "Email": (
        r"<[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}>|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]"
    ),
    "Phone": (
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "[PHONE]"
    ),
    "ZIP": (
        r"\b\d{5}(?:-\d{4})?\b",
        "[ZIPCODE]"
    ),
    "NameAndID": (
        r"[^\n()]{2,100}\([a-zA-Z0-9_]{3,15}\)",
        "[NAME_AND_ID]"
    ),
    "CamelCaseName": (
        r"\b[A-Z][a-z]+[A-Z][a-z]+\b",
        "[NAME]"
    ),
    "TitleName": (
        r"\b(?:Dr|Mr|Ms|Mrs|Prof)\.?\s+[A-Z][a-z]+\b",
        "[NAME]"
    ),
    "UserID": (
        r"\b[a-z]{3}\d{1,3}\b",
        "[USERID]"
    ),
    "AccountID": (
        r"\b[a-z]{2,6}\d{2,4}_[a-z0-9]+\b",
        "[ACCOUNT]"
    ),
    "DoctorName": (
        r"\bdr\.?\s?[a-z]+(?:'[a-z]+)?\b",
        "[NAME]"
    ),
    "TildeURL": (
        r"https?://[^\s]*~[a-z]{2,20}|www\.[^\s]*~[a-z]{2,20}",
        "[URL_USER]"
    ),
}

CURRENCY_GENERALIZATION = [
    (r"\$([1-9][0-9]{2})", "$100-$999"),
    (r"\$([1-9][0-9]{3})", "$1K-$9K"),
    (r"\$([1-9][0-9]{4})", "$10K-$99K"),
    (r"\$([1-9][0-9]{5})", "$100K-$999K"),
    (r"\$([1-9][0-9]{6})", "$1M+"),
]


def chunk_text(text, chunk_size=512):
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False)
    input_ids = tokens["input_ids"]
    chunks = []
    for start in range(0, len(input_ids), chunk_size):
        end = min(start + chunk_size, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def redact_with_model(text):
    chunks = chunk_text(text)
    redacted_chunks = []
    for chunk in chunks:
        entities = pii_detector(chunk)
        spans = sorted([(e["start"], e["end"], e["entity_group"]) for e in entities], reverse=True)
        for start, end, label in spans:
            chunk = chunk[:start] + f"[{label.upper()}]" + chunk[end:]
        redacted_chunks.append(chunk)
    return " ".join(redacted_chunks)

def apply_spacy_redaction(text):
    doc = nlp_spacy(text)
    for ent in reversed(doc.ents):
        if ent.label_ == "PERSON":
            text = text[:ent.start_char] + "[NAME]" + text[ent.end_char:]
    return text

def apply_regex_redaction(text):
    for _, (pattern, replacement) in SENSITIVE_PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in CURRENCY_GENERALIZATION:
        text = re.sub(pattern, replacement, text)
    return text

def redact_text(text):
    text = redact_with_model(text)
    text = apply_spacy_redaction(text)
    text = apply_regex_redaction(text)
    return text

def batch_redact(text_list):
    return [redact_text(t) for t in text_list]

def process_texts_parallel(text_list, num_workers=4, batch_size=500):
    num_workers = min(cpu_count(), num_workers)
    text_batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
    with Pool(num_workers) as pool:
        results = pool.map(batch_redact, text_batches)
    return [item for sublist in results for item in sublist]

def main(args):
    df = pd.read_csv(args.input_csv)

    text_col = "text" if "text" in df.columns else df.columns[0]
    mask = df[text_col].notna()

    syn_texts = df.loc[mask, text_col].astype(str).tolist()

    print(f"Found {len(syn_texts)} synthetic samples.")
    sanitized_texts = [redact_text(t) for t in tqdm(syn_texts)]

    os.makedirs(args.output_dir, exist_ok=True)

    out_df = df.copy()
    out_df.loc[mask, text_col] = sanitized_texts
    out_df.to_csv(args.output_csv, index=False)

    print(f"Sanitized dataset saved to {args.output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save sanitized output.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    args = parser.parse_args()
    main(args)
