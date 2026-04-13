# add_sentiment_label1.py
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

def build_sentiment_pipeline(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,  
        max_length=512     
    )
    return clf, model

def normalize_to_binary(raw_label: str, model) -> str:
    """
    Convert model output to 'negative' or 'positive'.
    """
    lab = str(raw_label).strip().upper()

    if lab.startswith("LABEL_"):
        try:
            k = int(lab.split("_")[-1])
            mapped = getattr(model.config, "id2label", {}).get(k, lab)
            lab = str(mapped).strip().upper()
        except Exception:
            pass

    if "NEG" in lab:
        return "negative"
    if "POS" in lab:
        return "positive"

    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="label1")
    ap.add_argument("--model", type=str, default="siebert/sentiment-roberta-large-english")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    assert args.text_col in df.columns, f"Missing column: {args.text_col}"

    clf, model = build_sentiment_pipeline(args.model)

    texts = df[args.text_col].astype(str).tolist()
    labels = []

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Sentiment labeling"):
        batch = texts[i:i + args.batch_size]
        preds = clf(batch)  
        for p in preds:
            labels.append(normalize_to_binary(p.get("label", ""), model))

    # Append label1 as the last column
    df[args.label_col] = labels

    unk = (df[args.label_col] == "unknown").sum()
    if unk > 0:
        print(f"[WARN] {unk} rows got label 'unknown'. "
              f"This usually happens if the model returns NEUTRAL or unexpected labels.")

    df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")

if __name__ == "__main__":
    main()
