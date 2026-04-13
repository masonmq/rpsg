import argparse
import os
import math
import pandas as pd
import torch
from transformers import pipeline
import torchvision

torchvision.disable_beta_transforms_warning()

def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        device=0 if torch.cuda.is_available() else -1
    )

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    assert "text" in df.columns, f"'text' column missing in {file_path}"
    return df

def _normalize_sentiment_label(raw_label, classifier):
    up = str(raw_label).strip().upper()
    if up.startswith("LABEL_"):
        idx = up.split("_")[-1]
        id2label = getattr(classifier.model.config, "id2label", {}) or {}
        mapped = None
        if idx in id2label:
            mapped = id2label[idx]
        else:
            try:
                mapped = id2label[int(idx)]
            except Exception:
                mapped = None
        if mapped:
            return str(mapped).strip().upper()
        fallback = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE", "LABEL_2": "NEUTRAL"}
        return fallback.get(up, up)
    return up

from collections import Counter

def classify_long_text(text, clf, max_chars=512, stride=400):

    if text is None or str(text).strip() == "":
        return "ERROR"

    t = str(text)
    chunks = []
    i = 0
    while i < len(t):
        chunk = t[i:i + max_chars]
        if not chunk:
            break
        chunks.append(chunk)
        if i + max_chars >= len(t):
            break
        i += stride

    try:
        results = clf(chunks)  
    except Exception:
        return "ERROR"

    labels_up = []
    for r in results:
        try:
            raw = r["label"]
            labels_up.append(_normalize_sentiment_label(raw, clf))
        except Exception:
            labels_up.append("ERROR")

    vote_pool = [l for l in labels_up if l != "ERROR"]
    if not vote_pool:
        return "ERROR"

    counts = Counter(vote_pool).most_common()
    if len(counts) >= 2 and counts[0][1] == counts[1][1]:
        return vote_pool[-1]
    return counts[0][0]

def _classify_series(text_series, classifier, batch_size=32, max_chars=512):

    return text_series.apply(lambda x: classify_long_text(x, classifier, max_chars=max_chars, stride=400))

def evaluate_sentiment_consistency(private_df, synthetic_df, sentiment_classifier, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary_path   = os.path.join(output_dir, "eval_sentiment.csv")
    detailed_path  = os.path.join(output_dir, "eval_sentiment_detailed.csv")
    synth_out_path = os.path.join(output_dir, "synthetic_with_label1.csv")

    # 1: Check private_df has label1 (ground truth)
    assert "label1" in private_df.columns, "private_file must contain a 'label1' column with ground-truth sentiment."

    # 2: Predict sentiment for synthetic texts and add as label1
    synthetic_df = synthetic_df.copy()
    synthetic_df["label1"] = _classify_series(synthetic_df["text"], sentiment_classifier)

    # Persist the synthetic file with predicted labels for downstream use
    synthetic_df.to_csv(synth_out_path, index=False)

    # 3: Compare -- 1:1 row alignment between private and synthetic
    assert len(private_df) == len(synthetic_df), "Datasets must have 1:1 mapping (same number of rows)."

    priv_labels  = private_df["label1"].astype(str).str.upper()
    synth_labels = synthetic_df["label1"].astype(str).str.upper()

    detailed = pd.DataFrame({
        "private_text":   private_df["text"].astype(str),
        "private_label1": priv_labels,
        "synthetic_text": synthetic_df["text"].astype(str),
        "synthetic_label1": synth_labels
    })
    detailed["match"] = (detailed["private_label1"] == detailed["synthetic_label1"])

    valid = (detailed["private_label1"] != "ERROR") & (detailed["synthetic_label1"] != "ERROR")
    total_pairs  = int(valid.sum())
    match_count  = int((detailed["match"] & valid).sum())
    match_rate   = round(100.0 * match_count / total_pairs, 2) if total_pairs else 0.0

    detailed.to_csv(detailed_path, index=False)

    summary_row = {
        "round": 1,
        "total_pairs": total_pairs,
        "match_count": match_count,
        "match_rate": match_rate
    }
    if os.path.isfile(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = pd.concat([df_summary, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([summary_row])
    df_summary.to_csv(summary_path, index=False)

    print(f"\nSynthetic with predicted labels saved to: {synth_out_path}")
    print(f"Detailed results saved to: {detailed_path}")
    print(f"Summary saved to:         {summary_path}")
    print(f"Sentiment alignment rate: {match_rate}% "
          f"(on {total_pairs} valid pairs; 'ERROR' rows excluded)\n")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private_file", type=str, required=True, help="CSV with columns: text,label1 (ground truth)")
    parser.add_argument("--synthetic_file", type=str, required=True, help="CSV with column: text (no label1); this script will add label1")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    sentiment_classifier = load_sentiment_model()
    private_df   = load_dataset(args.private_file)
    synthetic_df = load_dataset(args.synthetic_file)

    evaluate_sentiment_consistency(private_df, synthetic_df, sentiment_classifier, args.output_dir)

if __name__ == "__main__":
    main()
