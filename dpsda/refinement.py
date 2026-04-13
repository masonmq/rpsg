import numpy as np
import torch
import faiss
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import nltk
nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

from typing import Sequence, List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util
import math


def _norm_bin_label(x) -> int:
    """Return 1 for positive, 0 for negative; anything else -> None (ignored)."""
    if x is None: return None
    s = str(x).strip().lower()
    if s in {"pos","positive","+","1","true"}: return 1
    if s in {"neg","negative","-","0","false"}: return 0
    return None

@torch.inference_mode()
def abstract_texts(
    texts: Sequence[str],
    priv_sentiments: Optional[Sequence[str]] = None,   
    summarizer_model_name: str = "facebook/bart-large-cnn",
    embedding_model_name: str = "sentence-t5-base",
    num_candidates: int = 5,
    max_length: int = 100,
    min_length: int = 10,
    alpha: float = 0.75,           
    min_conf: float = 0.55,         
    attempts: int = 2,              
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    return_diag: bool = True,
    epsilon: Optional[float] = None 
):


    tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    abstr_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to(device)
    embed_model = SentenceTransformer(embedding_model_name, device=device)
    sent_pipe = pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        device=0 if device == "cuda" else -1
    )

    rng = np.random.RandomState(seed)
    out_texts, diags = [], []

 
    N = max(len(texts), 2) 
    if epsilon is None or (isinstance(epsilon, float) and (math.isinf(epsilon) or epsilon <= 0)):
        sigma = 0.0
    else:
        delta = 1.0 / (N * max(math.log(N), 1.0))  
        sigma = math.sqrt(2.0 * math.log(1.25 / max(delta, 1e-12))) * (1.0 / max(float(epsilon), 1e-12))

    def _truncate(t: str) -> str:
        return tokenizer.decode(
            tokenizer(t, return_tensors="pt", max_length=512, truncation=True)["input_ids"][0],
            skip_special_tokens=True
        )

    def _sent_bin(text: str) -> Tuple[int, float]:
        r = sent_pipe(text[:1000])[0]
        lab = 1 if r["label"].upper().startswith("POS") else 0
        return lab, float(r["score"])

    def _gen(src: str, sample: bool) -> List[str]:
        inputs = tokenizer([src]*num_candidates, return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(device)

        gen_kwargs = dict(
            num_return_sequences=num_candidates,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
        )

        if sample:
            gen_kwargs.update(dict(
                do_sample=True,
                top_p=0.92,
                top_k=50,
                temperature=1.05,
                num_beams=1,
                num_beam_groups=1,
                diversity_penalty=0.0,
            ))
            abstr_model.generation_config.num_beams = 1
            abstr_model.generation_config.num_beam_groups = 1
            abstr_model.generation_config.diversity_penalty = 0.0
        else:
            gen_kwargs.update(dict(
                do_sample=False,
                num_beams=max(8, num_candidates),
                num_beam_groups=4,
                diversity_penalty=0.3,
            ))

        outs = abstr_model.generate(**inputs, **gen_kwargs)
        return tokenizer.batch_decode(outs, skip_special_tokens=True)


    def _dp_argmax(scores_vec: np.ndarray, valid_idx: Optional[np.ndarray]) -> int:
        
        if valid_idx is not None and len(valid_idx) > 0:
            considered = scores_vec[valid_idx]
            idx_base = valid_idx
        else:
            considered = scores_vec
            idx_base = np.arange(len(scores_vec))

        if sigma <= 0.0:
            
            best_local = int(np.argmax(considered))
            return int(idx_base[best_local])

        m, M = float(np.min(considered)), float(np.max(considered))
        if M > m:
            norm = (considered - m) / (M - m)
        else:
            norm = np.zeros_like(considered)

        noisy = norm + rng.normal(loc=0.0, scale=sigma, size=norm.shape[0])
        best_local = int(np.argmax(noisy))
        return int(idx_base[best_local])

    for i, t in enumerate(texts):
        target = _norm_bin_label(priv_sentiments[i]) if priv_sentiments is not None else None
        if target == 1:
            t = "Keep positive tone: " + t
        elif target == 0:
            t = "Keep negative tone: " + t

        src = _truncate(t)

        def _pick(cands: List[str]) -> Tuple[str, Dict[str, Any]]:
            emb_src = embed_model.encode(src, convert_to_tensor=True)
            emb_c = embed_model.encode(cands, convert_to_tensor=True)
            sims = util.cos_sim(emb_src, emb_c)[0].cpu().numpy()  # may be [-1,1]

            agree = np.zeros(len(cands), dtype=float)
            confs = np.zeros(len(cands), dtype=float)

            for j, c in enumerate(cands):
                y, p = _sent_bin(c)
                confs[j] = p
                if target is None:
                    agree[j] = 0.5
                else:
                    agree[j] = 1.0 if y == target else 0.0

            score = alpha * sims + (1.0 - alpha) * agree
            if target is not None:
                score = score - 0.15 * (1.0 - agree)

            valid = np.where((agree == 1.0) & (confs >= min_conf))[0] if target is not None else np.array([], dtype=int)

            best_idx = _dp_argmax(score, valid_idx=valid if len(valid) > 0 else None)

            return cands[best_idx], {
                "best_idx": int(best_idx),
                "sim": float(sims[best_idx]),
                "agree": float(agree[best_idx]),
                "conf": float(confs[best_idx]),
                "target": (int(target) if target is not None else None),
                "dp_sigma": float(sigma),
                "dp_used": bool(sigma > 0.0)
            }

        c1 = _gen(src, sample=False)
        best, d = _pick(c1)

        if attempts >= 2 and target is not None and not (d["agree"] == 1.0 and d["conf"] >= min_conf):
            c2 = _gen(src, sample=True)
            best2, d2 = _pick(c2)
            
            if (d2["agree"] == 1.0 and d2["conf"] >= d["conf"]) or d2["sim"] > d["sim"]:
                best, d = best2, d2

        out_texts.append(best)
        if return_diag:
            diags.append(d)

    return (out_texts, diags) if return_diag else out_texts


