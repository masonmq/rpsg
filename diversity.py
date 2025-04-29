import tiktoken
import numpy as np
from time import time
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

import torch
import argparse
import csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import load_dataset

from dpsda.logging import *
from utility_eval.compute_mauve import *
from utility_eval.precision_recall import *
from apis.utils import set_seed

import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_from_string(string, encoding):
    try:
        num_tokens = len(encoding.encode(string))
    except:
        num_tokens = 0
    return num_tokens

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_self_bleu(hypothesis, references):
    hypothesis = hypothesis.split()
    references = [ref.split() for ref in references if ref != hypothesis]
    return sentence_bleu(references, hypothesis, smoothing_function=SmoothingFunction().method1)

def calculate_ngram_diversity(sentences, n=2):

    ngrams = []
    
    for sentence in sentences:
        tokens = sentence.split()  
        ngrams.extend(zip(*[tokens[i:] for i in range(n)]))  

    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))
    
    ngram_score = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    return ngram_score


def eval_one_file(syn_fname, 
                  csv_fname,  
                  dataset="yelp", 
                  min_token_threshold=100
                  ):
    syn_data = load_dataset("csv", data_files=syn_fname)

    synthetic_data = []
    if dataset == "yelp":
        for index, d in enumerate(syn_data['train']['text']):
            try:
                if not d.startswith("Business Category: "):
                    synthetic_data.append(d)
            except:
                continue
    elif dataset == "openreview" or dataset == "pubmed" or dataset == "sd":
        for index, d in enumerate(syn_data['train']['text']):

            synthetic_data.append(d)
    else:
        synthetic_data = [d for d in syn_data['train']['text']]
    print("--- syn data len %d  ---" % (len(synthetic_data)))

    all_run_results = []
    for sentence in synthetic_data:
        references = [ref for ref in synthetic_data if ref != sentence]
        hypothesis = sentence.split()
        references = [ref.split() for ref in references]

        score = sentence_bleu(references, hypothesis, smoothing_function=SmoothingFunction().method1)

        all_run_results.append(score)

    average_self_bleu = sum(all_run_results) / len(all_run_results)

    ngram = calculate_ngram_diversity(synthetic_data, n=2)

    print(f"Average Self-BLEU: {average_self_bleu:.4f}")
    print(f"ngram_score: {ngram:.4f}")

    with open(csv_fname, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["round","self_bleu", "n-gram"])
        row_list = [
            round(average_self_bleu, 4),
            round(ngram, 4),
        ]
        writer.writerow([1]+row_list)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--synthetic_file", type=str,
                        default="",
                        required=False)
    parser.add_argument("--synthetic_folder", type=str,
                        default="",
                        required=False)
    parser.add_argument("--synthetic_iteration", type=int,
                        default=20,
                        required=False)
    parser.add_argument("--synthetic_start_iter", type=int,
                        default=1,
                        required=False)
    parser.add_argument("--min_token_threshold", type=int,
                        default=100,
                        required=False)

    parser.add_argument("--dataset", type=str, default="sd",
                        choices=["yelp", "pubmed", "openreview", "sd"],
                        required=False)

    args = parser.parse_args()
    set_seed(seed=0, n_gpu=1)


    for _iter in range(args.synthetic_start_iter, args.synthetic_iteration + 1):
        syn_data_file = os.path.join(
            args.synthetic_folder, str(_iter)+'_all', 'filtered_samples.csv')
        if os.path.isfile(syn_data_file):
            csv_fname = os.path.join(
                args.synthetic_folder, str(_iter)+'_all', 'eval_diversity.csv')
            if os.path.exists(csv_fname):
                print(f'####### csv_fname exist: {csv_fname}')
                continue
            print(f'Processing {csv_fname}')
            eval_one_file(syn_fname=syn_data_file, 
                            csv_fname=csv_fname, 
                            dataset=args.dataset, 
                            min_token_threshold=args.min_token_threshold
                            )
        else:
            print(f"{syn_data_file} does not exist")


if __name__ == "__main__":
    main()
