import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--all_scores_sd', type=str)
parser.add_argument('--all_scores', type=str)
args = parser.parse_args()

with open(args.all_scores_sd, 'rb') as file:
    all_scores_sd = pickle.load(file)

with open(args.all_scores, 'rb') as file:
    all_scores = pickle.load(file)

original_scores_sd = []
neighbor_scores_sd = []

for entry in all_scores_sd:
    for key, value in entry.items():
        if '<original_text>' in key:
            original_scores_sd.append(value)  
        else:
            neighbor_scores_sd.append(value)

original_scores = []
neighbor_scores = []

for entry in all_scores:
    for key, value in entry.items():
        if '<original_text>' in key:
            original_scores.append(value)
        else:
            neighbor_scores.append(value)

ppl_original = np.exp(-np.array(original_scores_sd)).mean()

ppl_neighbors = np.exp(-np.array(neighbor_scores_sd)).mean()

likelihood_ratios = []

for orig, neigh in zip(original_scores_sd, neighbor_scores_sd):
    ratio = orig - neigh
    likelihood_ratios.append(ratio)

threshold = -5  # Example threshold
true_positive = sum(1 for r in likelihood_ratios if r > threshold)
false_positive = sum(1 for r in likelihood_ratios if r <= threshold)

y_true = [1] * len(original_scores_sd) + [0] * len(neighbor_scores_sd)
y_score = original_scores_sd + neighbor_scores_sd

auc = roc_auc_score(y_true, y_score)


# PPL-based membership inference, AUC (PPL):
ppl_original_scores = np.exp(-np.array(original_scores_sd))
ppl_neighbor_scores = np.exp(-np.array(neighbor_scores_sd))

y_true_ppl = [1] * len(ppl_original_scores) + [0] * len(ppl_neighbor_scores)
y_score_ppl = list(ppl_original_scores) + list(ppl_neighbor_scores)

auc_ppl = roc_auc_score(y_true_ppl, y_score_ppl) * 100
print(f"AUC (PPL): {auc_ppl:.2f}")


# REFER-based membership inference, AUC (REFER):
refer_scores = [orig - neigh for orig, neigh in zip(original_scores_sd, neighbor_scores_sd)]

y_true_refer = [1] * len(refer_scores) + [0] * len(refer_scores)
y_score_refer = refer_scores + [-r for r in refer_scores] 

auc_refer = roc_auc_score(y_true_refer, y_score_refer) * 100
print(f"AUC (REFER): {auc_refer:.2f}")

# LIRA-based membership inference AUC
y_true_lira = [1] * len(likelihood_ratios) + [0] * len(likelihood_ratios)
y_score_lira = likelihood_ratios + [1 / lr for lr in likelihood_ratios] 

auc_lira = roc_auc_score(y_true_lira, y_score_lira) * 100
print(f"AUC (LIRA): {auc_lira:.2f}")


likelihood_ratios = [
    np.exp(-orig) / np.exp(-neigh) for orig, neigh in zip(original_scores_sd, neighbor_scores_sd)
]




