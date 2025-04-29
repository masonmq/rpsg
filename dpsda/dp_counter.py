import logging
import random
import re
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

def dp_nn_histogram(public_features, private_features, noise_multiplier,
                    num_packing=1, num_nearest_neighbor=1, mode='L2',
                    threshold=0.0):
    assert public_features.shape[0] % num_packing == 0

    num_true_public_features = public_features.shape[0] // num_packing
    if public_features.shape[0] == 0:  # TODO debug, why this case exists
        return np.zeros(shape=num_true_public_features), np.zeros(shape=num_true_public_features)

    faiss_res = faiss.StandardGpuResources()
    if mode == 'L2':
        index = faiss.IndexFlatL2(public_features.shape[1])
    # inner product; need normalization (https://github.com/spotify/annoy)
    elif mode == 'IP':
        index = faiss.IndexFlatIP(public_features.shape[1])
    elif mode == 'cos_sim':
        # normalize the embeddings first
        faiss.normalize_L2(public_features)
        faiss.normalize_L2(private_features)
        index = faiss.IndexFlatIP(public_features.shape[1])
    else:
        raise Exception(f'Unknown mode {mode}')
    if torch.cuda.is_available():
        index = faiss.index_cpu_to_gpu(faiss_res, 0, index)


    index.add(public_features)

    distance, ids = index.search(private_features, k=num_nearest_neighbor)

    counter = Counter(list(ids.flatten()))
    count = np.zeros(shape=num_true_public_features)
    for k in counter:
        count[k % num_true_public_features] += counter[k]
    count = np.asarray(count)
    clean_count = count.copy()
    count += (np.random.normal(size=len(count)) * np.sqrt(num_nearest_neighbor)
              * noise_multiplier)
    count = np.clip(count, a_min=threshold, a_max=None)
    count = count - threshold
    torch.cuda.empty_cache()
    return count, clean_count


