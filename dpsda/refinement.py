import logging
import random
import re
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from tqdm import tqdm
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


from transformers import pipeline
def abstract_texts(
    texts,
    summarizer_model='sshleifer/distilbart-cnn-12-6',
    batch_size=32  
):
    summarizer = pipeline(
        "summarization",
        model=summarizer_model,
        device=0,  
        torch_dtype="auto",  
        batch_size=batch_size  
    )

    summaries = summarizer(
        texts,  
        max_length=100,  
        min_length=10,  
        truncation=True  
    )

    return [summary["summary_text"] for summary in summaries]

def get_synonym(word):
    synonyms = wordnet.synsets(word)
    if synonyms:
        lemmas = [lemma.name().replace("_", " ") for syn in synonyms for lemma in syn.lemmas()]
        lemmas = list(set(lemmas))  # Remove duplicates
        return random.choice(lemmas) if lemmas else None
    return None

def random_substitution(word):
    return "".join(random.sample(word, len(word))) if len(word) > 3 else word

def get_important_words(words):
    pos_tags = pos_tag(words)
    important_words = [word for word, tag in pos_tags if tag.startswith('N') or tag.startswith('V')]
    return important_words

def shuffle_sentence_structure(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)


import random

def semantic_transformations(texts, replace_prob=0.2):
    noisy_texts = []

    for text in texts:
        words = text.split()
        num_changes = max(1, int(len(words) * replace_prob)) 

        important_words = get_important_words(words)
        important_indices = [i for i, w in enumerate(words) if w in important_words]

        for _ in range(num_changes):
            if not words: 
                continue

            if important_indices and random.random() < 0.5:
                idx = random.choice(important_indices)
            else:
                idx = random.randint(0, len(words) - 1)

            word = words[idx]

            # 50% chance to use a synonym, 50% chance for minor scrambling
            if random.random() < 0.5:
                synonym = get_synonym(word)
                if synonym: 
                    words[idx] = synonym
            else:
                words[idx] = random_substitution(word)

        noisy_text = " ".join(words)

        if len(words) > 5 and random.random() < 0.9:
            noisy_text = shuffle_sentence_structure(noisy_text)

        noisy_text = " ".join([w for w in noisy_text.split() if w.lower() not in stop_words])

        noisy_texts.append(noisy_text)

    return noisy_texts

