import numpy as np
import logging
import collections
import csv
from datasets import load_dataset
from transformers import AutoTokenizer

def sample_dataset(data_name, dataset, label_column_name='label1', sample_size=5000, subsample_one_class=False, random_seed=123):
    if subsample_one_class == False and sample_size < 0:
        return dataset
    training_dataset = dataset['train']
    sample_indices = []
    np.random.seed(random_seed)
    if subsample_one_class:
        if data_name == "yelp":
            label1 = 'Business Category: Restaurants'
            label2 = 'Review Stars: 5.0'
            indices = np.where((np.array(training_dataset['label1']) == label1) & (
                np.array(training_dataset['label2']) == label2))[0]
        elif data_name == "openreview":
            area = "Area: Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)"
            recommendation = "Recommendation: 8: accept, good paper"
            indices = np.where((np.array(training_dataset['label1']) == area) & (
                np.array(training_dataset['label2']) == recommendation))[0]
            logging.info(f'indices {len(indices)}')
            if sample_size < 0:
                sample_indices = indices
            else:
                sample_indices = np.random.choice(
                    indices, size=sample_size, replace=False)
                np.random.shuffle(sample_indices)
        elif data_name == "pubmed":
            indices = list(range(len(training_dataset)))
        elif data_name == "sd":
            indices = list(range(len(training_dataset)))
        else:
            raise ValueError(f'Unknown dataset name {dataset}')
        if sample_size < 0:
            sample_indices = indices
        else:
            sample_indices = np.random.choice(
                indices, size=sample_size, replace=False)
            np.random.shuffle(sample_indices)
    else:
        if data_name == "pubmed" or data_name == "openreview" or data_name == "sd" :  # random sample
            indices = list(range(len(training_dataset)))
            sample_indices = np.random.choice(
                indices, size=sample_size, replace=False)
            np.random.shuffle(sample_indices)
        else:  # random sample based on label
            label_list = training_dataset.unique(label_column_name)
            for label in label_list:
                indices = np.where(
                    np.array(training_dataset[label_column_name]) == label)[0]
                sample_num = round(
                    sample_size * (len(indices)/len(training_dataset)))
                sample_indices.append(np.random.choice(
                    indices, size=sample_num, replace=False))
            sample_indices = np.concatenate(sample_indices)
            np.random.shuffle(sample_indices)
    print(f"sample_indices : {sample_indices}")
    training_dataset = training_dataset.select(sample_indices)
    dataset['train'] = training_dataset
    return dataset


def load_dataset_with_special(data_file, gen):
    if gen:
        try:  # in case there are some special characters in the text
            raw_datasets = load_dataset(
                "csv", data_files=data_file, quoting=csv.QUOTE_NONE,  quotechar='', escapechar='\\')
        except:
            raw_datasets = load_dataset("csv", data_files=data_file)
    else:
        raw_datasets = load_dataset("csv", data_files=data_file)
    return raw_datasets


def load_data(dataset="sd", data_file="data/sd/train.csv", num_samples=-1, subsample_one_class=False, gen=False, max_token_length=0):
    print("data_file", data_file)
    if dataset == "yelp":
        prompt_counter = collections.Counter()
        raw_datasets = load_dataset_with_special(data_file, gen)
        original_data = sample_dataset(dataset, raw_datasets, label_column_name='label1',
                                       sample_size=num_samples, subsample_one_class=subsample_one_class)
        prompt_idexer = dict()

        label_column_index = ['label1', 'label2']
        for i, line in enumerate(original_data['train']):
            prompt = "\t".join([line[idx] for idx in label_column_index])
            prompt_counter[prompt] += 1

            if prompt not in prompt_idexer.keys():
                prompt_idexer[prompt] = [i]
            else:
                prompt_idexer[prompt].append(i)

        train_data = [d for d in original_data['train']['text']]
        train_labels = ["\t".join([line[idx] for idx in label_column_index])
                        for line in original_data['train']]

        return train_data, train_labels, prompt_counter, prompt_idexer
    elif dataset == "openreview":

        prompt_counter = collections.Counter()

        raw_datasets = load_dataset_with_special(data_file, gen)
        original_data = sample_dataset(dataset, raw_datasets, label_column_name='label2',
                                       sample_size=num_samples, subsample_one_class=subsample_one_class)
        prompt_idexer = dict()

        train_data = []
        train_labels = []
        for i, line in enumerate(original_data['train']):
            prompt = f"{line['label1']}\t{line['label2']}"
            prompt_counter[prompt] += 1
            if prompt not in prompt_idexer.keys():
                prompt_idexer[prompt] = [i]
            else:
                prompt_idexer[prompt].append(i)
            train_data.append(line['text'])
            train_labels.append(prompt)
        return train_data, train_labels, prompt_counter, prompt_idexer
    elif dataset == "pubmed":
        prompt_counter = collections.Counter()
        raw_datasets = load_dataset_with_special(data_file, gen)
        original_data = sample_dataset(dataset, raw_datasets, label_column_name='',
                                       sample_size=num_samples, subsample_one_class=subsample_one_class,random_seed=123)
        prompt_idexer = dict()
        train_data = []
        train_labels = []

        # for i, line in enumerate(original_data['train']):
        #     text = line['text']
        #     tokens = tokenizer(text, truncation=False)["input_ids"]  # Tokenize without truncation

        #     # Split tokens into overlapping chunks
        #     for j in range(0, len(tokens), max_token_length - stride):
        #         chunk = tokens[j:j + max_token_length]  # Extract chunk
        #         if len(chunk) == 0:
        #             continue

        #         # Decode tokens back to text for storage
        #         chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        #         train_data.append(chunk_text)
        #         train_labels.append("pubmed")

        #         # Update prompt indexer and counter
        #         prompt_counter["pubmed"] += 1
        #         if "pubmed" not in prompt_idexer:
        #             prompt_idexer["pubmed"] = [i]
        #         else:
        #             prompt_idexer["pubmed"].append(i)

        for i, line in enumerate(original_data['train']):
            prompt = f"pubmed"
            prompt_counter[prompt] += 1

            if prompt not in prompt_idexer.keys():
                prompt_idexer[prompt] = [i]
            else:
                prompt_idexer[prompt].append(i)

            #===========================
            # if use private seeds, need to control the token length for gpt2 which max_token_length = 1024
            if max_token_length > 0:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with your model
                # Truncate the text to fit the max_length token limit
                tokenized = tokenizer(line['text'], truncation=True, max_length=max_token_length)
                truncated_text = tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True)
                # Append the truncated text and label
                train_data.append(truncated_text)
            #===========================
            else:
                train_data.append(line['text'])
            train_labels.append(prompt)

        return train_data, train_labels, prompt_counter, prompt_idexer
    elif dataset == "sd":
        prompt_counter = collections.Counter()
        raw_datasets = load_dataset_with_special(data_file, gen)

        # Expect CSV columns: text,label1
        original_data = sample_dataset(
            dataset,
            raw_datasets,
            label_column_name='label1',   # important: we’re using the sentiment column
            sample_size=num_samples,
            subsample_one_class=subsample_one_class,
            random_seed=123
        )

        prompt_idexer = dict()
        train_data = []
        train_labels = []

        for i, line in enumerate(original_data['train']):
            sentiment = str(line['label1']).strip().lower()          # "positive"/"negative"
            text = line['text']
            wc = len(text.split())
            target_words = max(1, round(0.8 * wc))                   # your “a bit shorter” rule

            prompt_counter[sentiment] += 1
            prompt_idexer.setdefault(sentiment, []).append(i)

            train_data.append(text)                                   # private text
            # PACK everything into additional_info (keeps framework)
            train_labels.append(f"sd\t{sentiment}\t{target_words}")


        return train_data, train_labels, prompt_counter, prompt_idexer
