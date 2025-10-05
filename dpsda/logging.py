import logging
import os
import numpy as np
import csv
import json
from dpsda.metrics import calculate_fid, knn_precision_recall_features


def setup_logging(log_file):
    log_formatter = logging.Formatter(
        fmt=('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  '
             '%(message)s'),
        datefmt='%m/%d/%Y %H:%M:%S %p')
    root_logger = logging.getLogger()
    # root_logger.setLevel(logging.DEBUG)
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)


def log_embeddings(embeddings, additional_info, folder, fname=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    savefname = os.path.join(folder, fname+'.embeddings.npz')
    print("save embeddings into", savefname)
    np.savez(
        savefname,
        embeddings=embeddings,
        additional_info=additional_info)


def load_embeddings(path):
    data = np.load(path)
    embeddings = data['embeddings']
    additional_info = data['additional_info']

    return embeddings, additional_info


def log_num_words(fname="num_word_lookahead.csv", all_gen_words=[], all_target_words=[]):
    if len(all_gen_words) == 0 or len(all_target_words) == 0:
        return
    with open(fname, 'w', newline='', encoding="utf-8") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(["target", "gen", "diff"])
        diff_list = []
        diff_abs_list = []
        for i in range(len(all_target_words)):
            try:
                diff_list.append(all_gen_words[i] - all_target_words[i])
                diff_abs_list.append(
                    abs(all_gen_words[i] - all_target_words[i]))
                csv_writer.writerow(
                    [all_target_words[i], all_gen_words[i], all_gen_words[i] - all_target_words[i]])
            except:
                continue
        csv_writer.writerow(["mean_abs", "var_abs", "mean", "var"])
        csv_writer.writerow([np.mean(diff_abs_list), np.std(
            diff_abs_list), np.mean(diff_list), np.std(diff_list)])


def log_prompt_generation(fname="prompt_generation.jsonl", prompts=[], generations=[]):
    new_variants_samples = []
    for x in generations:
        new_variants_samples.extend(x.tolist())

    if len(prompts) == 0 or len(new_variants_samples) == 0:
    #if len(new_variants_samples) == 0:
        return
    with open(fname, "w") as file:
        for i in range(len(prompts)):
        #for i in range(len(new_variants_samples)):
            try:
                json_str = json.dumps(
                        {"prompt": prompts[i], "generation": new_variants_samples[i]})
                file.write(json_str + "\n")
            except:
                continue

def log_generation(fname="real_abstracted.jsonl", abstracted=[], realdata=[]):
    real_samples = []
    new_abstracted = []
    for x in abstracted:
        new_abstracted.extend(x.tolist())
    for x in realdata:
        real_samples.extend(x.tolist())

    if len(new_abstracted) == 0 or len(real_samples) == 0:
        return
    with open(fname, "w") as file:
        for i in range(len(new_abstracted)):
            try:
                json_str = json.dumps(
                        {"abstracted": new_abstracted[i], "real": real_samples[i]})
                file.write(json_str + "\n")
            except:
                continue

def log_count(count, clean_count, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    title = ['type', 'count']
    with open(path, 'w', newline='', encoding="utf-8") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(title)
        csv_writer.writerow(["count", count.tolist()])
        csv_writer.writerow(["clean_count", clean_count.tolist()])


def compute_fid(synthetic_features, all_private_features, feature_extractor, folder='', step=0, log_online=False):

    logging.info(
        f'Computing FID and F1 for syn shape: {synthetic_features.shape}')
    fid = calculate_fid(synthetic_features, all_private_features)
    state = knn_precision_recall_features(
        ref_features=all_private_features, eval_features=synthetic_features,debug=False)
    logging.info(f'fid={fid} F1={state}')
    log_fid(folder, fid, state["f1"],
            state["precision"], state["recall"], step)
    # if log_online:
    #     import wandb
    #     wandb.log({f'metric/fid_{feature_extractor[:10]}': fid, }, step=step)


def log_fid(folder, fid, f1, precision, recall, t, save_fname='fid.csv'):
    with open(os.path.join(folder, save_fname), 'a') as f:
        f.write(f'{t} {fid} {f1} {precision} {recall}\n')


def log_fid_list(folder, fids, t, save_fname='fid.csv'):
    write_list = [t]
    write_list.extend(fids)
    with open(os.path.join(folder, save_fname), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(write_list)


def log_samples(samples, folder, additional_info=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    all_data = []
    first_tag = None

    for i, seq in enumerate(samples):
        if not seq:
            continue

        # normalize whitespace in the text
        seq_clean = " ".join(str(seq).split())

        row = [seq_clean]
        tag = ""

        if additional_info is not None:
            labels = additional_info[i]
            parts = labels.strip().split("\t") if isinstance(labels, str) else [str(labels)]
            tag = parts[0].lower() if parts else ""
            if "pubmed" in tag:
                row = [seq_clean]  # only save text
            elif "sd" in tag:
                label1 = parts[1] if len(parts) > 1 else ""
                # here label2 is the private length (already precomputed & stored as 3rd part)
                label2 = parts[2] if len(parts) > 2 else ""
                row = [seq_clean, label1, label2]
            else:
                row = [seq_clean] + parts

            if first_tag is None:
                first_tag = tag
        else:
            # no additional info provided, just keep the text
            row = [seq_clean]
            if first_tag is None:
                first_tag = ""

        all_data.append(row)

    # ----- headers -----
    if first_tag == "pubmed":
        title = ["text"]
    elif first_tag == "sd":
        # for synthetic sd data we include sentiment + label2
        title = ["text", "label1", "label2"]
    elif additional_info is not None:
        # generic case: just dump one 'label' column
        title = ["text", "label"]
    else:
        # nothing extra
        title = ["text"]

    try:
        with open(os.path.join(folder, "samples.csv"), "w", newline="", encoding="utf-8") as wf:
            writer = csv.writer(wf)
            writer.writerow(title)
            for row in all_data:
                cleaned = [
                    x.encode("utf-8", "replace").decode("utf-8") if isinstance(x, str) else x
                    for x in row
                ]
                writer.writerow(cleaned)
    except Exception as e:
        with open(os.path.join(folder, "samples.csv"), "w", newline="", encoding="utf-8") as wf:
            writer = csv.writer(wf, quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")
            writer.writerow(title)
            for row in all_data:
                cleaned = [
                    x.encode("utf-8", "replace").decode("utf-8") if isinstance(x, str) else x
                    for x in row
                ]
                writer.writerow(cleaned)

    return all_data


