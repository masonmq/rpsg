
<h1 align="center"> Private Seeds, Public LLMs: 
Realistic and Privacy-Preserving Synthetic Data Generation</h1>

Repository for the ACL 2026 Findings paper:
### 🔍 [arXiv](https://arxiv.org/abs/2604.07486) | [BibTeX](#bibtex)

RPSG is a realistic and privacy-preserving synthetic data generation method. Empirical results demonstrate that RPSG consistently outperforms SOTA baselines in generating high-quality synthetic data, achieving strong utility while safeguarding privacy. 
<p align="center">
  <img src="figures/figure1.png" width="90%">
</p>



## ⚙️ Setup
### Environment setup

```
conda env create -f rpsg.yml
conda activate rpsg
```

### Data Preparation

Datasets are located at  `data/<dataset>` where `dataset` is `reddit` and `pubmed`.

Download PubMed `train.csv` by executing:
```bash 
bash scripts/download_data.sh
```
Dataset description: 
- Reddit: We collected real-world self-disclosure content from Reddit to target posts authored by LSE populations. We first selected subreddits associated with financial hardship and poverty, and applied a keyword-based filtering strategy designed to capture language related to economic challenges. We collected posts published between January 1, 2024, and March 31, 2025. The final dataset consists of 8,948, 1,000, and 1,000 posts in the training, validation, and test sets, respectively. All posts were publicly available and collected in accordance with Reddit’s terms of service. The collected data is used solely for research purposes aimed at understanding and improving synthetic data generation and privacy-preserving methods. The dataset will be shared upon request with verified researchers.
- PubMed: Abstracts of medical papers in [PubMed](https://www.ncbi.nlm.nih.gov/) from 2023/08/01 to 2023/08/07 crawled by [(Yu et al. 2023)](https://openreview.net/forum?id=FKwtKzglFb), with 75316 abstracts for training, 14423 for validation, and 4453 for testng.


### Adding Sentiment GT to Private Data

```bash 
# Append label1 as the sentiment GT at the last column
python add_sentiment_label.py 
	--input_csv <private.csv> \ 
	--output_csv  <private_senti.csv>      
```

### Generating Private Data Embeddings

```bash 
# Pre-compute embeddings for private data
bash scripts/embeddings.sh --reddit 
bash scripts/embeddings.sh --pubmed       
```



## 🚀 Run Scripts
### Each end-to-end script consists of data generation, filtering, performance evaluation, and PII detection.

```bash 
bash scripts/{LLMs}/*.sh
# Replace `{LLMs}` with the names of LLMs. For example:
# bash scripts/gpt-4o-mini/sd_g4o.sh leverages the gpt-4o-mini and the reddit dataset
# bash scripts/hf/pubmed/rpsg-pubmed_phi4.sh leverages the phi-4 and the pubmed dataset
```

### Azure API
We use Azure API to access closed-source LLMs, please get your api key via [Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line,python&pivots=programming-language-python)
```python
DeepSeek_CONFIG={
        'DeepSeek-R1':{ "openai_api_key":  "YOUR_AZURE_OPENAI_API_KEY",
                        "openai_api_base": "YOUR_AZURE_OPENAI_ENDPOINT",
                        "engine": 'DeepSeek-R1',
                      },
    }
```

## 📂 Results Structure
```
${synthetic size}_n{private budget}_L2_initL2_var0_{dataset}_{llm}_rephrase_tone_rank_len{syn_length}var60wo2to5_t{temperature}
├── 0
│   ├── samples.csv # randomly selected private samples
├── 1
│   ├── samples.csv # synthetic variants, for training the surrogate model
├── 1_all
│   ├── samples.csv # synthetic samples before refinement
│   ├── filtered_samples.csv # final synthetic samples
│────
```

## 📊 Sentiment Evaluation

All evaluation results were automatically saved in the `1_all` folder when the script finished running.

```bash 
# optional
python eval_sentiment.py 
	--private_file <private.csv> \ 
	--synthetic_file  <synthetic.csv> \
	--output_dir <out_dir>   
```

## 📬
### BibTeX

```
@misc{ma2026privateseedspublicllms,
      title={Private Seeds, Public LLMs: Realistic and Privacy-Preserving Synthetic Data Generation}, 
      author={Qian Ma and Sarah Rajtmajer},
      year={2026},
      eprint={2604.07486},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2604.07486}, 
}
```

## 👥 Acknowledgement

- [AUG-PE](https://github.com/AI-secure/aug-pe)

