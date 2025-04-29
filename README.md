# rpsg
Realistic and Privacy-Preserving Synthetic Data Generation

<h1 align="center"> Private Seeds, Public LLMs: Realistic and Privacy-Preserving Synthetic Data Generation</h1>

In this work, we presented RPSG, a realistic and privacy-preserving synthetic data generation method. Empirical results demonstrate that RPSG consistently outperforms SOTA baselines in generating high-quality synthetic data, achieving strong utility while safeguarding privacy. We hope this work highlights the importance of integrating privacy considerations into synthetic data generation and inspires further research and real-world adoption of privacy-aware practices.

<p align="center">
  <img src="figures/overview1.png" width="100%">
</p>


 Under $\epsilon=1$, Aug-PE produces DP synthetic text that yields competitive utility with the SOTA DP-SGD finetuning baselines on OpenReview data.


<p align="center">
  <img src="figures/overview2.png" width="70%">
</p>



## Setup
### Environment setup

```
conda env create -f environment.yml
conda activate augpe
```

### Data Preparation

Datasets are located at  `data/{dataset}` where `dataset` is `yelp`,  `openreview` and `pubmed`.

Download the Yelp `train.csv` (1.21G)  and PubMed `train.csv` (117MB)  from [this link](https://drive.google.com/drive/folders/1oSICwgCAqdxEz4mF5ZK863RoN5sxMB_0?usp=sharing) or execute:
```bash 
bash scripts/download_data.sh # download yelp train.csv and pubmed train.csv
```
Dataset description: 
- Yelp: Processed Yelp dataset from [(Yue et al. 2023)](https://aclanthology.org/2023.acl-long.74/) with  1.9M reviews for training,
5000 for validation, and 5000 for testing.
- OpenReview: Crawled and processed ICLR 2023 reviews from [OpenReview website](https://openreview.net/group?id=ICLR.cc/2023/Conference), with  8396 reviews for training, 2798 for validation, and 2798 for testing.
- PubMed: Abstracts of medical papers in [PubMed](https://www.ncbi.nlm.nih.gov/) from 2023/08/01 to 2023/08/07 crawled by [(Yu et al. 2023)](https://openreview.net/forum?id=FKwtKzglFb), with 75316 abstracts for training, 14423 for validation, and 4453 for testng.



### Generating Private Data Embeddings

Pre-compute embeddings for private data (line 1 in Aug-PE algorithm):
```bash  
bash scripts/embeddings.sh --openreview  # Compute private embeddings  
bash scripts/embeddings.sh --pubmed      
bash scripts/embeddings.sh --yelp       
```
Note: Computing embeddings for OpenReview and PubMed is relatively quick. However, due to Yelp's large dataset size (1.9M training samples), the process may take approximately 40 minutes.


### Calculating the Noise Level Under $\epsilon$ privacy budget
Calculate the DP noise level for your dataset in `notebook/dp_budget.ipynb` given the privacy budget $\epsilon$. 
To achieve $\epsilon=1,2,4$ under 10 epochs,  we set noise level 
[15.34, 8.03, 4.24] for yelp,
[11.60, 6.22, 3.38] for openreview,
[13.26, 7.01, 3.75] for pubmed. 



### Wandb
 
For visualization with Wandb, configure the `--wandb_key` and `--project` with your key and project name in `dpsda/arg_utils.py`.


## 🚀 Run (open-source LLMs)

### 📂 Generate DP Synthetic Text with Aug-PE

Utilize open-source LLMs from Hugging Face to generate synthetic data:
```bash 
export CUDA_VISIBLE_DEVICES=0 
bash scripts/hf/{dataset}/generate.sh  # Replace `{dataset}` with yelp, openreview, or pubmed
```
Some key hyperparameters: 
- `noise`: DP noise.
- `epoch`: we use 10 epochs for DP setting. For the non-DP setting, we use 20 epochs for Yelp and 10 epochs for other datasets. 
- `model_type`: model on huggingface, such as ["gpt2", "gpt2-medium", "gpt2-large", "meta-llama/Llama-2-7b-chat-hf", "tiiuae/falcon-7b-instruct", "facebook/opt-6.7b", "lmsys/vicuna-7b-v1.5", "mistralai/Mixtral-8x7B-Instruct-v0.1"].
- `num_seed_samples`: number of synthetic samples. 
- `lookahead_degree`: number of variations for synthetic sample embedding estimation (line 5 in Aug-PE algorithm). Default is 0 (self-embedding).
- `L`: related to the number of variations to generate candidate synthetic samples (line 18 in Aug-PE algorithm)
- `feat_ext`: embedding model on [huggingface sentence-transformers](https://huggingface.co/sentence-transformers).
- `select_syn_mode`: select synthetic samples according to histogram votes or probability. Default is `rank` (line 19 in Aug-PE algorithm)
- `temperature`: temperature for LLM generation.



### 📊 Evaluate DP Synthetic Text 
#### Accuracy on Downstream Tasks
Finetune the downstream model with DP synthetic text and evaluate the model's accuracy on real test data:
```bash 
bash scripts/hf/{dataset}/downstream.sh # Finetune downstream model and evaluate performance
```
#### Similary between Synthetic and Real Data
Measure the embedding distribution distance:
```bash 
bash scripts/hf/{dataset}/metric.sh  # Calculate distribution distance
```

### Comprehensive End-to-End Scripts
For a streamlined process that combines all generation and evaluation steps:
```bash 
bash scripts/hf/template/{dataset}.sh # Complete workflow for each dataset
```

## 🚀 Run (closed-source LLMs)

### End-to-End Scripts

We use closed-source model via Azure OpenAI API. Please set your key and endpoint in `apis/azure_api.py` 
```python
MODEL_CONFIG={
        'gpt-3.5-turbo':{ "openai_api_key":  "YOUR_AZURE_OPENAI_API_KEY",
                            "openai_api_base": "YOUR_AZURE_OPENAI_ENDPOINT",
                            "engine": 'YOUR_DEPLOYMENT_NAME',
                            },
    }
```
Here `engine` could be `gpt-35-turbo` in [Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line,python&pivots=programming-language-python).


Run the following script to generate synthetic data,  evaluate it on the downstream task, and calculate the embedding distribution distance between real and synthetic data: 
```bash 
bash scripts/gpt-3.5-turbo/{dataset}.sh
```

We use text-length related prompts for GPT-3.5 to control the length of the generated text. We introduce several additional hyperparameters here:
- `dynamic_len` is used to enable the dynamic length mechanism.
- `word_var_scale`:  Gaussian noise variance used to determine targeted_word.  
- `max_token_word_scale`: max number of tokens per word. We set the max_token for LLM generation based on the targeted_word (specified in the prompt) and  max_token_word_scale. 


Use the notebook to calculate the text length distribution difference between real and synthetic data:  `notebook/text_lens_distribution.ipynb`


## Acknowledgement

- [AUG-PE](https://github.com/AI-secure/aug-pe)

