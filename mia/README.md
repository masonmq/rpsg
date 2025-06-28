

## Setup
### Environment setup

```
conda env create -f rpsg.yml
conda activate rpsg
```

### Run AUC Score

After launching the MI attacks, the corresponding .pkl files will be automatically generated. For instance, the directory `mia/rpsg_reddit_phi4` contains the result files for the case where synthetic data was generated using Phi-4 on the Reddit dataset.

The associated AUC scores can be obtained by executing:

```
python eval.py --all_scores_sd mia/rpsg_reddit_phi4/sd-rpsg-bertsm_all_scores_sd_bert-small_0.pkl --all_scores mia/rpsg_reddit_phi4/sd-rpsg-bertsm_all_scores0.pkl
```

This will display the MIA results which we reported in Table 4 (Evaluation of MIAs) in our paper. 

```
AUC (PPL): 47.52
AUC (REFER): 52.36
AUC (LIRA): 49.19
```