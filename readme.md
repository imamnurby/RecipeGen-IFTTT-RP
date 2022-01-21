# About
This is the official repository of "Accurate Generation of Trigger-Action Programs with Domain-Adapted Sequence-to-Sequence Learning".

# Dataset
We make the datasets that we used in our study publicly available in [zenodo](https://zenodo.org/record/5885850#.YeolYP5ByUl).

Steps:
1. Download the dataset
2. Unzip the file
3. Put the folder inside the unzipped folder to `/dataset`

The paths to the datasets used in our paper are listed below. All of these datasets are in pickle format. 
- `Quirk15` in `ready-test-chen-only/train-chen.pkl`
- `Mi17` in `ready-test-mi-only/train-mi.pkl`
- `Merged` in `ready-test-train-val-noisy/train-noisy.pkl`
- `Val17` in `ready-test-train-val-noisy/validation-noisy.pkl`
- `Gold15` in `ready-test-clean/test_gold_clean.pkl`
- `Noisy15` in `ready-test-clean/test_intel_clean.pkl`
- `Train+Field`, `Val+Field`, and `Test+Field` in `ready-train-val-test-mi-field`

We provide some samples of our preprocessed dataset in `dataset/csv`

# Pre-requisites
Install requirements: `pip -r requirements.txt`

# Running experiment
Important files:
- `config.py` contains the configuration for training, inference, and calculating the metrics
- `run_pipeline.py` contains the training and evaluation pipeline
- `inference.py` contains the inference pipeline
- `models.py` contains the implementation of all models used in our experiments
- `compute_metrics.py` contains the script to calculate MRR and BLEU score


Running experiment:
1. Select the model
    - Steps:
        - Open `config.py`
        - To load Rob2Rand and Code2Rand:
            - set `load_from_repo` to `True`
            - set `model_type`  to either `roberta` to instantiate Rob2Rand or `codebert` to instantiate Code2Rand
        - To load Rand2Rand:
            - set `load_from_repo` to `False`
            - set `model_type`  to either `roberta`
        - You can also change other settings (such as hyperparams and training setting)
2. Specify the dataset path in `config.py`
2. Run the script `python run_pipeline.py`

Running Inference:
1. Specify the environment path
    - Steps:
        - Open `config.py`
        - Specify `output_path` with the path to the model folder
        - Specify `checkpoint_path` with the path to the checkpoint
        - Specify `test_gold_datapath` and `test_intel_datapath` with the dataset path 
2. Run the script `python inference.py`
3. The inference results are saved in `<folder_path>/results`

Compute metrics:
1. Specify the path to the inference results
    - Steps:
        - Open `config.py`
        - Specify `result_path` with the path to the inference results
        - By default, the script will compute MRR@k, BLEU, and SR@k
2. Run the script `python compute_metrics.py`