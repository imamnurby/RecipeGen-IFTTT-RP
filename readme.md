- [About](#about)
- [Dataset](#dataset)
- [Pre-requisites](#pre-requisites)
- [Running experiment](#running-experiment)
- [Results](#results)
  * [BLEU score](#bleu-score)
    + [Function-level](#function-level)
  * [Mean Reciprocal Rank](#mean-reciprocal-rank)
    + [Function-level](#function-level-1)
    + [Field-level](#field-level)
  * [Sucess Rate](#sucess-rate)
    + [Function-level](#function-level-2)
    + [Field-level](#field-level-1)

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


# Results
## BLEU score
### Function-level

|   Model  	| Training Dataset 	| Gold 	| Noisy 	|
|:--------:	|:----------------:	|:----:	|:-----:	|
| Random  	|      Merged      	| 90.2 	|  60.6 	|
| LAM   	|                	| 86.4 	|  52.9 	|
| RoBerta 	|                	| 93.6 	|  64.0 	|
| CodeBERT 	|                	| 93.6 	|  63.7 	|
| Random  	|        Mi        	| 82.3 	|  58.7 	|
| LAM   	|                	| 76.2 	|  45.3 	|
| RoBerta 	|                	| 86.0 	|  62.9 	|
| CodeBERT 	|                	| 91.8 	|  75.5 	|
| Random  	|       Chen       	| 81.9 	|  56.4 	|
| LAM   	|                 	| 78.7 	|  49.9 	|
| RoBerta 	|                 	| 92.2 	|  62.4 	|
| CodeBERT 	|                	| 89.7 	|  61.7 	|

### Field-level

|   Model  	| Training Dataset 	| BLEU-score 	|
|:--------:	|:----------------:	|:----------:	|
| Random   	|     Mi-field     	| 55.5       	|
| RoBerta  	|                  	| 58.8       	|
| CodeBERT 	|                  	| 59.1       	|



## Mean Reciprocal Rank 
### Function-level

|  Model   	| Training Dataset 	| Gold MRR@3 	| Gold MRR@5 	| Gold MRR@10 	| Noisy MRR@3 	| Noisy MRR@5 	| Noisy MRR@10 	|
|:--------:	|:----------------:	|:----------:	|:----------:	|:-----------:	|:-----------:	|:-----------:	|:------------:	|
|  Random  	|      Merged      	|    90.9    	|    91.1    	|     91.2    	|     61.1    	|     62.4    	|     62.9     	|
|    LAM   	|                  	|    81.4    	|    81.4    	|     81.4    	|     40.3    	|     40.3    	|     40.3     	|
|  RoBerta 	|                  	|    94.7    	|    94.8    	|     94.8    	|     64.7    	|     66.0    	|     66.5     	|
| CodeBERT 	|                  	|    94.2    	|    94.4    	|     94.4    	|     64.5    	|     65.6    	|     66.1     	|
|  Random  	|        Mi        	|    79.6    	|    80.1    	|     80.3    	|     55.2    	|     56.1    	|     56.5     	|
|    LAM   	|                  	|    68.4    	|    68.4    	|     68.4    	|     32.5    	|     32.5    	|     32.5     	|
|  RoBerta 	|                  	|    83.0    	|    83.1    	|     83.1    	|     59.0    	|     59.8    	|     60.0     	|
| CodeBERT 	|                  	|    84.1    	|    84.2    	|     84.3    	|     58.9    	|     59.3    	|     59.5     	|
|  Random  	|       Chen       	|    81.2    	|    81.4    	|     81.7    	|     54.2    	|     55.7    	|     56.4     	|
|    LAM   	|                  	|    71.7    	|    71.7    	|     71.7    	|     38.1    	|     38.1    	|     38.1     	|
|  RoBerta 	|                  	|    92.3    	|    92.5    	|     92.5    	|     62.3    	|     63.7    	|     64.3     	|
| CodeBERT 	|                  	|    90.3    	|    90.6    	|     90.6    	|     61.7    	|     63.0    	|     63.4     	|

### Field-level

|  Model   	| Training Dataset 	| MRR@3 	| MRR@5 	| MRR@10 	|
|:--------:	|:----------------:	|-------	|-------	|--------	|
| Random   	|     Mi-field     	| 53.6  	| 0.546 	| 0.551  	|
| RoBerta  	|                  	| 57.1  	| 58.2  	| 58.6   	|
| CodeBERT 	|                  	| 57.5  	| 58.5  	| 59.0   	|

## Sucess Rate
### Function-level

|  Model   	| Training Dataset 	| Gold SR@1 	| Gold SR@3 	| Gold SR@5 	| Gold SR@10 	| Noisy SR@1 	| Noisy SR@3 	| Noisy SR@5 	| Noisy SR@10 	|
|:--------:	|:----------------:	|:---------:	|:---------:	|:---------:	|:----------:	|:----------:	|:----------:	|:----------:	|:-----------:	|
| Random   	|      Merged      	| 86.2      	| 96.1      	| 96.7      	| 97.4       	| 50.3       	| 73.9       	| 79.2       	| 82.8        	|
| LAM      	|                  	| 81.2      	| 81.6      	| 81.6      	| 81.6       	| 40.1       	| 40.5       	| 40.5       	| 40.5        	|
| RoBerta  	|                  	| 91.1      	| 98.4      	| 98.7      	| 99.0       	| 54.1       	| 77.5       	| 83.4       	| 86.7        	|
| CodeBERT 	|                  	| 91.1      	| 97.4      	| 98.4      	| 98.4       	| 53.1       	| 78.0       	| 82.7       	| 86.5        	|
| Random   	|        Mi        	| 76.4      	| 83.3      	| 85.6      	| 87.2       	| 47.7       	| 63.3       	| 67.6       	| 70.1        	|
| LAM      	|                  	| 68.4      	| 68.4      	| 68.4      	| 68.4       	| 32.4       	| 32.6       	| 32.6       	| 32.6        	|
| RoBerta  	|                  	| 80.7      	| 85.6      	| 86.2      	| 86.2       	| 52.5       	| 66.7       	| 70.4       	| 71.7        	|
| CodeBERT 	|                  	| 82.3      	| 86.2      	| 86.6      	| 87.5       	| 54.2       	| 64.6       	| 66.4       	| 68.1        	|
| Random   	|       Chen       	| 77.0      	| 86.2      	| 87.2      	| 88.9       	| 46.2       	| 64.1       	| 70.7       	| 75.6        	|
| LAM      	|                  	| 71.7      	| 71.7      	| 71.7      	| 71.7       	| 38.0       	| 38.2       	| 38.2       	| 38.2        	|
| RoBerta  	|                  	| 89.5      	| 95.7      	| 96.4      	| 96.7       	| 52.3       	| 74.0       	| 80.0       	| 84.3        	|
| CodeBERT 	|                  	| 86.2      	| 95.1      	| 96.4      	| 96.7       	| 51.9       	| 73.3       	| 79.1       	| 82.1        	|

### Field-level

|  Model   	| Training Dataset 	|  SR@1 	|  SR@3 	|  SR@5 	| SR@10 	|
|:--------:	|:----------------:	|:-----:	|:-----:	|:-----:	|:-----:	|
| Random   	|     Mi-field     	| 0.466 	| 0.621 	| 0.664 	| 0.698 	|
| RoBerta  	|                  	| 50.3  	| 65.4  	| 70.0  	| 73.2  	|
| CodeBERT 	|                  	| 50.6  	| 65.8  	| 70.3  	| 73.3  	|