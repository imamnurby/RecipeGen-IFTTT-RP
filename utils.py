import os
import torch
import pandas as pd
import numpy as np
import random
import multiprocessing
import logging
import time
import json
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset_from_dict(path, data_num):
    df = pd.read_pickle(os.path.join(os.getcwd(), path))

    if data_num > 0:
        df = df.sample(n=data_num, random_state=42)

    features_dict = df.to_dict(orient="list")
    return Dataset.from_dict(features_dict)

def load_train_data(config):
    # assert(config["is_train"]== True)
    return load_dataset_from_dict(config["training_datapath"], config["data_num"])

def load_validation_data(config):
    # assert(config["is_validation"] == True)
    return load_dataset_from_dict(config["validation_datapath"], config["data_num"]) 

def load_test_data_gold(config):
    # assert(config["is_test_intel"] == True)
    return load_dataset_from_dict(config["test_gold_datapath"], config["data_num"])
   
def load_test_data_intel(config):
    # assert(config["is_test_intel"] == True)
    return load_dataset_from_dict(config["test_intel_datapath"], config["data_num"])
    
def prepare_dataset(dataset, column_to_rename_list, mapping_function):
    dataset = dataset.map(mapping_function, batched=True)
    for pair in column_to_rename_list:
        old, new = pair
        dataset = dataset.rename_column(old, new)
    return dataset

def load_and_preprocess_dataset(config, split_tag, tokenizer=None, is_calculate_stats=False):

    def tokenize_dataset_source(example):
        return tokenizer(example["source"], max_length=config["max_length"], padding='max_length', truncation=True)

    def tokenize_dataset_target(example):
        return tokenizer(example["target"], max_length=config["max_length"], padding='max_length', truncation=True)
    
    if split_tag == "train":
        dataset = load_train_data(config)
    elif split_tag == "validation":
        dataset = load_validation_data(config)
    elif split_tag == "test_gold":
        dataset = load_test_data_gold(config)
    elif split_tag == "test_intel":
        dataset = load_test_data_intel(config)
    
    examples = build_example_object(dataset["source"], dataset["target"])
    
    if is_calculate_stats and tokenizer != None:
        calculate_dataset_stat(dataset, tokenizer)
    dataset = prepare_dataset(dataset, [("input_ids", "source_ids"), ("attention_mask", "source_attention_mask")], tokenize_dataset_source)
    dataset = prepare_dataset(dataset, [("input_ids", "target_ids"), ("attention_mask", "target_attention_mask")], tokenize_dataset_target)
    dataset = dataset.remove_columns(["source", "target"])
    dataset.set_format("torch")
    return dataset, examples

def set_seed(config):
    """set random seed."""
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["n_gpu"] > 0:
        torch.cuda.manual_seed_all(config["seed"])

def set_dist(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_count = multiprocessing.cpu_count()
    return device, cpu_count

def calculate_dataset_stat(dataset, tokenizer):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for example in dataset:
        avg_src_len.append(len(example["source"].split()))
        avg_trg_len.append(len(example["target"].split()))
        avg_src_len_tokenize.append(len(tokenizer.tokenize(example["source"])))
        avg_trg_len_tokenize.append(len(tokenizer.tokenize(example["target"])))
    logger.info(f"  Read {len(dataset)} examples, avg src len: {np.mean(avg_src_len)}, avg trg len: {np.mean(avg_trg_len)}, max src len: {max(avg_src_len)}, max trg len: {max(avg_trg_len)}")
    logger.info(f"  [TOKENIZE] avg src len: {np.mean(avg_src_len_tokenize)}, avg trg len: {np.mean(avg_trg_len_tokenize)}, max src len: {max(avg_src_len_tokenize)}, max trg len: {max(avg_trg_len_tokenize)}")
    
def get_batch_size(config, split):
    if split=="train":
        return config["train_batch_size"]
    elif split=="validation":
        return config["val_batch_size"]
    elif split in ["test_gold", "test_intel"]:
        return config["inference_batch_size"]

def get_output_path(config):
    cwd = os.getcwd()
    model_name = config["model_name"]
    model_type = config["model_type"]
    bs = config["train_batch_size"]
    seed = config["seed"]
    if config["is_clean"] == "clean":
        is_clean = "clean"
    elif config["is_clean"] == "noisy":
        is_clean = "noisy"
    elif config["is_clean"] == "mi":
        is_clean = "mi"
    elif config["is_clean"] == "chen":
        is_clean = "chen" 
    elif config['is_clean'] == "random":
        is_clean = "random"
    elif config["is_clean"] == "mi-field":
        is_clean = "mi-field"
    else:
        is_clean = ""

    output_path = f"{os.path.join(cwd, f'output/{model_type}-bs{bs}-{is_clean}-{seed}/{model_name}')}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path

def get_checkpoint_path(config):
    cwd = os.getcwd()
    model_name = config["model_name"]
    output_path = get_output_path(config)
    checkpoint_path = f"{os.path.join(cwd, f'{output_path}/checkpoint')}"

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    checkpoint_path = os.path.join(checkpoint_path, f'pytorch_model_{config["model_name"]}.bin')
    return checkpoint_path

def get_model_name(config):
    currentDT = time.localtime(time.time())
    timeframe = time.strftime("%H-%M-%S--", currentDT) + time.strftime("%d-%m-%Y", currentDT)
    return f"{config['model_type']}-{timeframe}"

def save_config(config, output_path):
    fpath = os.path.join(output_path, "config.json")
    with open(fpath, 'w') as fp:
        json.dump(config, fp)

###
# https://github.com/salesforce/CodeT5/blob/ad787aa6bb08ede41c94c397e577adfc7ebac39d/models.py
class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def build_example_object(list_of_source, list_of_target):
    examples = []
    for idx, (source, target) in enumerate(zip(list_of_source, list_of_target)):
        examples.append(
            Example(
                idx=idx,
                source=source,
                target=target
            )
        )
    return examples

def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
###
