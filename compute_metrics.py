import logging
import os
import math
import collections
import re
import numpy as np
import pandas as pd
from config import config_inference as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



### ***** BLEU SCORE *****
# https://github.com/salesforce/CodeT5
def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts
#

def preprocess_tap_list_bleu(list_of_tap, config, is_ref=False):
    ref_cleaned = []
    if config["is_LAM"]:
        list_of_tap = [x.split() for x in list_of_tap]
    else:
        list_of_tap = [x.split("<sep>") for x in list_of_tap]
    for item in list_of_tap:
        temp_list = []
        for subitem in item:
            temp_list.append(subitem.strip())
        if is_ref:
            ref_cleaned.append([temp_list])
        else:
            ref_cleaned.append(temp_list)
    return ref_cleaned

def get_ngrams_matches(reference_corpus, translation_corpus, config):
    matches_by_order = [0] * config["max_order_ngrams"]
    possible_matches_by_order = [0] * config["max_order_ngrams"]
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, config["max_order_ngrams"])
        translation_ngram_counts = _get_ngrams(translation, config["max_order_ngrams"])
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, config["max_order_ngrams"]+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches
    return matches_by_order, possible_matches_by_order, reference_length, translation_length

def compute_precision(ngrams_match_counts_overlap, 
                      all_possible_ngrams_match_counts, 
                      config):
    precisions = [0] * config["max_order_ngrams"]
    for i in range(0, config["max_order_ngrams"]):
        if config["smooth"]:
            precisions[i] = ((ngrams_match_counts_overlap[i] + 1.) / (all_possible_ngrams_match_counts[i] + 1.))
    else:
        if all_possible_ngrams_match_counts[i] > 0:
            precisions[i] = (float(ngrams_match_counts_overlap[i]) /
                         all_possible_ngrams_match_counts[i])
        else:
            precisions[i] = 0.0
    return precisions

def compute_exponent_mean(precisions, config):
    if min(precisions) > 0:
        p_log_sum = sum((1. / config["max_order_ngrams"]) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0
    return geo_mean

def compute_brevity_penalty(ref_len, tran_len):
    ratio = float(ref_len) / tran_len

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)
    return bp

def bleu_score(brevity_penalty, exponent_mean):
    return brevity_penalty*exponent_mean

###
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# alist=[
#     "something1",
#     "something12",
#     "something17",
#     "something2",
#     "something25",
#     "something29"]
# alist.sort(key=natural_keys)
###

def get_list_of_file(result_path, key, extension):
    list_of_file = os.listdir(result_path)
    list_of_file = [x for x in list_of_file if key in x and extension in x]
    list_of_file.sort(key=natural_keys)
    return list_of_file

def convert_to_list_bleu(list_input, result_path):
    output = []
    for path in list_input:
        fpath = os.path.join(result_path, path)
        with open(fpath, "r") as f:
            output.append(f.readline().strip())
    return output

def compute_bleu_score(config, key):
    
    truth = get_list_of_file(result_path=config["result_path"], key=key, extension=".gold")
    pred = get_list_of_file(result_path=config["result_path"], key=key, extension=".output")

    
    truth = convert_to_list_bleu(list_input=truth, result_path=config["result_path"])
    pred = convert_to_list_bleu(list_input=pred, result_path=config["result_path"])
    
    truth = preprocess_tap_list_bleu(list_of_tap=truth, is_ref=True, config=config)
    pred = preprocess_tap_list_bleu(list_of_tap=pred, config=config)
    
    ngrams_match_counts_overlap, all_possible_ngrams_match_counts, ref_len, tran_len = get_ngrams_matches(reference_corpus=truth,
                                                                                                       translation_corpus=pred,
                                                                                                       config=config)
    
    precisions = compute_precision(ngrams_match_counts_overlap=ngrams_match_counts_overlap, 
                                    all_possible_ngrams_match_counts=all_possible_ngrams_match_counts, 
                                    config=config)

    exponent_mean = compute_exponent_mean(precisions=precisions,
                                        config=config)
    
    brevity_penalty = compute_brevity_penalty(ref_len=ref_len,
                                            tran_len=tran_len)
    
    bleu = bleu_score(brevity_penalty=brevity_penalty,
                        exponent_mean=exponent_mean)
    logger.info(f"  *****  BLEU score for {key} dataset: {round(bleu, 3)}  *****  ")
    return bleu

### ***** MRR *****
def preprocess_tap_list_mrr(list_of_tap, is_ref=False):
    ref_cleaned = []
    for item in list_of_tap:
        temp_list = []
        for subitem in item:
            subitem = subitem.replace("<sep>", "")
            temp_list.append(subitem.strip())
        ref_cleaned.append(temp_list)
    return ref_cleaned

def convert_to_list_mrr(config, list_input, result_path):
    output = []
    for path in list_input:
        fpath = os.path.join(result_path, path)
        with open(fpath, "r") as f:
            temp_list = f.readlines()[:config["top_k_mrr"]]
            temp_list = [x.strip() for x in temp_list]
            output.append(temp_list)
    return output

###
# https://gist.github.com/bwhite/3726239
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
###

def compute_mrr(config, key):
    truth = get_list_of_file(result_path=config["result_path"], key=key, extension=".gold")
    pred = get_list_of_file(result_path=config["result_path"], key=key, extension=".output")

    truth = convert_to_list_mrr(config=config, list_input=truth, result_path=config["result_path"])
    pred = convert_to_list_mrr(config=config, list_input=pred, result_path=config["result_path"])
    
    truth = preprocess_tap_list_mrr(list_of_tap=truth, is_ref=True)
    pred = preprocess_tap_list_mrr(list_of_tap=pred)

    relevancy_list = np.array(truth) == np.array(pred)
    relevancy_list = relevancy_list.astype("int").tolist()
    
    mrr = mean_reciprocal_rank(relevancy_list)
    logger.info(f"  *****  MRR@{config['top_k_mrr']} score for {key} dataset: {round(mrr, 3)}  *****  ")
    return mrr

### ***** Success Rate *****

def compute_success_rate(config, key):
    truth = get_list_of_file(result_path=config["result_path"], key=key, extension=".gold")
    pred = get_list_of_file(result_path=config["result_path"], key=key, extension=".output")

    truth = convert_to_list_mrr(config=config, list_input=truth, result_path=config["result_path"])
    pred = convert_to_list_mrr(config=config, list_input=pred, result_path=config["result_path"])
    
    truth = preprocess_tap_list_mrr(list_of_tap=truth, is_ref=True)
    pred = preprocess_tap_list_mrr(list_of_tap=pred)

    success_rate = np.array(truth) == np.array(pred)
    success_rate = np.max(success_rate, axis=1)
    success_rate = np.sum(success_rate)/len(success_rate)
    logger.info(f"  *****  SuccessRate@{config['top_k_mrr']} score for {key} dataset: {round(success_rate, 3)}  *****  ")
    return success_rate

### ***** Individual Accuracy *****
def calculate_acc(arr_input):
    return round(np.sum(arr_input)/len(arr_input), 3)*100

def compute_individual_acc(config, key):
    out = get_list_of_file(result_path=config["result_path"], key=key, extension=".output")
    gold = get_list_of_file(result_path=config["result_path"], key=key, extension=".gold")
    # src = get_list_of_file(result_path=config["result_path"], key=key, extension=".src")

    if out and gold:
        out = convert_to_list_mrr(config=config, list_input=out, result_path=config["result_path"])
        out = preprocess_tap_list_mrr(list_of_tap=out)

        gold = convert_to_list_mrr(config=config, list_input=gold, result_path=config["result_path"])
        gold = preprocess_tap_list_mrr(list_of_tap=gold)

        # src = convert_to_list_mrr(config=config, list_input=src, result_path=config["result_path"])
        # src = preprocess_tap_list_mrr(list_of_tap=src)

        df = pd.DataFrame(out, columns=["top1"])

        gold = [x for item in gold for x in item]
        # src = [x for item in src for x in item]
        df["gold"] = gold
        # df["src"] = src

        for idx, col_name in enumerate(("tc_pred", "tf_pred", "ac_pred", "af_pred")):
            df[col_name] = df.top1.apply(lambda x: x.split()[idx])

        for idx, col_name in enumerate(("tc_gold", "tf_gold", "ac_gold", "af_gold")):
            df[col_name] = df.gold.apply(lambda x: x.split()[idx])

        tf_pred = (df["tf_pred"] == df["tf_gold"]).to_numpy()
        tc_pred = (df["tc_pred"] == df["tc_gold"]).to_numpy()
        ac_pred = (df["ac_pred"] == df["ac_gold"]).to_numpy()
        af_pred = (df["af_pred"] == df["af_gold"]).to_numpy()

        tf_acc = calculate_acc(arr_input=tf_pred)
        tc_acc = calculate_acc(arr_input=tc_pred)
        af_acc = calculate_acc(arr_input=af_pred)
        ac_acc = calculate_acc(arr_input=ac_pred)

        logger.info(f"  *****  tc_acc score for {key} dataset: {round(tc_acc, 3)}  *****  ")
        logger.info(f"  *****  tf_acc score for {key} dataset: {round(tf_acc, 3)}  *****  ")
        logger.info(f"  *****  ac_acc score for {key} dataset: {round(ac_acc, 3)}  *****  ")
        logger.info(f"  *****  af_acc score for {key} dataset: {round(af_acc, 3)}  *****  ")

        return tc_acc, tf_acc, ac_acc, af_acc
    else:
        return 0, 0, 0, 0

def compute_all_metrics(config):
    logger.info(f"  Result path: {config['result_path']}")
    
    if config["do_gold"]:
        bleu_gold = compute_bleu_score(config=config, key="test_gold")
        fgold = open(os.path.join(config["result_path"], 'metrics_gold.log'), 'w')
        fgold.write(f"Result path: {config['result_path']}\n")
        fgold.write(f"\n")
        fgold.write(f"BLEU-4 score: {round(bleu_gold, 3)}\n")
        fgold.write(f"\n")
        for k in (3, 5, 10):
            config["top_k_mrr"] = k
            mrr_gold = compute_mrr(config=config, key="test_gold")
            fgold.write(f"MRR@{k}: {round(mrr_gold, 3)}\n")
        fgold.write(f"\n")
        
        for k in (1, 3, 5, 10):
            config["top_k_mrr"] = k
            success_rate_gold = compute_success_rate(config=config, key="test_gold")
            fgold.write(f"SuccessRate@{k}: {round(success_rate_gold, 3)}\n")
        fgold.close()

    if config["do_intel"]:
        bleu_intel = compute_bleu_score(config=config, key="test_intel")
        fintel = open(os.path.join(config["result_path"], 'metrics_intel.log'), 'w')
        fintel.write(f"Result path: {config['result_path']}\n")
        fintel.write(f"\n")
        fintel.write(f"BLEU-4 score: {round(bleu_intel, 3)}\n")
        fintel.write(f"\n")

        for k in (3, 5, 10):
            config["top_k_mrr"] = k
            mrr_intel = compute_mrr(config=config, key="test_intel")
            fintel.write(f"MRR@{k}: {round(mrr_intel, 3)}\n")
        fintel.write(f"\n")
        
        for k in (1, 3, 5, 10):
            config["top_k_mrr"] = k
            success_rate_intel = compute_success_rate(config=config, key="test_intel")
            fintel.write(f"SuccessRate@{k}: {round(success_rate_intel, 3)}\n")
        fintel.close()

    if config["do_individual_acc"]:
        save_path = os.path.join(config["result_path"], "ablation_individual_acc.txt")
        tc_acc, tf_acc, ac_acc, af_acc = compute_individual_acc(config=config, key="test_gold")
        if tc_acc != 0 and tf_acc !=0 and ac_acc != 0 and af_acc != 0:
            with open(save_path, "+a") as f:
                f.write(f"TC_acc_gold: {tc_acc}\n")
                f.write(f"TF_acc_gold: {tf_acc}\n")
                f.write(f"AC_acc_gold: {ac_acc}\n")
                f.write(f"AF_acc_gold: {af_acc}\n")
                f.write("\n")
            
        tc_acc, tf_acc, ac_acc, af_acc = compute_individual_acc(config=config, key="test_intel")
        if tc_acc != 0 and tf_acc !=0 and ac_acc != 0 and af_acc != 0:
            with open(save_path, "+a") as f:
                f.write(f"TC_acc_intel: {tc_acc}\n")
                f.write(f"TF_acc_intel: {tf_acc}\n")
                f.write(f"AC_acc_intel: {ac_acc}\n")
                f.write(f"AF_acc_intel: {af_acc}\n")
                f.write("\n")

if __name__ == "__main__":
    compute_all_metrics(config)
