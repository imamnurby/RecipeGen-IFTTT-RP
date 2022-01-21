import os
import math
import collections
import re

###
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
###


def preprocess_tap_list(list_of_tap, is_ref=False):
    ref_cleaned = []
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

def compute_bleu(brevity_penalty, exponent_mean):
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

def convert_to_list(list_input, result_path):
    output = []
    for path in list_input:
        fpath = os.path.join(result_path, path)
        with open(fpath, "r") as f:
            output.append(f.readline().strip())
    return output