from utils import load_and_preprocess_dataset, build_example_object, calculate_dataset_stat, prepare_dataset, get_batch_size, set_seed, set_dist
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import time
import logging
import os
from tqdm import tqdm
from config import config_inference as config
from models import load_model, get_model_size
import torch
import transformers


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split(list_input, n):
    k, m = divmod(len(list_input), n)
    return (list_input[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def generate_prediction(config, dataset, examples, model, tokenizer, split_tag, cur_epoch, output_path, device=None): 
    logger.info("  ***** Running inference on {} data *****".format(split_tag))
    logger.info("  Num examples = %d",len(dataset) )
    logger.info("  Batch size = %d", get_batch_size(config=config, split=split_tag))
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=get_batch_size(config, split_tag))

    model.eval()
    pred_ids = []
 
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Inference for {} set".format(split_tag)):
        batch = {k: v.to(device) for k, v in batch.items()}
        if config["model_type"] in ("codebert", "roberta", "bert"):
            batch["source_attention_mask"] = batch["source_attention_mask"].type(torch.bool)
            batch["target_attention_mask"] = batch["target_attention_mask"].type(torch.bool)
        
        with torch.no_grad():
            if config["model_type"] in ("codebert", "roberta"):
                preds = model(source_ids=batch["source_ids"], source_mask=batch["source_attention_mask"])
                top_preds = [pred[0:config["top_k_inference"]].cpu().numpy() for pred in preds]
            
            else:
                # type(preds) == tensor
                preds = model.generate(batch["source_ids"],
                                       attention_mask=batch["source_attention_mask"],
                                       use_cache=True,
                                       num_beams=config["beam_size"],
                                       early_stopping=False,
                                       max_length=config["max_length"],
                                       num_return_sequences=config["top_k_inference"])
                
                top_preds = list(preds.cpu().numpy())
            # type(pred_ids) == list (this is list of numpy array)
            pred_ids.extend(top_preds)

    results_path = os.path.join(output_path, "results")
    if not(os.path.exists(results_path)):
        os.makedirs(results_path)
    
    if config["model_type"] in ("codebert", "roberta"):
        for idx, (item, ground_truth) in enumerate(zip(pred_ids, examples)):
            
            gold_path = f"{results_path}/prediction-{split_tag}-id{idx}.gold"
            output_path = f"{results_path}/prediction-{split_tag}-id{idx}.output"
            src_path = f"{results_path}/prediction-{split_tag}-id{idx}.src"
            
            with open(output_path, "w")  as f0, open(gold_path, "w") as f1, open(src_path, "w") as f2:            
                for subitem in item:
                    prediction = tokenizer.decode(subitem, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    f0.write(prediction.strip() + '\n')
                f1.write(ground_truth.target.strip() + "\n")
                f2.write(ground_truth.source.strip() + "\n")

    else:
        pred_ids = list(split(pred_ids, len(dataset)))
        for idx, (item, ground_truth) in enumerate(zip(pred_ids, examples)):
            # logger.info(f"target: {ground_truth.target}")
            
            gold_path = f"{results_path}/prediction-{split_tag}-id{idx}.gold"
            output_path = f"{results_path}/prediction-{split_tag}-id{idx}.output"
            src_path = f"{results_path}/prediction-{split_tag}-id{idx}.src"
            
            with open(output_path, "w")  as f0, open(gold_path, "w") as f1, open(src_path, "w") as f2:
                for subitem in item:
                    prediction = tokenizer.decode(subitem, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    f0.write(prediction.strip() + '\n')
                f1.write(ground_truth.target.strip() + "\n")
                f2.write(ground_truth.source.strip() + "\n")
 
def inference(config):
    t0 = time.time()
    set_seed(config)
    device, config["n_cpu"] = set_dist(config)

    checkpoint_path = config["checkpoint_path"]
    output_path = config["output_path"]
    
    logger.info(f"  Checkpoint path: {checkpoint_path}")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Device used: {device}, Cpu_count: {config['n_cpu']}")
    logger.info("  ***** Initialization success!  ***** ")

    # load model
    model_config, model, tokenizer = load_model(config)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info("  Reload model from {}".format(checkpoint_path))
    model.to(device)
    logger.info(f"  Finish loading model {get_model_size(model)} from {config['model_type']}")
    logger.info("  ***** Model is loaded successfully!  ***** \n")

        
    # fa = open(os.path.join(output_path, 'summary.log'), 'a+')


    logger.info(" ***** Testing ***** ")

    dataset_list = []
    if (config["test_gold_datapath"]):
        test_gold = load_and_preprocess_dataset(config=config, split_tag="test_gold", tokenizer=tokenizer, is_calculate_stats=False)
        dataset_list.append((test_gold, "test_gold"))
        logger.info("  ***** test_gold data is loaded successfully!  ***** ")
    if (config["test_intel_datapath"]):
        test_intel = load_and_preprocess_dataset(config=config, split_tag="test_intel", tokenizer=tokenizer, is_calculate_stats=False)
        dataset_list.append((test_intel, "test_intel"))
        logger.info("  ***** test_intel is loaded successfully!  ***** ")

    for dataset, split_tag in dataset_list:
        data, examples = dataset

        result = generate_prediction(config=config,
                        dataset=data, 
                        examples=examples, 
                        model=model, 
                        tokenizer=tokenizer, 
                        split_tag=split_tag, 
                        cur_epoch=split_tag, 
                        output_path=output_path, 
                        device=device)

        # test_bleu, test_em = result['bleu'], result['em']
        # result_str = "[%s] bleu-4: %.2f, em: %.4f\n" % (split_tag, test_bleu, test_em)
        # logger.info(result_str)
        # fa.write(result_str)
        
        # fpath = os.path.join(output_path, "test_result.out")
        # with open(fpath, 'a+') as f:
            # f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), checkpoint_path))
            # f.write(result_str)

    # logger.info("Finish and take {}".format(get_elapse_time(t0)))
    # fa.write("Finish and take {}".format(get_elapse_time(t0)))
    # fa.close()          
    del model
if __name__ == "__main__":
    inference(config=config)