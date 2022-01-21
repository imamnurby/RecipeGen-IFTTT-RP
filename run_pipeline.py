from utils import (set_seed, set_dist, load_and_preprocess_dataset, build_example_object, 
                    get_elapse_time, get_output_path, get_checkpoint_path, get_batch_size,
                     get_model_name, save_config)
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from models import load_model, get_model_size
from metric import _bleu
from config import config_training as config
from tqdm import tqdm
import os
import sys
import time
import logging
import torch
import math
import numpy as np 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_bleu_score(config, dataset, examples, model, tokenizer, split_tag, cur_epoch, output_path, device=None): 
    logger.info("  ***** Running bleu evaluation on {} data *****".format(split_tag))
    logger.info("  Num examples = %d",len(dataset))
    logger.info("  Batch size = %d", get_batch_size(config=config, split=split_tag))
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=get_batch_size(config=config, split=split_tag))

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        batch = {k: v.to(device) for k, v in batch.items()}
        if config["model_type"] in ("codebert", "roberta", "bert"):
            batch["source_attention_mask"] = batch["source_attention_mask"].type(torch.bool)
            batch["target_attention_mask"] = batch["target_attention_mask"].type(torch.bool)
        
        with torch.no_grad():
            if config["model_type"] in ("codebert", "roberta"):
                preds = model(source_ids=batch["source_ids"], source_mask=batch["source_attention_mask"])
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                # type(preds) == tensor
                preds = model.generate(batch["source_ids"],
                                       attention_mask=batch["source_attention_mask"],
                                       use_cache=True,
                                       num_beams=config["beam_size"],
                                       early_stopping=config["early_stopping"],
                                       max_length=config["max_length"])
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
            
    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    if split_tag == "validation":
        out_path = f"{split_tag}_{cur_epoch}.output"
        gold_path =  f"{split_tag}_{cur_epoch}.gold"
        src_path = f"{split_tag}_{cur_epoch}.src"
    else:
        out_path = f"{split_tag}.output"
        gold_path =  f"{split_tag}.gold"
        src_path = f"{split_tag}.src"
    
    output_fn = os.path.join(output_path, out_path)
    gold_fn = os.path.join(output_path, gold_path)
    src_fn = os.path.join(output_path, src_path)

    dev_accs, predictions = [], []
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        for pred_nl, gold in zip(pred_nls, examples):
            dev_accs.append(pred_nl.strip() == gold.target.strip())
            f.write(pred_nl.strip() + '\n')
            f1.write(gold.target.strip() + '\n')
            f2.write(gold.source.strip() + '\n')

    bleu = round(_bleu(gold_fn, output_fn), 2)
    em = np.mean(dev_accs) * 100
    
    result = {'em': em, 'bleu': bleu}
    logger.info("  ***** Eval results  ***** ")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    logger.info("  ***** Evaluation finished!  ***** ")
    return result

def main():
    t0 = time.time()
    set_seed(config)
    device, config["n_cpu"] = set_dist(config)

    if config["model_name"] == None:
        config["model_name"] = get_model_name(config)
        output_path = get_output_path(config)
        checkpoint_path = get_checkpoint_path(config)
        save_config(config, output_path)
    else:
        output_path = get_output_path(config)
        checkpoint_path = get_checkpoint_path(config)
    
    logger.info(f"  Device used: {device}, Cpu_count: {config['n_cpu']}")
    logger.info("  ***** Initialization success!  ***** ")

    # load model
    model_config, model, tokenizer = load_model(config)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info("  Reload model from {}".format(checkpoint_path))
    model.to(device)
    logger.info(f"  Finish loading model {get_model_size(model)} from {config['model_type']}")
    logger.info("  ***** Model is loaded successfully!  ***** ")

    if config["n_gpu"] > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    
    is_validation_data_loaded = False
    fa = open(os.path.join(output_path, 'summary.log'), 'a+')
    # train model
    if config["do_train"]:
        train_data, _ = load_and_preprocess_dataset(config=config, split_tag="train", tokenizer=tokenizer, is_calculate_stats=True)
        logging.debug(train_data)
        logging.debug("  ***** Training data is tokenized successfully!  ***** ")
        
        train_sampler = RandomSampler(train_data) if config["local_rank"] == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, num_workers=4, batch_size=get_batch_size(config=config, split="train"), pin_memory=True)
        logging.info("  ***** Training data is loaded successfully!  ***** \n")
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': config["weight_decay"]},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config["learning_rate"], eps=config["adam_epsilon"])
        num_train_optimization_steps = config["num_train_epochs"] * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=config["num_warmup_steps"],
                                                    num_training_steps=num_train_optimization_steps)
        
        # Start training
        train_example_num = len(train_data)
        logger.info("  ***** Running training  ***** ")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", get_batch_size(config=config, split="train"))
        logger.info("  Batch num = %d", math.ceil(train_example_num / get_batch_size(config=config, split="train")))
        logger.info("  Num epoch = %d", config["num_train_epochs"])

        dev_dataset = {}
        global_step, best_bleu_em = 0, -1
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if config["do_validation"] else 1e6
        
        for cur_epoch in range(config["start_epoch"], config["num_train_epochs"]):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()

            for step, batch in enumerate(bar):
                if config["model_type"] in ("codebert", "roberta", "bert"):
                    batch["source_attention_mask"] = batch["source_attention_mask"].type(torch.bool)
                    batch["target_attention_mask"] = batch["target_attention_mask"].type(torch.bool)        
                
                batch = {k: v.to(device) for k, v in batch.items()}

                if config["model_type"] in ("codebert", "roberta"):
                    loss, _, _ = model(source_ids=batch["source_ids"], source_mask=batch["source_attention_mask"],
                                        target_ids=batch["target_ids"], target_mask=batch["target_attention_mask"])                
                elif config["model_type"] in ("codet5"):
                    outputs = model(input_ids=batch["source_ids"], attention_mask=batch["source_attention_mask"],
                                    labels=batch["target_ids"], decoder_attention_mask=batch["target_attention_mask"])
                    loss = outputs.loss

                if config["n_gpu"] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if config["gradient_accumulation_steps"] > 1:
                    loss = loss / config["gradient_accumulation_steps"]
                tr_loss += loss.item()
                nb_tr_examples += batch["source_ids"].size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % config["gradient_accumulation_steps"] == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * config["gradient_accumulation_steps"] / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if config["do_validation"]:
                if (cur_epoch+1) % config["eval_every_n_epoch"] == 0:
                    if is_validation_data_loaded == False:
                        validation_data, validation_examples = load_and_preprocess_dataset(config=config, split_tag="validation", tokenizer=tokenizer, is_calculate_stats=False)
                        is_validation_data_loaded = True
                        logger.debug(validation_data)
                        logger.info("  ***** Validation data is loaded successfully!  ***** \n")
                    
                    result = compute_bleu_score(config=config, 
                                                dataset=validation_data,
                                                examples=validation_examples,
                                                model=model,
                                                tokenizer=tokenizer,
                                                split_tag="validation",
                                                cur_epoch=cur_epoch,
                                                output_path=output_path,
                                                device=device)
                    
                    dev_bleu, dev_em = result['bleu'], result['em']
                    dev_bleu_em = dev_bleu + dev_em
                    
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)", cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        # logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        

                        if config["save_checkpoint"] and checkpoint_path != None:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), checkpoint_path)
                            logger.info("  Save the best bleu model into %s\n", checkpoint_path)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("  Bleu does not increase for %d epochs\n", not_bleu_em_inc_cnt)
                        if all([x > config["patience"] for x in [not_bleu_em_inc_cnt]]):
                            stop_early_str = "  [%d] Early stop as not_bleu_em_inc_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break


    # if config["is_test_gold"] or config["is_test_intel"]:
    #     logger.info(" ***** Testing ***** ")
    #     # logger.info("  Batch size = %d", get_batch_size("test"))

    #     if config["is_test_gold"]:
    #         test_gold = load_and_preprocess_dataset(split_tag="test_gold", tokenizer=tokenizer, is_calculate_stats=False)
    #     if config["is_test_intel"]:
    #         test_intel = load_and_preprocess_dataset(split_tag="test_intel", tokenizer=tokenizer, is_calculate_stats=False)
    
    #     logger.info("  ***** Test data is loaded successfully!  ***** \n")

    #     for dataset, split_tag in zip([test_gold, test_intel], ["test_gold", "test_intel"]):
    #         data, examples = dataset
  
    #         result = compute_bleu_score(config=config,
    #                                     dataset=data, 
    #                                     examples=examples, 
    #                                     model=model, 
    #                                     tokenizer=tokenizer, 
    #                                     split_tag=split_tag, 
    #                                     cur_epoch=split_tag, 
    #                                     output_path=output_path, 
    #                                     device=device)

    #         test_bleu, test_em = result['bleu'], result['em']
    #         result_str = "[%s] bleu-4: %.2f, em: %.4f\n" % (split_tag, test_bleu, test_em)
    #         logger.info(result_str)
    #         fa.write(result_str)
            
    #         fpath = os.path.join(output_path, "test_result.out")
    #         with open(fpath, 'a+') as f:
    #             f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), checkpoint_path))
    #             f.write(result_str)
    
    # logger.info("Finish and take {}".format(get_elapse_time(t0)))
    # fa.write("Finish and take {}".format(get_elapse_time(t0)))
    # fa.close()          

if __name__ == "__main__":
    main()