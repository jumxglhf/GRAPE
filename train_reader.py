# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

import time
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.options import Options
import multiprocessing as mp
import src.util
import src.evaluation
import src.data
import src.model
import dgl
import json


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, test_dataset, opt, collator, best_dev_em, checkpoint_path, relation_bank):

    torch.manual_seed(opt.local_rank + opt.seed) #different seed for different sampling depending on global_rank
    
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    model.zero_grad()
    inner_step = 0
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            inner_step += 1
            (idx, labels, target_mask, context_ids, context_mask, graphs, node_indices) = batch
            train_loss = model(
                input_ids=context_ids.to(opt.device),
                attention_mask=context_mask.to(opt.device),
                labels=labels.to(opt.device),
                graphs = [g.to(opt.device) for g in graphs],
                node_indices = node_indices,
                relation_bank_ids = relation_bank['input_ids'].to(opt.device),
                relation_bank_masks = relation_bank['attention_mask'].to(opt.device),
            )[0]

            train_loss.backward()
            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if inner_step % opt.accumulation_steps == 0:
                step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                if opt.wandb and opt.is_main:
                    wandb.log({"dynamic training loss": train_loss.item()})

                if step % opt.eval_freq == 0:
                    dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt, relation_bank)
                    test_em = 0
                    if test_dataset != None:
                        collator.for_eval = True
                        test_em = evaluate(model, test_dataset, tokenizer, collator, opt, relation_bank, ifTest=True, step=step, checkpoint_path=checkpoint_path)
                        collator.for_eval = False
                        # ----------------
                    if opt.is_main:
                        if dev_em > best_dev_em:
                            best_dev_em = dev_em
                            src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                    opt, checkpoint_path, 'best_dev')
                        log = f"{step} / {opt.total_steps} |"
                        log += f"train: {curr_loss/(opt.eval_freq*opt.accumulation_steps):.3f} |"
                        log += f"dev evaluation: {100*dev_em:.2f} EM |"
                        log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                        if test_dataset != None:
                            log += f"test evaluation: {100*test_em:.2f} EM |"
                            model_outputs = {}
                            with open(checkpoint_path / 'test_outputs_{}.jsonl'.format(str(step)), encoding='utf8') as f:
                                for line in f:
                                    line = json.loads(line)
                                    model_outputs[line['id']] = line

                            ground_truth = json.load(open(opt.ground_truth_for_test, encoding='utf8'))
                            total_onehop_solvable = [idx for idx, element in ground_truth.items() if element['1hop_solvable']]
                            correct = []
                            for _, element in model_outputs.items():
                                if src.evaluation.normalize_answer(element['hyp']) in [src.evaluation.normalize_answer(e) for e in element['gold']]:
                                    correct.append(str(element['id']))
                            my_onehop_solvable = []
                            for correcdt_idx in correct:
                                if ground_truth[correcdt_idx]['1hop_solvable']:
                                    my_onehop_solvable.append(correcdt_idx)
                            log += '| 1-hop solvables: {:.4f} ({}/{})'.format(len(my_onehop_solvable)/len(total_onehop_solvable), len(my_onehop_solvable), len(total_onehop_solvable))
                        
                        logger.info(log)    
                        curr_loss = 0.

                        if opt.wandb:
                            wandb.log({"dev EM": dev_em, "test EM":test_em })

                model.train()
                if opt.is_main and step % opt.save_freq == 0:
                    src.util.save(model, optimizer, scheduler, step, best_dev_em,
                            opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt, relation_bank, ifTest = False, step=0, checkpoint_path=None):
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size*2,
        drop_last=False,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():        
        for i, batch in enumerate(dataloader):
            if ifTest:
                (idx, labels, target_mask, context_ids, context_mask, graphs, node_indices, real_index) = batch
            else:
                (idx, labels, target_mask, context_ids, context_mask, graphs, node_indices) = batch

            outputs = model.generate(input_ids=context_ids.to(opt.device),
                        attention_mask=context_mask.to(opt.device), 
                        graphs = [g.to(opt.device) for g in graphs],
                        node_indices = node_indices,
                        max_length = 10,
                        relation_bank_ids = relation_bank['input_ids'].to(opt.device),
                        relation_bank_masks = relation_bank['attention_mask'].to(opt.device),
                        )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)
                if ifTest:
                    with open(checkpoint_path / 'test_outputs_{}.jsonl'.format(str(step)), mode='a', encoding='utf-8') as f:
                        f.write(json.dumps({'id':real_index[k]+1, 'hyp':ans, 'gold':gold}) + "\n")

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)

    return exactmatch

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    if opt.use_mpi:
        from src.mpi import MPIAdapter
        adapter = MPIAdapter()
        adapter.init_process_group(backend="nccl")
        adapter.log_info()

    if opt.is_distributed:
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        if opt.local_rank == 0:
            opt.is_main = True
        else:
            opt.is_main = False
        opt.device = "cuda:{}".format(opt.local_rank)
        opt.world_size = torch.cuda.device_count()
        torch.distributed.init_process_group(backend="nccl", world_size=opt.world_size, rank=opt.local_rank)
        torch.cuda.set_device(opt.local_rank)
    else:
        opt.device = "cuda:0"
        opt.is_main = True
        opt.local_rank = 0
        opt.world_size = 1

    dgl.seed(opt.seed)
    torch.manual_seed(opt.seed)
    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed, # is_distributed=
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    # add special tokens 
    additional_special_tokens = ['<CLS_GRAPH>', '<QM_LEFT>', '<QM_RIGHT>', '<PM_LEFT>', '<PM_RIGHT>']
    tokenizer.add_tokens(additional_special_tokens)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # ------------------------- BEGIN DATA LOADING -------------------------

    logger.info(f"Loading and pre-processing training data")
    train_examples = src.data.split_data(opt.train_data, opt.train_graph, opt.local_rank, opt.world_size)
    train_dataset = src.data.Dataset(train_examples, n_context=opt.n_context)

    logger.info(f"Loading and pre-processing dev data")
    eval_examples = src.data.split_data(opt.eval_data, opt.eval_graph, opt.local_rank, opt.world_size)
    eval_dataset = src.data.Dataset(eval_examples, n_context=opt.n_context)

    if opt.test_data == 'none':
        logger.info(f"No test data is given")
        test_dataset = None
    else:
        logger.info(f"Loading and pre-processing test data")
        test_examples = src.data.split_data(opt.test_data, opt.test_graph, opt.local_rank, opt.world_size)
        test_dataset = src.data.Dataset(test_examples, n_context=opt.n_context)
    
    relation_bank = list(json.load(open(opt.relation_base, encoding='utf8')).values())
    relation_bank = tokenizer(relation_bank, padding='longest', return_tensors='pt')
    # ------------------------- END OF DATA LOADING -------------------------

    # ------------------------- MODEL LOADING/INITILIZATION -------------------------

    if not checkpoint_exists or opt.model_path == "none":
        logger.info(f"Initializing model")
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        t5.resize_token_embeddings(len(tokenizer))
        model_d = t5.model_dim
        gnn_config = {'in_feat':model_d, 'hid_feat':model_d, 'n_heads':opt.n_heads, 'mode':opt.gnn_mode,
                      'n_layer':opt.gnn_layer,'layer2insert':opt.layer2insert, 'dropout':opt.gnn_dropout, 'bpe':opt.bpe}
        logger.info("GNN CONFIG: "+json.dumps(gnn_config, indent=2))
        model = src.model.FiDT5(t5.config, **gnn_config)
        model.load_t5(t5.state_dict())
        del t5
        model = model.to(opt.device)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
        
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    # ------------------------- END OF MODEL LOADING/INITILIZATION -------------------------
    model.set_checkpoint(opt.use_checkpoint)
    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=False)
    if opt.wandb and opt.is_main:
        import wandb
        wandb.init(project="", entity="")
        wandb.config = opt
    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        test_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path,
        relation_bank
    )
