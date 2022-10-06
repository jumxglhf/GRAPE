# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import errno
import torch
import sys
import logging
import json
import types
from pathlib import Path
import torch.distributed as dist
import csv
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
import dgl


logger = logging.getLogger(__name__)

def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger

def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, checkpoint_exists

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def save(model, optimizer, scheduler, step, best_eval_metric, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name) #"step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)
    with open('{}/gnn_config.json'.format(epoch_path), 'w') as fp: json.dump(model_to_save.gnn_config, fp, sort_keys=True, indent=4)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
        "best_eval_metric": best_eval_metric,
    }
    torch.save(checkpoint, fp)
    # symlink_force(epoch_path, cp)


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)
    gnn_config = json.load(open(epoch_path + '/gnn_config.json'))
    model = model_class.from_pretrained(epoch_path, **gnn_config)
    model = model.to(opt.device)
    logger.info("loading checkpoint %s" %optimizer_path)
    checkpoint = torch.load(optimizer_path, map_location=opt.device)
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    if "best_eval_metric" in checkpoint:
        best_eval_metric = checkpoint["best_eval_metric"]
    else:
        best_eval_metric = checkpoint["best_dev_em"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric

class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio)*step/float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
            1.0 + (self.min_ratio - 1) * (step - self.warmup_steps)/float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_steps
        else:
            scheduler_steps = opt.scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0., fixed_lr=opt.fixed_lr)
    return optimizer, scheduler


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    with open(output_path, 'w') as outfile:
        for path in files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / 'tmp_dir'
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f'{opt.global_rank}.json'
    with open(tmp_path, 'w') as fw:
        json.dump(data, fw)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / 'dataset_wscores.json'
        logger.info(f'Writing dataset with scores at {final_path}')
        glob_path = write_path / '*'
        results_path = write_path.glob('*.json')
        alldata = []
        for path in results_path:
            with open(path, 'r') as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, 'w') as fout:
            json.dump(alldata, fout, indent=4)
        write_path.rmdir()

def load_passages(path):
    if not os.path.exists(path):
        logger.info(f'{path} does not exist')
        return
    logger.info(f'Loading passages from: {path}')
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    logger.warning(f'The following input line has not been correctly loaded: {row}')
    return passages

def truncate_graph_wrt_n_passage(graph, node_indices, n_passage):
    if n_passage == 100:
        return graph, node_indices
        
    mention_nodes_to_remove = []
    new_mention_indices = []
    for index, triple in enumerate(node_indices['mention_nodes']):
        if triple[0] >= n_passage:
            mention_nodes_to_remove.append(index)
        else:
            new_mention_indices.append(triple)
    graph.remove_nodes(mention_nodes_to_remove, ntype = 'mention')
    graph.remove_nodes(list(range(n_passage, 100)), ntype = 'passage')
    node_indices['mention_nodes'] = new_mention_indices
    node_indices['passage_nodes'] = node_indices['passage_nodes'][:n_passage]
    return node_indices

# freeze the parameters of t5
def freeze_t5(model):
    model = model.module if hasattr(model, "module") else model
    for name, child in model.named_children():
        if name == 'gnn_model':
            continue
        for param in child.parameters():
            param.requires_grad = False

# unfreeze the parameters of t5
def unfreeze_t5(model):
    model = model.module if hasattr(model, "module") else model
    for name, child in model.named_children():
        if name == 'gnn_model':
            continue
        for param in child.parameters():
            param.requires_grad = True
    
# overwrite the forward method of a T5Stack Encoder
def overwrite_t5stack_forward(t5_stack):

    # forward with gnn embedding 
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            gnn_fid=None,
            layer2insert=None,
            graphs=None,
            node_indices=None,
            relation_memory=None,
            graph_mode = True
        ):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(self.first_device)
                self.embed_tokens = self.embed_tokens.to(self.first_device)
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                err_msg_prefix = "decoder_" if self.is_decoder else ""
                raise ValueError(
                    f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
                )
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                err_msg_prefix = "decoder_" if self.is_decoder else ""
                raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

            if inputs_embeds is None:
                assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
                inputs_embeds = self.embed_tokens(input_ids)

            batch_size, seq_length = input_shape

            # required mask seq length can be calculated via length of past
            mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

            if use_cache is True:
                assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
            if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
                encoder_seq_length = encoder_hidden_states.shape[1]
                encoder_attention_mask = torch.ones(
                    batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
                )

            # initialize past_key_values with `None` if past does not exist
            if past_key_values is None:
                past_key_values = [None] * len(self.block)

            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None

            # Prepare head mask if needed
            head_mask = self.get_head_mask(head_mask, self.config.num_layers)
            cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
            present_key_value_states = () if use_cache else None
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            all_cross_attentions = () if (output_attentions and self.is_decoder) else None
            position_bias = None
            encoder_decoder_position_bias = None

            hidden_states = self.dropout(inputs_embeds)
            # hidden_states_ = hidden_states.view(len(graphs), gnn_fid.encoder.n_passages, gnn_fid.encoder.text_length, -1)
            # for idx, graph in enumerate(graphs):
            #     feat = gnn_fid.graph_attriution_before(hidden_states_[idx], node_indices[idx], question_entities[idx], passage_entities[idx], graph, input_ids[0][2].unsqueeze(0))
            #     hidden_states_[idx] = gnn_fid.graph_attriution_after(hidden_states_[idx],
            #                                                             gnn_fid.gnn_model(graph, feat, len(question_entities[idx]), gnn_fid.encoder.n_passages), 
            #                                                             node_indices[idx])
            # hidden_states = hidden_states_.view(hidden_states.shape)

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
                layer_head_mask = head_mask[i]
                cross_attn_layer_head_mask = cross_attn_head_mask[i]
                # Model parallel
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if position_bias is not None:
                        position_bias = position_bias.to(hidden_states.device)
                    if encoder_hidden_states is not None:
                        encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                    if encoder_extended_attention_mask is not None:
                        encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                    if encoder_decoder_position_bias is not None:
                        encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                    if layer_head_mask is not None:
                        layer_head_mask = layer_head_mask.to(hidden_states.device)
                    if cross_attn_layer_head_mask is not None:
                        cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    if use_cache:
                        logger.warning(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return tuple(module(*inputs, use_cache, output_attentions))

                        return custom_forward

                    layer_outputs = checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        extended_attention_mask,
                        position_bias,
                        encoder_hidden_states,
                        encoder_extended_attention_mask,
                        encoder_decoder_position_bias,
                        layer_head_mask,
                        cross_attn_layer_head_mask,
                        None,  # past_key_value is always None with gradient checkpointing
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=extended_attention_mask,
                        position_bias=position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
                if use_cache is False:
                    layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                hidden_states, present_key_value_state = layer_outputs[:2]
                if (i+1) == layer2insert and graph_mode:
                    hidden_states_ = hidden_states.view(len(graphs), gnn_fid.encoder.n_passages, gnn_fid.encoder.text_length, -1)
                    for idx, graph in enumerate(graphs):
                        if graph.num_nodes() != 0 and len(node_indices[idx]['passage_mention']) !=0 and len(node_indices[idx]['question_mention']) !=0:
                            feat = gnn_fid.graph_attriution_before(hidden_states_[idx], node_indices[idx])
                            hidden_states_[idx] = gnn_fid.graph_attriution_after(hidden_states_[idx],
                                                                                gnn_fid.gnn_model(graph, feat, relation_memory, node_indices[idx]), 
                                                                                node_indices[idx])
                    hidden_states = hidden_states_.view(hidden_states.shape)

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
                # (cross-attention position bias), (cross-attention weights)
                position_bias = layer_outputs[2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + (present_key_value_state,)

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[3],)
                    if self.is_decoder:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)
            
            # Add last layer
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        present_key_value_states,
                        all_hidden_states,
                        all_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )

    t5_stack.forward = types.MethodType(forward, t5_stack)

def missing_grad_handler(model):
    temp = 0
    for k, v in model.named_parameters():
        temp += v.sum()
    return temp*0.
    