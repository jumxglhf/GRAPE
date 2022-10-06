# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import numpy as np
import torch
import json
import os
import dgl


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.f = self.title_prefix + " {}. " + self.passage_prefix + " {}"

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            return example['target']
        elif 'answers' in example:
            return random.choice(example['answers']) # + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        return {
            'index' : index,
            'question' : self.question_prefix + " " + self.data[index]['question'],
            'target' : self.get_target(self.data[index]),
            'passages' : [self.f.format(c['title'], c['text']) for c in self.data[index]['ctxs'][:self.n_context]],
            'question_entities':self.data[index]['linked_question_entity'],
            'question_entities_link_offset':[e+len(self.question_prefix)+1 for e in self.data[index]['link_offset']],
            'question_entities_link_length':self.data[index]['link_length'],
            'passage_related_question_entity':[c['related_question_entity'] for c in self.data[index]['ctxs'][:self.n_context]],
            'passage_related_passage_entity':[c['related_passage_entity'] for c in self.data[index]['ctxs'][:self.n_context]],
            'graph' :self.data[index]['graph'],
            'real_index' :self.data[index]['real_index'],
        }

    def get_example(self, index):
        return self.data[index]

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, for_eval=False):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        identifiers = tokenizer.encode('<PM_LEFT> <PM_RIGHT> <QM_LEFT> <QM_RIGHT>')[:-1]
        self.pes, self.pee, self.qes, self.qee = identifiers[0], identifiers[1], identifiers[2], identifiers[3]
        self.for_eval = for_eval

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            output = []
            for i, t in enumerate(example['passages']):
                if example['passage_related_question_entity'][i] == []:
                    output.append(example['question'] + " " + t)
                else:
                    offset = [e for j, e in enumerate(example['question_entities_link_offset']) if j in example['passage_related_question_entity'][i]]
                    link_length = [e for j, e in enumerate(example['question_entities_link_length']) if j in example['passage_related_question_entity'][i]]
                    output.append(insert_markers_psg(example['question'], offset, link_length)[0] + " " + t)
            return output

        
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)
        n_qes = []
        for g_id in range(len(batch)):
            this_n_qes = 0
            for p_id in range(len(text_passages[0])):
                this_n_qes += len(batch[g_id]['passage_related_question_entity'][p_id])
            n_qes.append(this_n_qes)
        
        total_graph_neighbors = []
        for g_id in range(len(batch)):
            batch_graph_neighbors = []
            accumulated_offset = n_qes[g_id]
            for p_id in range(len(text_passages[0])):
                n_indices = [e+accumulated_offset for e, _ in enumerate(batch[g_id]['passage_related_passage_entity'][p_id])]
                batch_graph_neighbors.append(n_indices)
                accumulated_offset += len(n_indices)
            total_graph_neighbors.append(batch_graph_neighbors)

        graphs = [e['graph'].clone() for e in batch]
        node_indices = []
        for graph_index, this_example_ids in enumerate(passage_ids):
            nodes_to_remove = []
            node_indice = {'question_mention':[],
                           'passage_mention':[]}
            if batch[graph_index]['graph'].num_nodes != 0:
                for passage_index, this_passage_ids in enumerate(this_example_ids):
                    idx_qs = (this_passage_ids == self.qes).nonzero(as_tuple=True)[0]
                    idx_qe = (this_passage_ids == self.qee).nonzero(as_tuple=True)[0]
                    idx_ps = (this_passage_ids == self.pes).nonzero(as_tuple=True)[0]
                    idx_pe = (this_passage_ids == self.pee).nonzero(as_tuple=True)[0]
                    graph_neighbors = total_graph_neighbors[graph_index][passage_index]
                    if len(idx_pe) != len(idx_ps):
                        idx_ps = idx_ps[:len(idx_pe)]
                    if len(idx_pe) != len(graph_neighbors):
                        nodes_to_remove.append(torch.tensor(graph_neighbors[len(idx_pe):]))
                    for qs, qe in zip(idx_qs.tolist(), idx_qe.tolist()):
                        node_indice['question_mention'] += [[passage_index, qs, qe]]
                    for ps, pe in zip(idx_ps.tolist(), idx_pe.tolist()):
                        node_indice['passage_mention'] += [[passage_index, ps, pe]]
                if len(nodes_to_remove) != 0:
                    graphs[graph_index].remove_nodes(torch.cat(nodes_to_remove))
            node_indices.append(node_indice)

        if self.for_eval:
            return (index, target_ids, target_mask, passage_ids, passage_masks, graphs, node_indices, [e['real_index'] for e in batch])

        return (index, target_ids, target_mask, passage_ids, passage_masks, graphs, node_indices)


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        sp = p['input_ids'][None].shape
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])
        
    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


def split_data(jsonl_path, graph_path, local_rank, world_size, graph_block_size=5000):
    train_data = []
    data_name = graph_path.split('/')[-1]
    graph_path = graph_path[:graph_path.index(data_name)]
    prefixed = [filename for filename in os.listdir(graph_path) if filename.startswith(data_name)]

    def graph_sprt(file_name):
        index = int(file_name.split('_')[-1].split('.')[0])
        return index
        
    prefixed.sort(key=graph_sprt)
    current_block_index = 0
    current_graph_block, _ = dgl.load_graphs(graph_path + '/' + prefixed[current_block_index])
    with open(jsonl_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            if index == 76367:
                # this is designed for a problem sample in tqa
                continue
            if index % world_size == local_rank:
                train_data.append(json.loads(line))
                if index >= (current_block_index+1)*graph_block_size:
                    current_block_index += 1
                    current_graph_block, _ = dgl.load_graphs(graph_path + '/'+ prefixed[current_block_index])
                train_data[-1]['graph'] = current_graph_block[index%graph_block_size]
                train_data[-1]['real_index'] = index
    return train_data

def insert_markers_psg(passage_text, gold_link_offsets, gold_link_length):
    accumulated_offsets = 0
    start_marker = '<QM_LEFT>'
    end_marker   = '<QM_RIGHT>'
    new_gold_link_offsets = []
    for offset, length in zip(gold_link_offsets, gold_link_length):
        current_offset = offset+accumulated_offsets
        if len(end_marker) != 0:
            passage_text = passage_text[:current_offset] + start_marker + ' ' \
            + passage_text[current_offset:current_offset+length] + ' ' + end_marker + passage_text[current_offset+length:]

            new_gold_link_offsets.append(offset + accumulated_offsets + len(start_marker) + 1)
            accumulated_offsets += len(start_marker) + len(end_marker) + 2
        else:
            passage_text = passage_text[:current_offset] + start_marker + ' ' \
            + passage_text[current_offset:]
            new_gold_link_offsets.append(offset + accumulated_offsets + len(start_marker) + 1)
            accumulated_offsets += len(start_marker) + 1
        
    return passage_text, new_gold_link_offsets