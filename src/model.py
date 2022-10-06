# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
# import dgl.nn.pytorch as dglnn
import dgl
import src.util

class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config, **gnn_config):
        super().__init__(config)
        src.util.overwrite_t5stack_forward(self.encoder)
        self.wrap_encoder()
        self.gnn_model = self.create_gnn(**gnn_config)
        # self.gnn_model2 = self.create_gnn(**gnn_config)
        self.gnn_config = gnn_config

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, graphs=None, node_indices=None, relation_bank_ids=None, relation_bank_masks=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
                self.encoder.text_length = input_ids.size(2)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        
        return self.forward_with_gnn(
            input_ids       =input_ids,
            attention_mask  =attention_mask,
            graphs          =graphs,
            node_indices    =node_indices,
            relation_bank_ids=relation_bank_ids, 
            relation_bank_masks=relation_bank_masks,
            **kwargs
        )

    def forward_with_gnn(
        self,
        input_ids, 
        attention_mask,
        graphs,
        node_indices,
        relation_bank_ids=None, 
        relation_bank_masks=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
        ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            relation_memory = self.encoder(
                input_ids=relation_bank_ids,
                attention_mask=relation_bank_masks,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                graph_mode=False,
                graph_mode_=False,
                **kwargs
            )[0]
            mask = relation_bank_masks.float()/(relation_bank_masks.float().sum(1).unsqueeze(-1))
            relation_memory = torch.multiply(mask.unsqueeze(-1), relation_memory).sum(1).detach()
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                gnn_fid = self,
                layer2insert=self.gnn_config['layer2insert'],
                graphs=graphs,
                node_indices=node_indices,
                relation_memory=relation_memory,
                **kwargs
            )
            hidden_states = encoder_outputs[0]
            encoder_outputs.last_hidden_state = hidden_states
            encoder_outputs.attention_mask = attention_mask

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            attention_mask = encoder_outputs.attention_mask
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0]
            )
            encoder_outputs.attention_mask = attention_mask

        # both generation and training use this! 
        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoder_outputs.attention_mask
        

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss += src.util.missing_grad_handler(self.gnn_model)
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(self, input_ids, attention_mask, graphs, node_indices, max_length, relation_bank_ids, relation_bank_masks, 
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):

        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
                self.encoder.text_length = input_ids.size(2)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        # get embeddings of passages 
        relation_memory = self.encoder(
                input_ids=relation_bank_ids,
                attention_mask=relation_bank_masks,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                graph_mode=False,
                graph_mode_=False,
            )[0]
        mask = relation_bank_masks.float()/(relation_bank_masks.float().sum(1).unsqueeze(-1))
        relation_memory = torch.multiply(mask.unsqueeze(-1), relation_memory).sum(1)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gnn_fid = self,
            layer2insert=self.gnn_config['layer2insert'],
            graphs=graphs,
            node_indices=node_indices,
            relation_memory=relation_memory,
        )
        hidden_states = encoder_outputs[0]
        encoder_outputs.last_hidden_state = hidden_states
        encoder_outputs.attention_mask = attention_mask
        
        hidden_states = encoder_outputs[0]
        encoder_outputs.last_hidden_state = hidden_states
        encoder_outputs.attention_mask = attention_mask

        # package encoder outputs and gnn outputs for generation 
        return super().generate(encoder_outputs = encoder_outputs, max_length = max_length)

    def gnn_warm_up(self, input_ids=None, attention_mask=None, graphs = None, node_indices = None):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
                self.encoder.text_length = input_ids.size(2)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        hidden_states = encoder_outputs[0]
        batched_graphs = []
        for i, graph in enumerate(graphs):
            batched_graphs.append(self.graph_attriution(hidden_states[i].detach(), node_indices[i], graph))
        return batched_graphs


    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict = False)
        self.wrap_encoder()

    def load_model(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint


    # attribute the given graph with encoder output and LM embedding 
    def graph_attriution_before(self, representation, node_indices):

        if self.gnn_config['bpe']:
            representation = torch.clone(representation.view(self.encoder.n_passages, self.encoder.text_length, -1))
        else:
            representation = representation.view(self.encoder.n_passages, self.encoder.text_length, -1).detach()

        mention_nodes_context_feat = self.efficient_extract(representation, node_indices['passage_mention']) # torch.stack([representation[lst[0]][(lst[1]+1):lst[2]].mean(dim = 0) for lst in node_indices['passage_mention']])
        question_mention_nodes_context_feat = self.efficient_extract(representation, node_indices['question_mention']) # torch.stack([representation[lst[0]][(lst[1]+1):lst[2]].mean(dim = 0) for lst in node_indices['question_mention']])
        return torch.cat([question_mention_nodes_context_feat, mention_nodes_context_feat], dim=0)

    def graph_attriution_after(self, representation_lm, representation_gnn, node_indices):

        question_mention = torch.tensor(node_indices['question_mention'])
        representation_lm[question_mention[:,0], question_mention[:,1]] += representation_gnn['qe_left']
            # representation_lm[question_mention[:,0], question_mention[:,2]] += representation_gnn['qe_right']
        passage_mention = torch.tensor(node_indices['passage_mention'])
        representation_lm[passage_mention[:,0], passage_mention[:,1]] += representation_gnn['pe_left']
            # representation_lm[passage_mention[:,0], passage_mention[:,2]] += representation_gnn['pe_right']
        return representation_lm

    def create_gnn(self, in_feat, hid_feat, n_layer, dropout, n_heads, mode, **kwargs):
        # in_feats, hid_global, hid_feats, ntypes, n_bases, dropout, n_layers,
        if mode == 'EGAT':
            return GNN(in_feats=in_feat, hid_feats=hid_feat, n_layers=n_layer, dropout=dropout, n_heads=n_heads)

        elif mode == 'NO_RELATION':
            return GNN_no_relation(in_feats=in_feat, hid_feats=hid_feat, n_layers=n_layer, dropout=dropout, n_heads=n_heads)

        elif mode == 'NO_ATTENTION':
            return GNN_no_attention(in_feats=in_feat, hid_feats=hid_feat, n_layers=n_layer, dropout=dropout, n_heads=n_heads)

    def efficient_extract(self, representation, indices):
        mask = torch.zeros(len(indices), self.encoder.n_passages, self.encoder.text_length) # .to(representation.device)
        for i, lst in enumerate(indices): mask[i][lst[0]][(lst[1]+1):lst[2]] = 1
        mask = mask.view(len(indices), -1).to(representation.device)
        mask = torch.div(mask, (mask.sum(1)).unsqueeze(1))
        return torch.matmul(mask, representation.view(-1, representation.shape[-1]))



class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)
        self.main_input_name = None

    def forward(self, input_ids=None, attention_mask=None, graph_mode_=True, **kwargs,):
        # total_length = n_passages * passage_length
        if not graph_mode_:
            return self.encoder(input_ids, attention_mask, **kwargs)
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages*passage_length, -1)
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


class GNN(nn.Module):
    def __init__(self, in_feats, hid_feats, dropout, n_layers, n_heads,
                activation_function = torch.nn.GELU()):
        super().__init__()

        # heterogeneous message passing layers 
        self.layers = nn.ModuleList([
            dgl.nn.EGATConv(in_node_feats=in_feats, in_edge_feats=in_feats, out_node_feats=hid_feats, out_edge_feats=128, num_heads=n_heads)
            if l == 0 else
            dgl.nn.EGATConv(in_node_feats=hid_feats, in_edge_feats=in_feats, out_node_feats=hid_feats, out_edge_feats=128, num_heads=n_heads)
            for l in range(n_layers)
        ])

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, h_prev, relation_memory, node_indices):
        # layer-wise message passing
        e_feat = relation_memory[graph.edata['etype']]
        graph = graph.add_self_loop()
        e_feat = torch.cat([e_feat, torch.zeros(graph.num_nodes(), e_feat.shape[-1]).to(e_feat.device)], dim=0)
        e_feat = self.dropout(e_feat)
        for index, layer in enumerate(self.layers):
            # message passing
            h_prev, _ = layer(graph, h_prev, e_feat)
            h_prev = self.dropout(h_prev)
            h_prev = h_prev.sum(1)

        # final projection
        output = {}
        n_pes = len(node_indices['question_mention'])
        output['qe_left'] = h_prev[:n_pes] # self.projection_layers_output['mention_left'](h_prev[n_passages:n_passages+n_passages*n_question_entities])
        output['qe_right'] = h_prev[:n_pes] # self.projection_layers_output['mention_right'](h_prev[n_passages:n_passages+n_passages*n_question_entities])
        output['pe_left'] = h_prev[n_pes:] # self.projection_layers_output['mention_left'](h_prev[n_passages+n_passages*n_question_entities:])
        output['pe_right'] = h_prev[n_pes:] # self.projection_layers_output['mention_right'](h_prev[n_passages+n_passages*n_question_entities:])
        return output 

class GNN_GINE(nn.Module):
    def __init__(self, in_feats, hid_feats, dropout, n_layers, n_heads,
                activation_function = torch.nn.GELU()):
        super().__init__()

        # heterogeneous message passing layers 
        self.layers = nn.ModuleList([dgl.nn.GINEConv(nn.Linear(in_feats, hid_feats))])
        for _ in range(n_layers - 2):
            self.layers.append(dgl.nn.GATv2Conv(hid_feats, hid_feats, num_heads=n_heads))
        self.layers.append(dgl.nn.GATv2Conv(hid_feats, in_feats, num_heads=n_heads))

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_function

    def forward(self, graph, h_prev, relation_memory, node_indices):
        # layer-wise message passing
        e_feat = relation_memory[graph.edata['etype']]
        graph = graph.add_self_loop()
        e_feat = torch.cat([e_feat, torch.zeros(graph.num_nodes(), e_feat.shape[-1]).to(e_feat.device)], dim=0)
        e_feat = self.dropout(e_feat)
        e_feat = self.dropout(e_feat)
        for index, layer in enumerate(self.layers):
            # message passing
            if index == 0:
                h_prev = self.activation(layer(graph, h_prev, e_feat))
                h_prev = self.dropout(h_prev)
            else:
                h_prev = self.activation(layer(graph, h_prev))
                h_prev = self.dropout(h_prev)
                h_prev = h_prev.sum(1)

        # final projection
        output = {}
        n_pes = len(node_indices['question_mention'])
        output['qe_left'] = h_prev[:n_pes] # self.projection_layers_output['mention_left'](h_prev[n_passages:n_passages+n_passages*n_question_entities])
        output['qe_right'] = h_prev[:n_pes] # self.projection_layers_output['mention_right'](h_prev[n_passages:n_passages+n_passages*n_question_entities])
        output['pe_left'] = h_prev[n_pes:] # self.projection_layers_output['mention_left'](h_prev[n_passages+n_passages*n_question_entities:])
        output['pe_right'] = h_prev[n_pes:] # self.projection_layers_output['mention_right'](h_prev[n_passages+n_passages*n_question_entities:])
        return output 

class GNN_no_relation(nn.Module):
    def __init__(self, in_feats, hid_feats, dropout, n_layers, n_heads,
                activation_function = torch.nn.GELU()):
        super().__init__()

        # heterogeneous message passing layers 
        self.layers = nn.ModuleList([dgl.nn.GATConv(in_feats, hid_feats, num_heads=n_heads)])
        for _ in range(n_layers - 2):
            self.layers.append(dgl.nn.GATConv(hid_feats, hid_feats, num_heads=n_heads))
        self.layers.append(dgl.nn.GATConv(hid_feats, in_feats, num_heads=n_heads))

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_function

    def forward(self, graph, h_prev, relation_memory, node_indices):
        graph = graph.add_self_loop()
        for index, layer in enumerate(self.layers):
            # message passing
            h_prev = layer(graph, h_prev)
            h_prev = self.dropout(h_prev)
            h_prev = h_prev.sum(1)

        # final projection
        output = {}
        n_pes = len(node_indices['question_mention'])
        output['qe_left'] = h_prev[:n_pes] # self.projection_layers_output['mention_left'](h_prev[n_passages:n_passages+n_passages*n_question_entities])
        output['qe_right'] = h_prev[:n_pes] # self.projection_layers_output['mention_right'](h_prev[n_passages:n_passages+n_passages*n_question_entities])
        output['pe_left'] = h_prev[n_pes:] # self.projection_layers_output['mention_left'](h_prev[n_passages+n_passages*n_question_entities:])
        output['pe_right'] = h_prev[n_pes:] # self.projection_layers_output['mention_right'](h_prev[n_passages+n_passages*n_question_entities:])
        return output 

class GNN_no_attention(nn.Module):
    def __init__(self, in_feats, hid_feats, dropout, n_layers, n_heads,
                activation_function = torch.nn.GELU()):
        super().__init__()

        # heterogeneous message passing layers 
        self.layers = nn.ModuleList([dgl.nn.GraphConv(in_feats, hid_feats)])
        for _ in range(n_layers - 2):
            self.layers.append(dgl.nn.GraphConv(hid_feats, hid_feats))
        self.layers.append(dgl.nn.GraphConv(hid_feats, in_feats))

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_function

    def forward(self, graph, h_prev, relation_memory, node_indices):
        graph = graph.add_self_loop()
        for index, layer in enumerate(self.layers):
            # message passing
            h_prev = layer(graph, h_prev)
            h_prev = self.dropout(h_prev)

        # final projection
        output = {}
        n_pes = len(node_indices['question_mention'])
        output['qe_left'] = h_prev[:n_pes] # self.projection_layers_output['mention_left'](h_prev[n_passages:n_passages+n_passages*n_question_entities])
        output['qe_right'] = h_prev[:n_pes] # self.projection_layers_output['mention_right'](h_prev[n_passages:n_passages+n_passages*n_question_entities])
        output['pe_left'] = h_prev[n_pes:] # self.projection_layers_output['mention_left'](h_prev[n_passages+n_passages*n_question_entities:])
        output['pe_right'] = h_prev[n_pes:] # self.projection_layers_output['mention_right'](h_prev[n_passages+n_passages*n_question_entities:])
        return output 