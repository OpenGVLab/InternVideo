# PREVALENT, 2020, weituo.hao@duke.edu
# Modified in Recurrent VLN-BERT, 2020, Yicong.Hong@anu.edu.au

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers import BertPreTrainedModel, BertConfig
import pdb

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = True

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores


class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Lang self-att and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)
        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        ''' Cross Attention -- cross for vision not for language '''
        visn_att_output, attention_scores = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return visn_att_output, attention_scores

    def self_att(self, visn_input, visn_attention_mask):
        ''' Self Attention -- on visual features with language clues '''
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return visn_att_output

    def output_fc(self, visn_input):
        ''' Feed forward '''
        visn_inter_output = self.visn_inter(visn_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask, tdx):

        ''' visual self-attention with state '''
        visn_att_output = torch.cat((lang_feats[:, 0:1, :], visn_feats), dim=1)
        state_vis_mask = torch.cat((lang_attention_mask[:,:,:,0:1], visn_attention_mask), dim=-1)

        ''' state and vision attend to language '''
        visn_att_output, cross_attention_scores = self.cross_att(lang_feats[:, 1:, :], lang_attention_mask[:, :, :, 1:], visn_att_output, state_vis_mask)

        language_attention_scores = cross_attention_scores[:, :, 0, :]

        state_visn_att_output = self.self_att(visn_att_output, state_vis_mask)
        state_visn_output = self.output_fc(state_visn_att_output[0])

        visn_att_output = state_visn_output[:, 1:, :]
        lang_att_output = torch.cat((state_visn_output[:, 0:1, :], lang_feats[:,1:,:]), dim=1)

        visual_attention_scores = state_visn_att_output[1][:, :, 0, 1:]

        return lang_att_output, visn_att_output, language_attention_scores, visual_attention_scores


class VisionEncoder(nn.Module):
    def __init__(self, vision_size, config):
        super().__init__()
        feat_dim = vision_size
        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats = visn_input
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        output = self.dropout(x)
        return output


class VLNBert(BertPreTrainedModel):
    def __init__(self, config):
        super(VLNBert, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim            # 2176
        logger.info('VLNBert Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type  # ''
        self.vl_layers = config.vl_layers                # 4
        self.la_layers = config.la_layers                # 9
        self.lalayer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.la_layers)])
        self.addlayer = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.vl_layers)])
        # self.vision_encoder = VisionEncoder(self.config.img_feature_dim, self.config)
        # self.apply(self.init_weights)
        self.init_weights()
        # del self.img_dim
        # del self.vision_encoder
        # del self.embeddings

    def forward(self, mode, input_ids, token_type_ids=None,
            attention_mask=None, lang_mask=None, vis_mask=None,
            position_ids=None, head_mask=None, img_feats=None):

        attention_mask = lang_mask

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            if input_ids.shape[-1] == 768: # rxr instruction
                text_embeds = input_ids
            else:                          # r2r instruction
                embedding_output = self.embeddings(input_ids,
                    position_ids=position_ids, token_type_ids=token_type_ids)
                text_embeds = embedding_output

            for layer_module in self.lalayer:
                temp_output = layer_module(text_embeds, extended_attention_mask)
                text_embeds = temp_output[0]

            sequence_output = text_embeds
            pooled_output = self.pooler(sequence_output)

            return pooled_output, sequence_output


        elif mode == 'visual':
            ''' LXMERT visual branch (no language processing during navigation) '''
            text_embeds = input_ids
            text_mask = extended_attention_mask

            # img_embedding_output = self.vision_encoder(img_feats)
            img_embedding_output = img_feats
            img_seq_len = img_feats.shape[1]
            batch_size = text_embeds.size(0)

            img_seq_mask = vis_mask

            extended_img_mask = img_seq_mask.unsqueeze(1).unsqueeze(2)
            extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_img_mask = (1.0 - extended_img_mask) * -10000.0
            img_mask = extended_img_mask

            lang_output = text_embeds
            visn_output = img_embedding_output

            for tdx, layer_module in enumerate(self.addlayer):
                lang_output, visn_output, language_attention_scores, visual_attention_scores = layer_module(lang_output, text_mask, visn_output, img_mask, tdx)

            sequence_output = lang_output
            pooled_output = self.pooler(sequence_output)

            visual_action_scores = visual_attention_scores.mean(dim=1)

            return pooled_output, visual_action_scores
