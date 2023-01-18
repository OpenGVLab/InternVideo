import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm


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
        self.output_attentions = config.output_attentions

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

        # recurrent vlnbert use attention scores
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
        if isinstance(config.hidden_act, str):
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


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, 
                                         None if head_mask is None else head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


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


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

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
        return context_layer, attention_probs

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.no_lang_ca = config.no_lang_ca # do not update language embeds

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
        # Cross Attention
        if self.no_lang_ca:
            lang_att_output = lang_input
        else:
            lang_att_output, _ = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output, cross_attention_score = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output, cross_attention_score

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        if self.no_lang_ca:
            lang_att_output = (lang_input, )
        else:
            lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        if not self.no_lang_ca:
            lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        if self.no_lang_ca:
            lang_output = lang_input
        else:
            lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output, cross_attention_score = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_attention_score = cross_attention_score[:,:,0,:]
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output[0], visn_att_output[0])

        return lang_output, visn_output, lang_attention_score

class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_l_layers = config.num_l_layers
        self.num_r_layers = config.num_r_layers
        self.num_h_layers = config.num_h_layers
        self.num_x_layers = config.num_x_layers
        self.update_lang_bert = config.update_lang_bert

        # Using self.layer instead of self.l_layers to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad_(False)

        self.h_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_h_layers)]
        ) if self.num_h_layers > 0 else None
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        ) if self.num_r_layers > 0 else None
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, extended_txt_masks, hist_embeds,
                extended_hist_masks, img_embeds=None, extended_img_masks=None):
        # text encoding
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        # if not self.update_lang_bert:
        #     txt_embeds = txt_embeds.detach()

        # image encoding
        if img_embeds is not None:
            if self.r_layers is not None:
                for layer_module in self.r_layers:
                    temp_output = layer_module(img_embeds, extended_img_masks)
                    img_embeds = temp_output[0]

        # history encoding
        if self.h_layers is not None:
            for layer_module in self.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]
        hist_max_len = hist_embeds.size(1)
        
        # cross-modal encoding
        if img_embeds is None:
            hist_img_embeds = hist_embeds
            extended_hist_img_masks = extended_hist_masks
        else:
            hist_img_embeds = torch.cat([hist_embeds, img_embeds], 1)
            extended_hist_img_masks = torch.cat([extended_hist_masks, extended_img_masks], -1)
        
        for layer_module in self.x_layers:
            txt_embeds, hist_img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                hist_img_embeds, extended_hist_img_masks)

        hist_embeds = hist_img_embeds[:, :hist_max_len]
        if img_embeds is not None:
            img_embeds = hist_img_embeds[:, hist_max_len:]
        return txt_embeds, hist_embeds, img_embeds



class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dep_linear = nn.Linear(config.depth_feat_size, config.hidden_size)
        self.dep_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dis_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.dis_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 0: non-navigable, 1: navigable, 2: stop
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, rgb_feat, dep_feat, ang_feat, dis_feat, type_embeddings, nav_types=None):
        transformed_im = self.img_layer_norm(self.img_linear(rgb_feat))
        transformed_dep = self.dep_layer_norm(self.dep_linear(dep_feat))
        transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
        transformed_dis = self.dis_layer_norm(self.dis_linear(dis_feat))
        embeddings = transformed_im + transformed_dep + transformed_ang + transformed_dis + type_embeddings
        if nav_types is not None:
            nav_embeddings = self.nav_type_embedding(nav_types)
            embeddings = embeddings + nav_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class HistoryEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # self.dep_linear = nn.Linear(config.depth_feat_size, config.hidden_size)
        # self.dep_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
        self.position_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        # special type embedding for history
        self.type_embedding = nn.Embedding(1, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hist_enc_pano = config.hist_enc_pano
        if config.hist_enc_pano:
            self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            # self.pano_dep_linear = nn.Linear(config.depth_feat_size, config.hidden_size)
            # self.pano_dep_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.pano_ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
            self.pano_ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            pano_enc_config = copy.copy(config)
            pano_enc_config.num_hidden_layers = config.num_h_pano_layers
            self.pano_encoder = BertEncoder(pano_enc_config)
        else:
            self.pano_encoder = None

    def forward(self, img_feats, dep_feats, ang_feats, pos_ids, 
                pano_img_feats=None, pano_dep_feats=None, pano_ang_feats=None):
        '''Args:
        - img_feats: (batch_size, dim_feat)
        - pos_ids: (batch_size, )
        - pano_img_feats: (batch_size, pano_len, dim_feat)
        '''
        device = next(iter(self.parameters())).device
        if img_feats is not None:
            batch_size = img_feats.size(0)
        else:
            batch_size = 1

        type_ids = torch.zeros((batch_size, )).long().to(device)
        type_embeddings = self.type_embedding(type_ids)

        if img_feats is None:
            cls_embeddings = self.dropout(self.layer_norm(
                self.cls_token.expand(batch_size, -1, -1)[:, 0] + type_embeddings))
            return cls_embeddings

        # history embedding per step
        embeddings = self.img_layer_norm(self.img_linear(img_feats)) + \
                     self.ang_layer_norm(self.ang_linear(ang_feats)) + \
                     self.position_embeddings(pos_ids) + \
                     type_embeddings

        if self.pano_encoder is not None:
            pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                              self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
            pano_embeddings = self.dropout(pano_embeddings)
            # TODO: mask is always True
            batch_size, pano_len, _ = pano_img_feats.size()
            extended_pano_masks = torch.zeros(batch_size, pano_len).float().to(device).unsqueeze(1).unsqueeze(2)
            pano_embeddings = self.pano_encoder(pano_embeddings, extended_pano_masks)[0]
            pano_embeddings = torch.mean(pano_embeddings, 1)

            embeddings = embeddings + pano_embeddings # history既包含了orientation方位的img，也包含了pano的context

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class NavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)
        
        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        self.init_weights()
        
        if self.config.fix_lang_embedding:
            for name, param in self.embeddings.named_parameters():
                if 'token_type_embeddings' not in name:
                    param.requires_grad_(False)
        if self.config.fix_hist_embedding:
            for parma in self.hist_embeddings.parameters():
                parma.requires_grad_(False)
        if self.config.fix_obs_embedding:
            for parma in self.img_embeddings.parameters():
                parma.requires_grad_(False)


    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_rgb_fts=None, hist_depth_fts=None, hist_ang_fts=None, 
                hist_pano_rgb_fts=None, hist_pano_depth_fts=None, hist_pano_ang_fts=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None,
                ob_rgb_fts=None, ob_dep_fts=None, ob_ang_fts=None, ob_dis_fts=None, ob_nav_types=None, 
                ob_masks=None):
        
        # text embedding            
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            txt_token_type_ids = torch.zeros_like(txt_ids)
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            for layer_module in self.encoder.layer:
                temp_output = layer_module(txt_embeds, extended_txt_masks)
                txt_embeds = temp_output[0]
            # if self.config.fix_lang_embedding:
            #     txt_embeds = txt_embeds.detach()
            if self.config.no_lang_ca: # run self-attn layers for lang
                all_txt_embeds = [txt_embeds]
                for layer_module in self.encoder.x_layers:
                    lang_att_output = layer_module.lang_self_att(txt_embeds, extended_txt_masks)[0]
                    lang_inter_output = layer_module.lang_inter(lang_att_output)
                    lang_output = layer_module.lang_output(lang_inter_output, lang_att_output)
                    all_txt_embeds.append(lang_output)
                return all_txt_embeds
            return txt_embeds

        # history embedding per step
        if mode == 'history':
            hist_embeds = self.hist_embeddings(hist_rgb_fts, hist_depth_fts, hist_ang_fts, ob_step_ids,
                pano_img_feats=hist_pano_rgb_fts, pano_dep_feats=hist_pano_depth_fts, pano_ang_feats=hist_pano_ang_fts)
            # if self.config.fix_hist_embedding:
            #     hist_embeds = hist_embeds.detach()
            return hist_embeds
            
        # cross-modal encoding per step
        elif mode == 'navigation':
            ''' LXMERT visual branch'''
            # history embedding
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

            # if self.encoder.h_layers is not None:
            #     for layer_module in self.encoder.h_layers:
            #         temp_output = layer_module(hist_embeds, extended_hist_masks)
            #         hist_embeds = temp_output[0]

            # image embedding
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0

            ob_token_type_ids = torch.ones(ob_rgb_fts.size(0), ob_rgb_fts.size(1), dtype=torch.long, device=self.device)
            ob_embeds = self.img_embeddings(ob_rgb_fts, ob_dep_fts, ob_ang_fts, ob_dis_fts,
                self.embeddings.token_type_embeddings(ob_token_type_ids), 
                nav_types=ob_nav_types)
            # if self.encoder.r_layers is not None:
            #     for layer_module in self.encoder.r_layers:
            #         temp_output = layer_module(ob_embeds, extended_ob_masks)
            #         ob_embeds = temp_output[0]
            # if self.config.fix_obs_embedding:
            #     ob_embeds = ob_embeds.detach()

            # multi-modal encoding
            hist_max_len = hist_embeds.size(1)
            hist_ob_embeds = torch.cat([hist_embeds, ob_embeds], 1)
            extended_hist_ob_masks = torch.cat([extended_hist_masks, extended_ob_masks], -1)

            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0
            
            if self.config.no_lang_ca:
                all_txt_embeds = txt_embeds
            for l, layer_module in enumerate(self.encoder.x_layers):
                if self.config.no_lang_ca:
                    txt_embeds = all_txt_embeds[l]
                txt_embeds, hist_ob_embeds, lang_attention_score = layer_module(
                    txt_embeds, extended_txt_masks, 
                    hist_ob_embeds, extended_hist_ob_masks,
                )

            hist_embeds = hist_ob_embeds[:, :hist_max_len]
            ob_embeds = hist_ob_embeds[:, hist_max_len:]

            # TODO
            if self.config.no_lang_ca:
                act_logits = self.next_action(ob_embeds).squeeze(-1)
            else:
                if self.config.act_pred_token == 'ob_txt':
                    act_logits = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob':
                    act_logits = self.next_action(ob_embeds).squeeze(-1)
                elif self.config.act_pred_token == 'ob_hist':
                    act_logits = self.next_action(ob_embeds * hist_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob_txt_hist':
                    act_logits = self.next_action(ob_embeds * (txt_embeds[:, :1] + hist_embeds[:, :1])).squeeze(-1)
            act_logits.masked_fill_(ob_nav_types==0, -float('inf'))
            # ob_nav_type的形状: 1,1,1,1,2,0,0,0,0,0

            return act_logits, txt_embeds, hist_embeds, ob_embeds, lang_attention_score

