from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch
import torch as th
import torch.nn as nn
from typing import Optional

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

class TransformerNetModel(nn.Module):
    """
    The ScanDL transformer.
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        num_transformer_layers,
        num_transformer_heads,
        one_noise_step,
        mask_padding,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
            config.num_hidden_layers = num_transformer_layers
            config.num_attention_heads = num_transformer_heads
            config.hidden_size = input_dims

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode

        self.mask_padding = mask_padding
        self.one_noise_step = one_noise_step

        self.bert_for_embedding = BertModel.from_pretrained(config_name)
        # freeze BERT parameters (so that embeddings are freezed)
        for param in self.bert_for_embedding.parameters():
            param.requires_grad = False

        self.sn_sp_repr_embedding = nn.Embedding(self.hidden_t_dim, self.input_dims)

        self.positional_encoding = nn.Embedding(self.hidden_t_dim, self.input_dims)
        self.sn_input_ids_embedding = self.bert_for_embedding.embeddings.word_embeddings
        # additional linear layer needed if hidden is not 768 to map from pretrained BERT embeddings to other dim
        if self.input_dims != 768:
            self.proj_bert_emb = nn.Linear(768, self.input_dims)

        self.lm_head = nn.Linear(self.input_dims, self.hidden_t_dim)
        with torch.no_grad():
            self.lm_head.weight = self.sn_sp_repr_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        self.input_transformers = BertEncoder(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def get_embeds(
            self,
            sn_sp_repr,
            sn_input_ids,
            indices_pos_enc,
    ):

        sn_sp_emb = self.sn_sp_repr_embedding(sn_sp_repr)
        pos_enc = self.positional_encoding(indices_pos_enc)
        if self.input_dims == 768:
            sn_input_ids_emb = self.sn_input_ids_embedding(sn_input_ids)
        else:
            sn_input_ids_emb_bert_embs = self.sn_input_ids_embedding(sn_input_ids)
            sn_input_ids_emb = self.proj_bert_emb(sn_input_ids_emb_bert_embs)
        return sn_sp_emb, pos_enc, sn_input_ids_emb


    def get_logits(self, model_output):
        if self.logits_mode == 1:
            return self.lm_head(model_output)
        elif self.logits_mode == 2:  # standard cosine similarity
            raise NotImplementedError('standard cosine similarity not yet implemented for sp model output.')
        else:
            raise NotImplementedError


    def forward(
            self,
            x,  # x_t
            ts,
            sn_input_ids_emb,
            pos_enc,
            attention_mask: Optional[torch.tensor] = None,
            atten_vis: Optional[bool] = False,
    ):
        """
        Apply the model to an input batch.

        :param x: the noised input ID embeddings
        :param ts: a 1-D batch of timesteps.
        :param sn_input_ids_emb: the BERT embeddings
        :param pos_enc: the positional embeddings
        :param attention_mask: the attention mask (only given during training, not during inference)
        :atten_vis: visualise attention
        """
        # timestep embedding
        emb_t = self.time_embed(timestep_embedding(ts, self.hidden_t_dim))

        # add the input x_t, the positional encoding pos_enc, the word ID embedding word_id_emb, and the timestep emb
        emb_inputs = x + pos_enc + sn_input_ids_emb + emb_t.unsqueeze(1).expand(-1, x.size(1), -1)

        # pipe through dropout and layer normalisation
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        if self.mask_padding:
            if attention_mask == None:
                raise ValueError('padding should be masked, but no attention mask given.')

            extended_attention_mask = attention_mask[:, None, None, :]

            if atten_vis:
                model_out = self.input_transformers(
                    emb_inputs, attention_mask=extended_attention_mask, output_attentions=True
                )
                input_trans_hidden_states = model_out.last_hidden_state
                attention_scores = model_out.attentions

            else:
                input_trans_hidden_states = self.input_transformers(
                    emb_inputs, attention_mask=extended_attention_mask
                ).last_hidden_state

        else:
            if atten_vis:
                model_out = self.input_transformers(emb_inputs, output_attentions=True)
                input_trans_hidden_states = model_out.last_hidden_state
                attention_scores = model_out.attentions
            else:
                input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state


        h = input_trans_hidden_states
        h = h.type(x.dtype)
        if atten_vis:
            return h, attention_scores
        else:
            return h

