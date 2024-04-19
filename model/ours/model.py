import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead
from model.vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()
from model.transformers.src.transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from model.vcd_utils.vcd_add_noise import add_diffusion_noise

from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
class MILU(nn.Module):
    def __init__(self, lm_path, input_dim, freeze_word=False, max_v_len=256):
        super().__init__()

        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        self.lm = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', local_files_only=False)

        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.lm_proj = nn.Linear(input_dim, lm_dim)
        self.v_emb = nn.Parameter(torch.randn((1, 1, lm_dim)))
        if freeze_word:
            for name, param in self.lm.named_parameters():
                if 'shared' in name:
                    param.requires_grad = False

        self.nlq_head = NLQHead(in_dim=lm_dim, max_v_len=max_v_len)

        self.ln_pre_t = nn.LayerNorm(lm_dim)
        self.ln_pre_i = nn.LayerNorm(lm_dim)
        self.ln_post = nn.LayerNorm(lm_dim)
        self.cross_attn = nn.MultiheadAttention(lm_dim, lm_dim // 64, batch_first=True)
        self.mlm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(lm_dim, lm_dim)),
                        ('gelu', nn.GELU()),
                        ('ln', nn.LayerNorm(lm_dim)),
                        ('fc', nn.Linear(lm_dim, 32128))]))
        self.cross_modal_transformer = nn.Transformer(lm_dim, nhead=lm_dim // 64,
                                   num_encoder_layers=6,
                                   num_decoder_layers=6,
                                   dim_feedforward=2048)
    def forward(self, v_feat, v_mask, q_token, q_mask, gt_segments, gt_labels,
                labels=None, **remains):
        # remains: dict() ['video_id', 'q_text', 'v_len', 'query_id', 'sample_ratio', 'a_text', 'task', 'descriptions_token', mask_idx, mask_num, masked_description, interactive_token]
        # encoder
        # todo: random mask descriptions_token and apply mlm
        if remains['answer_list'] != None:
            answer_list = [i.reshape(-1) for i in remains['answer_list']] # 8 * tensor[4 * len]
            q_token_with_answer = pad_sequence([torch.cat([i, j], dim=-1) for i,j in zip(q_token, answer_list)], batch_first=True)

            q_mask_with_answer = torch.ones_like(q_token_with_answer, device=q_token.device).bool()
            encoder_out_, mask_ = self.forward_encoder(v_feat, v_mask, q_token_with_answer, q_mask_with_answer) 
            encoder_out_v_ = encoder_out_[:, -v_feat.shape[1]:]
            nlq_results_ = self.nlq_head(
                feat=encoder_out_v_.permute(0, 2, 1),  # (B, D, T)
                mask=v_mask.unsqueeze(1),  # (B, 1, T)
                gt_segments=gt_segments,
                gt_labels=gt_labels
            )


        mlm_loss = None
        if remains['descriptions_token'] != None:
            masked_description = remains['masked_description']
            masked_feats = self.lm.encoder.embed_tokens(masked_description)
            mlm_feats = self.cross_former(masked_feats, v_feat, v_feat)
            mlm_feats = self.mlm_head(mlm_feats)
            scores = mlm_feats.float().reshape(-1, 32128)
            mlm_obj = nn.CrossEntropyLoss()
            mlm_loss = mlm_obj(scores, remains['descriptions_token'].reshape(-1))

        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask) 
        # localizer
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            gt_segments=gt_segments,
            gt_labels=gt_labels
        )
        time_loss = nlq_results['final_loss'] * 1.0 #+ nlq_results_['final_loss']
        if remains['answer_list'] != None:
            time_loss += nlq_results_['final_loss']
        # decoder
        outputs = self.lm(
            encoder_outputs=(encoder_out,),
            attention_mask=mask,
            labels=labels,
        )

        # todo: add other losses
        lm_loss = outputs.loss

        total_loss = 0.5 * time_loss + 0.5 * lm_loss
        if mlm_loss != None:
            total_loss += mlm_loss
        return total_loss, lm_loss, time_loss

    def generate(self, v_feat, v_mask, q_token, q_mask, v_len, **remains):
        # v_feat = add_diffusion_noise(v_feat, 300)
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask)
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]


        v_feat_cd = add_diffusion_noise(v_feat, 500)
        encoder_out_cd, mask_cd = self.forward_encoder(v_feat_cd, v_mask, q_token, q_mask)
        print(q_token)
        encoder_out_v_cd = encoder_out_cd[:, -v_feat_cd.shape[1]:]
        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            training=False,
            v_lens=v_len,
            feat_cd=encoder_out_v_cd.permute(0, 2, 1)  # (B, D, T)
        )

        answer_tokens = self.lm.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
            attention_mask=mask,
            images_cd = BaseModelOutput(last_hidden_state=encoder_out_cd),
            cd_beta  = 1,
            cd_alpha  = 0.1,
            max_new_tokens=32
        )
        return nlq_results, answer_tokens

    def forward_encoder(self, v_feat, v_mask, q_token, q_mask):
        B, L, D = v_feat.shape
        v_feat = self.lm_proj(v_feat)
        v_feat = v_feat + self.v_emb.expand((B, L, -1))
        q_feat = self.lm.encoder.embed_tokens(q_token)
        lm_input = torch.cat([q_feat, v_feat], dim=1)
        lm_mask = torch.cat([q_mask, v_mask], dim=1)
        out = self.lm.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask
        )
        return out.last_hidden_state, lm_mask

    def cross_former(self, q, k, v):
        k = k.reshape([q.shape[0], -1, q.shape[-1]])
        v = v.reshape([q.shape[0], -1, q.shape[-1]])
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x, x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x