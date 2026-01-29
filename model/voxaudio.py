import os
import json
import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

from model.mm.ext.rotary_embeddings import compute_rope_rotations
from model.mm.embeddings import TimestepEmbedder
from model.mm.low_level import MLP, ChannelLastConv1d, ConvMLP, CausalChannelLastConv1d, CausalConvMLP
from model.mm.transformer_layers_new import CausalFinalBlock, CausalMMDitSingleBlock, CausalTextConditionedBlock, LinearFinalBlock
from model.mm.utils.features_utils_audio import FeaturesUtils
import time

@dataclass
class PreprocessedConditions:
    text_f: torch.Tensor
    text_f_c: torch.Tensor
    text_lens: torch.Tensor


class VoxAudio(nn.Module):
    def __init__(self,
                 *,
                 args,
                 latent_dim: int,
                 text_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 text_seq_len: int = 77,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 v2: bool = False,
                 chunk_size=None) -> None:
        super().__init__()

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.joint_depth = depth - fused_depth
        self.chunk_size = chunk_size

        # === Input Projections ===
        if v2:
            self.audio_input_proj = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim, bias=True),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )
            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim, bias=True),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )
        else:
            self.audio_input_proj = nn.Sequential(
                CausalChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7),
                nn.SELU(),
                CausalConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7),
            )
            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim, bias=True),
                MLP(hidden_dim, hidden_dim * 4),
            )
        
        # Conditional projections
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)

        # Final layer
        if v2:
            self.final_layer = LinearFinalBlock(hidden_dim, latent_dim)
        else:
            self.final_layer = CausalFinalBlock(hidden_dim, latent_dim)

        # Timestep embedding
        self.t_embed = TimestepEmbedder(hidden_dim,
                                        frequency_embedding_size=hidden_dim,
                                        max_period=1)
        
        if v2:
            kernel_size = 1
        else:
            kernel_size = 3
        # Joint Blocks
        self.joint_blocks = nn.ModuleList([
            CausalTextConditionedBlock(hidden_dim,
                       num_heads,
                       mlp_ratio=mlp_ratio,
                       pre_only=(i == depth - fused_depth - 1),
                       kernel_size=kernel_size)
            for i in range(depth - fused_depth)
        ])

        # Fused Blocks
        self.fused_blocks = nn.ModuleList([
            CausalMMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
            for _ in range(fused_depth)
        ])

        self.eoc_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)

        if getattr(args, 'use_repa', False):
            zs_dim = getattr(args, 'zs_dim', 768)
            self.zs_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, zs_dim)
            )

        # Rotations (RoPE)
        self.initialize_rotations()

        # Weight init
        self.apply(self._basic_init)
        self._zero_init_layers()

        mode = '16k'
        audiovae_ckpt = getattr(args, 'audiovae_ckpt_path', None)
        self.feature_extractor = FeaturesUtils(audiovae_ckpt=audiovae_ckpt, max_text_length=self._text_seq_len, pad_to_max_length=getattr(args, 'pad_to_max_length', True), text_encoder_for_gen=getattr(args, 'text_encoder_for_gen', True), use_clap=False, use_qwen3_audio_encoder=False,
        use_peav=False, use_ae_predictor=False, use_whisper=False).cuda().eval()
        self._freeze_feature_extractor()
    def _freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        
        # self.latent_rot = latent_rot.to(device=self.device)
        self.register_buffer('latent_rot', latent_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self.initialize_rotations()

    def _basic_init(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _zero_init_layers(self):
        # Zero-out adaLN modulation layers
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def set_input_tensor(self, input_tensor):
        pass

    def preprocess_conditions(self, text_f: torch.Tensor, text_lens: torch.Tensor) -> PreprocessedConditions:
        """
        Args:
            text_f: [B, L, D_text] 原始文本特征
            text_lens: [B] 每个 sample 的有效长度
        """
        bs, seq_len, _ = text_f.shape
        text_f = self.text_input_proj(text_f) # [B, L, hidden_dim]
        
        device = text_f.device
        mask = torch.arange(seq_len, device=device).expand(bs, seq_len) < text_lens.unsqueeze(1)
        mask_float = mask.unsqueeze(-1).to(text_f.dtype)
        sum_feat = torch.sum(text_f * mask_float, dim=1) # [B, hidden_dim]
        clamped_lens = text_lens.clamp(min=1).unsqueeze(-1).to(text_f.dtype)
        masked_mean_feat = sum_feat / clamped_lens # [B, hidden_dim]
        text_f_c = self.text_cond_proj(masked_mean_feat) # [B, hidden_dim]

        return PreprocessedConditions(
            text_f=text_f, 
            text_f_c=text_f_c,
            text_lens=text_lens
        )


    def predict_eoc(self, latent_h: torch.Tensor, id2chunk: torch.Tensor) -> torch.Tensor:

        B, L, H = latent_h.shape
        device = latent_h.device
        dtype = latent_h.dtype
        id2chunk = id2chunk - id2chunk.min().item() + 1
        
        max_chunk_in_batch = id2chunk.max().item()
        attn = torch.ones(B, L, dtype=dtype, device=device)

        denom = id2chunk.new_zeros(B, max_chunk_in_batch + 1, dtype=dtype).scatter_add_(
            dim=1, index=id2chunk, src=attn
        )[:,1:]
        frame2chunk = id2chunk.unsqueeze(-1).repeat(1, 1, H)  # [B, L] -> [B, T,H], with padding included
        chunk_aggregate = frame2chunk.new_zeros(B, max_chunk_in_batch + 1, H, dtype=dtype).scatter_add_(
            dim=1, index=frame2chunk, src=latent_h
        )[:,1:]
        chunk_aggregate = chunk_aggregate / (denom.unsqueeze(-1) + 1e-5)
        eco_logits = self.eoc_prediction_head(chunk_aggregate).squeeze(-1)
        return eco_logits

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor, conditions: PreprocessedConditions, text_lens, id2chunk, return_zs=False) -> torch.Tensor:
        text_f = conditions.text_f
        text_f_c = conditions.text_f_c
        latent = self.audio_input_proj(latent)
        global_c = self.global_cond_mlp(text_f_c).unsqueeze(1)
        t_emb = self.t_embed(t)
        extended_c = t_emb + global_c
        block_mask = None
        zs = None

        for block_id, block in enumerate(self.joint_blocks):
            latent, text_f, block_mask = block(latent, text_f, global_c, extended_c, self.latent_rot[:, :latent.shape[1]], text_lens=text_lens, block_mask=block_mask, id2chunk=id2chunk)

            if return_zs and block_id == 3:
                zs = self.zs_proj(latent)
                
        block_mask = None
        for block in self.fused_blocks:
            latent, block_mask = block(latent, extended_c, self.latent_rot[:, :latent.shape[1]], block_mask=block_mask, id2chunk=id2chunk)

        eco_logits = self.predict_eoc(latent, id2chunk)
        flow = self.final_layer(latent, global_c)
        if return_zs:
            return flow, eco_logits, zs
        else:
            return flow, eco_logits

    def predict_flow_inference(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions, text_lens, id2chunk, kv_cache, current_start=0, current_end=0) -> torch.Tensor:
        text_f = conditions.text_f
        text_f_c = conditions.text_f_c
        

        latent = self.audio_input_proj(latent)
        global_c = self.global_cond_mlp(text_f_c).unsqueeze(1)
        t_emb = self.t_embed(t)
        extended_c = t_emb + global_c
        block_mask = None
        
        # t1_time = time.time()
        for idx, block in enumerate(self.joint_blocks):
            latent, text_f, block_mask = block(latent, text_f, global_c, extended_c, self.latent_rot[:, current_start:current_end], kv_cache[idx], current_start=current_start, current_end=current_end, text_lens=text_lens, block_mask=block_mask, id2chunk=id2chunk)
        # t2_time = time.time()

        block_mask = None
        for idx, block in enumerate(self.fused_blocks):
            latent, block_mask = block(latent, extended_c, self.latent_rot[:, current_start:current_end], kv_cache[idx+self.joint_depth], current_start=current_start, current_end=current_end, block_mask=block_mask, id2chunk=id2chunk)
        
        eco_logits = self.predict_eoc(latent, id2chunk[:, current_start:current_end])
        flow = self.final_layer(latent, global_c)
        
        return flow, eco_logits

    def forward(self, latent: torch.Tensor, text_f: torch.Tensor, t: torch.Tensor, text_lens, id2chunk, kv_cache=None, current_start=0, current_end=0, return_zs=False) -> torch.Tensor:
        conditions = self.preprocess_conditions(text_f, text_lens)
        if kv_cache is None:
            if return_zs:
                flow, eco_logits, zs = self.predict_flow(latent, t, conditions, text_lens=text_lens, id2chunk=id2chunk, return_zs=return_zs)
                return flow, eco_logits, zs
            else:
                flow, eco_logits = self.predict_flow(latent, t, conditions, text_lens=text_lens, id2chunk=id2chunk, return_zs=return_zs)
                return flow, eco_logits
        else:
            flow, eco_logits = self.predict_flow_inference(latent, t, conditions, kv_cache, text_lens=text_lens, id2chunk=id2chunk, current_start=current_start, current_end=current_end)
            return flow, eco_logits

    @property
    def device(self):
        return next(self.parameters()).device

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_conditions(self, bs, text_lens) -> PreprocessedConditions:
        empty_text = self.get_empty_string_sequence(1)[:bs, :text_lens.max()]
        conditions = self.preprocess_conditions(empty_text, text_lens)
        
        conditions.text_f = conditions.text_f.expand(bs, -1, -1)
        conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions
    
    
    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float, text_lens, id2chunk) -> torch.Tensor:
        t = t * torch.ones(len(latent), 1, device=latent.device, dtype=latent.dtype)

        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions, text_lens, id2chunk)
        else:
            flow, eco_logits = self.predict_flow(latent, t, conditions, text_lens, id2chunk)
            flow_w, eco_logits_w = self.predict_flow(latent, t, empty_conditions, text_lens, id2chunk)
            return cfg_strength * flow + (1 - cfg_strength) * flow_w, eco_logits
    
    def ode_wrapper_cache_cfg(self, t: torch.Tensor, latent: torch.Tensor, st, ed, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength, text_lens, id2chunk, kv_cache_con, kv_cache_un) -> torch.Tensor:
        t = t * torch.ones(len(latent), 1, device=latent.device, dtype=latent.dtype)
        if cfg_strength < 1.0:
            return self.predict_flow_inference(latent, t, conditions, text_lens, id2chunk, kv_cache_con, st, ed)
        else:
            flow, eco_logits = self.predict_flow_inference(latent, t, conditions, text_lens, id2chunk, kv_cache_con, st, ed)
            flow_w, eco_logits_w = self.predict_flow_inference(latent, t, empty_conditions, text_lens, id2chunk, kv_cache_un, st, ed)
            return cfg_strength * eco_logits+ (1 - cfg_strength) * eco_logits_w, eco_logits

    def ode_wrapper_cache(self, t, latent, st, ed, conditions, text_lens, id2chunk, kv_cache_con):
        t = t * torch.ones(len(latent), 1, device=latent.device, dtype=latent.dtype)
        return self.predict_flow_inference(latent, t, conditions, text_lens, id2chunk, kv_cache_con, st, ed)

    def ode_wrapper_flush_cache(self, t, latent, st, ed, conditions, text_lens, id2chunk, kv_cache_con):
        return self.predict_flow_inference(latent, t, conditions, text_lens, id2chunk, kv_cache_con, st, ed)

    def ode_wrapper_test(self, t, latent, conditions, text_lens, id2chunk):
        return self.predict_flow(latent, t, conditions, text_lens, id2chunk)

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len