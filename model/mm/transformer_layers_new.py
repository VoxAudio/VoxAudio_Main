from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .ext.rotary_embeddings import apply_rope
from .low_level import MLP, ChannelLastConv1d, ConvMLP, CausalChannelLastConv1d, CausalConvMLP
 
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算 RMS: x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 保持原始 dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f'dim={self.weight.shape[0]}, eps={self.eps}'

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask = None):
    """
    q, k, v: [B, H, N, D]
    block_mask: BlockMask or None
    """
    if block_mask is not None:
        # flex_attention 要求 shape [B, H, N, D]
        out = flex_attention(
            query=q,
            key=k,
            value=v,
            block_mask=block_mask
        )  # [B, H, N, D]
    else:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = F.scaled_dot_product_attention(q, k, v)
    
    return rearrange(out, 'b h n d -> b n (h d)').contiguous()

class SelfAttention(nn.Module):

    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # self.q_norm = nn.RMSNorm(dim // nheads)
        # self.k_norm = nn.RMSNorm(dim // nheads)
        self.q_norm = RMSNorm(dim // nheads)
        self.k_norm = RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nheads,
                                          d=dim // nheads,
                                          j=3)
    def pre_attention(
            self, x: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if rot is not None:
            # print(q.shape, rot.shape)
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x)
        out = attention(q, k, v)
        return out

def create_chunk_block_mask(B, H, Q_LEN, KV_LEN, id2chunk, device, q_start=0):
    """
    Creates a safe BlockMask for attention, handling out-of-bounds indices
    by mapping them to an 'overflow' chunk ID.
    """
    # Ensure id2chunk has a batch dimension: (B, L)
    if id2chunk.dim() == 1:
        id2chunk = id2chunk.unsqueeze(0).expand(B, -1)
    elif id2chunk.shape[0] == 1:
        id2chunk = id2chunk.expand(B, -1)

    B_id, L_orig = id2chunk.shape
    assert B_id == B, f"Batch size mismatch in id2chunk: got {B_id}, expected {B}"
    
    def mask_fn(b, h, q_idx, kv_idx):
        q_abs = q_idx + q_start
        q_id = id2chunk[b, torch.clamp(q_abs, 0, L_orig - 1)]
        kv_id = id2chunk[b, torch.clamp(kv_idx, 0, L_orig - 1)]
        
        return (q_id >= kv_id) & (q_idx < Q_LEN) & (kv_idx < KV_LEN)

    return create_block_mask(
        mask_fn,
        B=B, H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        _compile=False,
        device=device
    )

def create_mask_with_padding_exclusion(B, H, Q_LEN, KV_LEN, joint_chunk_ids, device, q_start=0):
    """
    Creates a BlockMask for attention, enforcing three rules:
    1. Causality/Chunk: Q[i] can only attend to K[j] if Q.chunk_id >= K.chunk_id.
    2. Padding Exclusion: Q[i] cannot attend to K[j] if K.chunk_id == 0 (Text Padding).
    3. Index Clamping: Safely handles out-of-bounds indices from FlexAttention internal mechanism.
    
    Args:
        joint_chunk_ids: [B, L_orig], combined chunk IDs for Text + Latent. ID=0 must be Padding.
    """
    # 确保 id2chunk 有 (B, L) 形状
    if joint_chunk_ids.dim() == 1:
        id2chunk = joint_chunk_ids.unsqueeze(0).expand(B, -1)
    elif joint_chunk_ids.shape[0] == 1:
        id2chunk = joint_chunk_ids.expand(B, -1)
    else:
        id2chunk = joint_chunk_ids

    B_id, L_orig = id2chunk.shape
    assert B_id == B, f"Batch size mismatch in id2chunk: got {B_id}, expected {B}"
    
    # 确保 id2chunk 是 long 类型
    if id2chunk.dtype != torch.long:
         id2chunk = id2chunk.long()

    def mask_fn(b, h, q_idx, kv_idx):
        q_abs = q_idx + q_start
        q_id = id2chunk[b, torch.clamp(q_abs, 0, L_orig - 1)]
        kv_id = id2chunk[b, torch.clamp(kv_idx, 0, L_orig - 1)]
        return (q_id >= kv_id) & (kv_id > 0) & (q_idx < Q_LEN) & (kv_idx < KV_LEN)

    return create_block_mask(
        mask_fn,
        B=B, H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        _compile=False,
        device=device
    )
    
class CausalMMDitSingleBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                 pre_only: bool = False,
                 kernel_size: int = 7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)
        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = CausalChannelLastConv1d(dim, dim, kernel_size=kernel_size)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = CausalConvMLP(dim, int(dim * mlp_ratio), kernel_size=kernel_size)

            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]):
        # x: BS * N * D
        # cond: BS * D
        modulation = self.adaLN_modulation(c)
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor]):
        if self.pre_only:
            return x

        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp

        return x
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rot: Optional[torch.Tensor], kv_cache=None, current_start=0, current_end=0, id2chunk=None, block_mask=None) -> torch.Tensor:
        if kv_cache is None:
            return self.forward_train(x, cond, rot, id2chunk=id2chunk, block_mask=block_mask)
        else:
            return self.forward_inference(x, cond, rot, kv_cache, current_start=current_start, current_end=current_end, id2chunk=id2chunk, block_mask=block_mask)
        
        return x

    def forward_train(self, x: torch.Tensor, cond: torch.Tensor,
                    rot: Optional[torch.Tensor], id2chunk: Optional[list] = None, block_mask = None) -> torch.Tensor:
        B, L_latent, D = x.shape
        device = x.device

        x_qkv, x_conditions = self.pre_attention(x, cond, rot)

        H = x_qkv[0].shape[1]
        if block_mask == None:
            block_mask = create_chunk_block_mask(
                B=B, H=H, Q_LEN=L_latent, KV_LEN=L_latent,
                id2chunk=id2chunk, device=device, q_start=0
            )
        
        attn_out = attention(*x_qkv, block_mask=block_mask)
        x = self.post_attention(x, attn_out, x_conditions)

        return x, block_mask

    def forward_inference(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor,
        rot: Optional[torch.Tensor], 
        kv_cache,
        current_start = 0,
        current_end = 0,
        block_mask=None,
        id2chunk=None) -> torch.Tensor:

        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        H = x_qkv[0].shape[1]

        B, _, _ = x.shape
        device = x.device
        if block_mask is None:
            block_mask = create_chunk_block_mask(
                B=B, H=H, Q_LEN=current_end-current_start, KV_LEN=current_end,
                id2chunk=id2chunk, device=device, q_start=current_start
            )
        
        kv_cache["k"][:, :, current_start:current_end] = x_qkv[1]
        kv_cache["v"][:, :, current_start:current_end] = x_qkv[2]
        attn_out = attention(x_qkv[0], kv_cache["k"][:, :, :current_end], kv_cache["v"][:, :, :current_end], block_mask=block_mask)
        x = self.post_attention(x, attn_out, x_conditions)
        
        return x, block_mask

class CausalTextConditionedBlock(nn.Module):
    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, pre_only: bool = False, kernel_size: int = 3):
        super().__init__()
        self.pre_only = pre_only

        self.latent_block = CausalMMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=False, kernel_size=kernel_size)
        self.text_block = CausalMMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)

    def forward(
        self,
        latent: torch.Tensor,
        text_f: torch.Tensor,
        global_c: torch.Tensor,
        extended_c: torch.Tensor,
        latent_rot: torch.Tensor,
        kv_cache=None,
        current_start=0,
        current_end=0,
        text_lens: Optional[torch.Tensor] = None,
        block_mask = None,
        id2chunk=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_cache is None:
            return self.forward_train(latent, text_f, global_c, extended_c, latent_rot, text_lens=text_lens, block_mask=block_mask, id2chunk=id2chunk)
        else:
            return self.forward_inference(latent, text_f, global_c, extended_c, latent_rot, kv_cache, current_start, current_end, text_lens=text_lens, block_mask=block_mask, id2chunk=id2chunk)
    def forward_train(
        self,
        latent: torch.Tensor,
        text_f: torch.Tensor,
        global_c: torch.Tensor,
        extended_c: torch.Tensor,
        latent_rot: torch.Tensor,
        text_lens: Optional[torch.Tensor] = None,
        block_mask = None,
        id2chunk=None) -> tuple[torch.Tensor, torch.Tensor]:
        
        B, L_latent, D = latent.shape
        L_text_padded = text_f.shape[1] 
        device = latent.device

        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

        H = x_qkv[0].shape[1]
        
        joint_qkv = [
            torch.cat([t_qkv[i], x_qkv[i]], dim=2)
            for i in range(3)
        ]
        total_len = L_text_padded + L_latent
        text_mask_bool = torch.arange(L_text_padded, device=text_f.device).expand(B, L_text_padded) < text_lens.unsqueeze(1)
        
        if block_mask is None:
            text_indices = torch.arange(L_text_padded, device=device).unsqueeze(0).expand(B, -1)
            text_mask = text_indices < text_lens.unsqueeze(-1)
            text_chunk_ids = torch.where(text_mask, torch.ones_like(text_indices), torch.zeros_like(text_indices))
            
            latent_chunk_ids = id2chunk + 1 # [B, L_latent]
            joint_chunk_ids = torch.cat([text_chunk_ids, latent_chunk_ids], dim=1) # [B, total_len]

            block_mask = create_mask_with_padding_exclusion(
                B=B, H=H, Q_LEN=total_len, KV_LEN=total_len,
                joint_chunk_ids=joint_chunk_ids, device=device, q_start=0
            )

        attn_out = attention(*joint_qkv, block_mask=block_mask)

        # 分离输出
        t_attn_out = attn_out[:, :L_text_padded]
        x_attn_out = attn_out[:, L_text_padded:]

        # 后处理
        latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
        if not self.pre_only:
            t_attn_out_masked = t_attn_out * text_mask_bool.unsqueeze(-1).to(t_attn_out.dtype)
            text_f_masked = text_f * text_mask_bool.unsqueeze(-1).to(text_f.dtype)
            text_f = self.text_block.post_attention(text_f_masked, t_attn_out_masked, t_mod)
            text_f = text_f * text_mask_bool.unsqueeze(-1).to(text_f.dtype)

        return latent, text_f, block_mask

    def forward_inference(
        self,
        latent: torch.Tensor,
        text_f: torch.Tensor,
        global_c: torch.Tensor,
        extended_c: torch.Tensor,
        latent_rot: torch.Tensor,
        kv_cache,
        current_start = 0,
        current_end = 0,
        text_lens: Optional[torch.Tensor] = None,
        block_mask = None,
        id2chunk=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        B, L_latent, D = latent.shape
        L_text_padded = text_f.shape[1]
        device = latent.device
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
        H = x_qkv[0].shape[1]
        text_mask_bool = torch.arange(L_text_padded, device=text_f.device).expand(B, L_text_padded) < text_lens.unsqueeze(1)

        if not kv_cache["is_init"]:
            assert current_start == 0, 'current_start must be 0 when kv_cache is not initialized'
            t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

            joint_qkv = [
                torch.cat([t_qkv[i], x_qkv[i]], dim=2)
                for i in range(3)
            ]
            kv_cache["k"][:, :, :L_text_padded + L_latent] = joint_qkv[1]
            kv_cache["v"][:, :, :L_text_padded + L_latent] = joint_qkv[2]
            kv_cache["is_init"] = True
            if block_mask is None:
                text_indices = torch.arange(L_text_padded, device=device).unsqueeze(0).expand(B, -1)
                text_mask = text_indices < text_lens.unsqueeze(-1)
                text_chunk_ids = torch.where(text_mask, torch.ones_like(text_indices), torch.zeros_like(text_indices))

                latent_chunk_ids = id2chunk + 1 # [B, L_latent]
                joint_chunk_ids = torch.cat([text_chunk_ids, latent_chunk_ids], dim=1)            
                
                block_mask = create_mask_with_padding_exclusion(
                    B=B, H=H, Q_LEN=L_text_padded + L_latent, KV_LEN=L_text_padded + L_latent,
                    joint_chunk_ids=joint_chunk_ids, device=device, q_start=0
                )

            attn_out = attention(*joint_qkv, block_mask=block_mask)
            t_attn_out = attn_out[:, :L_text_padded]
            x_attn_out = attn_out[:, L_text_padded:]

            latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
            if not self.pre_only:
                t_attn_out_masked = t_attn_out * text_mask_bool.unsqueeze(-1).to(t_attn_out.dtype)
                text_f_masked = text_f * text_mask_bool.unsqueeze(-1).to(text_f.dtype)
                text_f = self.text_block.post_attention(text_f_masked, t_attn_out_masked, t_mod)
                text_f = text_f * text_mask_bool.unsqueeze(-1).to(text_f.dtype)
        
        else:
            if block_mask is None:
                text_indices = torch.arange(L_text_padded, device=device).unsqueeze(0).expand(B, -1)
                text_mask = text_indices < text_lens.unsqueeze(-1)
                text_chunk_ids = torch.where(text_mask, torch.ones_like(text_indices), torch.zeros_like(text_indices))
                latent_chunk_ids = id2chunk + 1
                joint_chunk_ids = torch.cat([text_chunk_ids, latent_chunk_ids], dim=1)

                Q_LEN_current = current_end - current_start
                KV_LEN_current = L_text_padded + current_end
                
                block_mask = create_mask_with_padding_exclusion(
                    B=B, H=H, Q_LEN=Q_LEN_current, KV_LEN=KV_LEN_current,
                    joint_chunk_ids=joint_chunk_ids, device=device, 
                    q_start=L_text_padded + current_start
                )

            kv_cache["k"][:, :, L_text_padded + current_start:L_text_padded + current_end] = x_qkv[1]
            kv_cache["v"][:, :, L_text_padded + current_start:L_text_padded + current_end] = x_qkv[2]

            attn_out = attention(x_qkv[0], 
                                 kv_cache["k"][:, :, : L_text_padded + current_end], 
                                 kv_cache["v"][:, :, : L_text_padded + current_end], 
                                 block_mask=block_mask)
            latent = self.latent_block.post_attention(latent, attn_out, x_mod)

        return latent, text_f, block_mask

class CausalFinalBlock(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = CausalChannelLastConv1d(dim, out_dim, kernel_size=7)
    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent

class LinearFinalBlock(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = nn.Linear(dim, out_dim, bias=True)
    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent

class FinalBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent