from pathlib import Path

import click
import numpy as np
from dataclasses import dataclass
import soundfile as sf
import torch
import torchaudio

def load_model_from_config(hps= None,checkpoint_path = None, device="cuda",ver=0):
    print("import ver vae")
    from megatron.model.audiovae.modded_dac_vae import DAC, WindowLimitedTransformer, find_multiple,ModelArgs
    from megatron.model.audiovae.vae import DownsampleVAE as DownsampleResidualVectorQuantize
    assert hps is not None
    @dataclass
    class TModelArgs:
        block_size: int = hps.targs.block_size
        n_layer: int = hps.targs.n_layer
        n_head: int = hps.targs.n_head
        dim: int = hps.targs.dim
        intermediate_size: int = hps.targs.intermediate_size
        n_local_heads: int = hps.targs.n_local_heads
        head_dim: int = hps.targs.head_dim
        rope_base: float = hps.targs.rope_base
        norm_eps: float = hps.targs.norm_eps
        dropout_rate: float = hps.targs.dropout_rate
        attn_dropout_rate: float = hps.targs.attn_dropout_rate
        channels_first: bool = hps.targs.channels_first  # to be compatible with conv1d input/output
        pos_embed_type: str = "rope"  # can be "rope" or "conformer"
        max_relative_position: int = 128  # for conformer-style relative position embedding

        def __post_init__(self):
            if self.n_local_heads == -1:
                self.n_local_heads = self.n_head
            if self.intermediate_size is None:
                hidden_dim = 4 * self.dim
                n_hidden = int(2 * hidden_dim / 3)
                self.intermediate_size = find_multiple(n_hidden, 256)
            assert self.pos_embed_type in [
                "rope",
                "conformer",
            ], "pos_embed_type must be either 'rope' or 'conformer'"

    tranformer_config_quantizer = ModelArgs(
        block_size=hps.post_module.args.block_size,
        n_layer=hps.post_module.args.n_layer,
        n_head=hps.post_module.args.n_head,
        dim=hps.post_module.args.dim,
        intermediate_size=hps.post_module.args.intermediate_size,
        n_local_heads=hps.post_module.args.n_local_heads,
        head_dim=hps.post_module.args.head_dim,
        rope_base=hps.post_module.args.rope_base,
        norm_eps=hps.post_module.args.norm_eps,
        dropout_rate=hps.post_module.args.dropout_rate,
        attn_dropout_rate=hps.post_module.args.attn_dropout_rate,
        channels_first=hps.post_module.args.channels_first,
    )

    post_module = WindowLimitedTransformer(
        config = tranformer_config_quantizer,
        input_dim = hps.post_module.input_dim,
        window_size = hps.post_module.window_size,
        causal = hps.post_module.causal,
        # look_ahead_conv = None,
    )

    pre_module = WindowLimitedTransformer(
        config = tranformer_config_quantizer,
        input_dim = hps.post_module.input_dim,
        window_size = hps.post_module.window_size,
        causal = hps.post_module.causal,
        # look_ahead_conv = None,
    )
    
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=hps.quantizer.input_dim,
        n_codebooks=hps.quantizer.n_codebooks,
        codebook_size=hps.quantizer.codebook_size,
        codebook_dim=hps.quantizer.codebook_dim,
        quantizer_dropout=hps.quantizer.quantizer_dropout,
        downsample_factor=hps.quantizer.downsample_factor,
        post_module = post_module,
        pre_module = pre_module,
        semantic_codebook_size= hps.quantizer.semantic_codebook_size,
    )

    model = DAC(
        encoder_dim = hps.dac.encoder_dim,
        encoder_rates = hps.dac.encoder_rates,
        latent_dim= None,
        decoder_dim = hps.dac.decoder_dim,
        decoder_rates = hps.dac.decoder_rates,
        quantizer=quantizer,
        sample_rate = hps.dac.sample_rate,
        causal = hps.dac.causal,
        encoder_transformer_layers = hps.dac.encoder_transformer_layers,
        decoder_transformer_layers = hps.dac.decoder_transformer_layers,
        transformer_general_config=TModelArgs,
    )
    if checkpoint_path is not None:
        state_dict = torch.load(
            checkpoint_path, map_location=device, mmap=True, weights_only=True
        )
        if "model" in state_dict:
            state_dict = state_dict["model"]

        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }

        result = model.load_state_dict(state_dict, strict=False, assign=True)
        # result = None
        print(f"Loaded model: {result}")

    model.to(device)
    return model