from typing import Literal, Optional

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import Normalize

from model.mm.utils.distributions import DiagonalGaussianDistribution
from transformers import T5EncoderModel, AutoTokenizer, AutoModelForCausalLM

from model.audiovae.inference import load_model_from_config
from model.audiovae.audiovae_utils import get_hparams_from_file
import torchaudio
from transformers import T5GemmaModel, AutoTokenizer
import torchaudio.transforms as T

class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        audiovae_ckpt: Optional[str] = None,
        max_text_length: int = 100,
        compile_audiovae: bool = False,
        pad_to_max_length: bool = True,
        text_encoder_for_gen = 'qwen3',
        use_clap: bool = True,
        use_peav: bool = True,
        use_qwen3_audio_encoder: bool = True,
        use_ae_predictor: bool = True,
        use_whisper: bool = True,
    ):
        super().__init__()

        model_path = "./ext_weights/t5gemma-l-l-ul2-it"
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.t5_model = T5GemmaModel.from_pretrained(model_path).encoder
        
        self.max_length = max_text_length
        config_path = os.path.join(os.path.dirname(audiovae_ckpt), "config.yaml")
        hps = get_hparams_from_file(config_path)
        self.audiovae = load_model_from_config(hps.model, audiovae_ckpt, ver='vae')
        
        if compile_audiovae:
            self.compile_audiovae()

    def compile_audiovae(self):
        self.audiovae = torch.compile(self.audiovae.to(torch.bfloat16), mode="max-autotune")

    def compile(self):
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """

        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths
    def get_audio_features(self, audios):
        text = ''
        inputs = self.audio_processor(text=text, audio=audios, return_tensors="pt", padding=True)
        input_features = inputs['input_features'].to(self.device).to(self.dtype)
        feature_attention_mask = inputs['feature_attention_mask'].to(self.device)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens.long(),
        )
        audio_features = audio_outputs.last_hidden_state
        output_lengths = self._get_feat_extract_output_lengths(feature_lens)
    
        split_sizes = output_lengths.tolist()
        if audio_features.shape[0] != sum(split_sizes):
            print(f"Error:the total length of splits: ({sum(split_sizes)}), the actual length: ({audio_features.shape[0]})is not equal.")
            
        audio_features_list = audio_features.split(split_sizes, dim=0)
        audio_features_padded = torch.nn.utils.rnn.pad_sequence(
            audio_features_list, batch_first=True, padding_value=0.0
        )
        return audio_features_padded

    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_t5_text(self, text: list[str], layer_indices: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.t5_model is not None, 'T5 model is not loaded'
        assert self.t5_tokenizer is not None, 'T5 Tokenizer is not loaded'

        # 1. Tokenize
        inputs = self.t5_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        text_lens = inputs.attention_mask.sum(dim=1)

        # 2. Forward pass
        outputs = self.t5_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True if layer_indices is not None else False,
        )

        if layer_indices is None:
            prompt_embeds = outputs.last_hidden_state
        else:
            all_hidden_states = outputs.hidden_states
            num_layers = len(all_hidden_states)
            
            selected_features = []
            for idx in layer_indices:
                actual_idx = idx if idx >= 0 else num_layers + idx
                if 0 <= actual_idx < num_layers:
                    selected_features.append(all_hidden_states[actual_idx])
                else:
                    raise IndexError(f"Layer index {idx} out of range (total layers: {num_layers})")
            
            if len(selected_features) > 1:
                prompt_embeds = torch.stack(selected_features).mean(dim=0)
            else:
                prompt_embeds = selected_features[0]

        return prompt_embeds, text_lens.to(torch.long)
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result
    @torch.inference_mode()
    def encode_qwen_text(self, text: list[str], layer_indices: list = None) -> torch.Tensor:
        assert self.qwen3_model is not None, 'qwen3 model is not loaded'
        assert self.qwen3_tokenizer is not None, 'qwen3 Tokenizer is not loaded'
        if layer_indices is None:
            layer_indices = [-1]

        drop_idx = self.prompt_template_encode_start_idx
        txt = [self.qwen3_template.format(t) for t in text]
        model_inputs = self.qwen3_tokenizer(txt, 
            max_length=drop_idx + self.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt").to(self.device)
        encoder_hidden_states = self.qwen3_model(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states
        selected_layers = []
        num_hidden_layers = len(hidden_states) - 1
        for idx in layer_indices:
            real_idx = idx if idx >= 0 else num_hidden_layers + idx + 1
            if real_idx < 0 or real_idx >= len(hidden_states):
                raise IndexError(f"Layer index {idx} out of range for {len(hidden_states)} layers.")
            selected_layers.append(hidden_states[real_idx])  # each: [B, T, D]

        fused_hidden = torch.stack(selected_layers).mean(dim=0)  # 可选：mean 融合

        full_attention_mask = model_inputs.attention_mask # [B, T_full]
        target_mask = full_attention_mask[:, drop_idx:] # [B, T_target]
        text_lens = target_mask.sum(dim=1) # [B]
                
        if self.pad_to_max_length:
            split_hidden_states_full = self._extract_masked_hidden(fused_hidden, full_attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states_full]
            max_seq_len = self.max_length
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) 
                 for u in split_hidden_states]
            ) 
        else:
            prompt_embeds = fused_hidden[:, drop_idx:] # [B, L_target, D]
        
        return prompt_embeds, text_lens.to(torch.long) 
    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.audiovae is not None, 'VAE is not loaded'
        x = self.audiovae.encode_to_dist(x)
        dist = DiagonalGaussianDistribution(x)
        latent = dist.sample()
        
        return latent

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
