import os
import sys
import copy
import yaml
import random
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import subprocess
import glob
import argparse
from model.voxaudio import VoxAudio, PreprocessedConditions
import random
from typing import List, Optional
import json
import torch.distributed as dist
import shutil

def get_id2chunk(
    B: int,
    L: int,
    chunk_size: int,
    seq_len: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:

    if seq_len is None:
        seq_len = [L] * B

    id2chunk = torch.zeros(B, L, dtype=torch.long, device=device)

    for b in range(B):
        actual_len = seq_len[b]
        if actual_len <= 0:
            continue

        start = 0
        chunk_id = 1

        while start < actual_len:
            end = min(start + chunk_size, actual_len)
            id2chunk[b, start:end] = chunk_id
            start = end
            chunk_id += 1

        if start <= L:
            id2chunk[b, start:L] = chunk_id

    return id2chunk
    
def add_inference_args():
    parser = argparse.ArgumentParser(description="Inference parser for VoxAudio")
    parser.add_argument('--config', type=str, default='./doft_audio_t5.yaml', help='tts config.yaml')
    parser.add_argument('--load_ckpt_path', type=str, default='./ckpt/model_only_last.ckpt')
    parser.add_argument('--meta_fn', type=str, default='./data/voxaudio_test.jsonl', help='input meta path')
    parser.add_argument('--output_dir', type=str, default='inference_output', help='Output directory for generated audio')
    parser.add_argument('--num_samples_per_prompt', type=int, default=1, help='Number of samples to generate for each caption')
    parser.add_argument('--text_cfg', type=float, default=6.0, help='text classifier free guidance')
    parser.add_argument('--hist_cfg', type=float, default=1.0, help='history classifier free guidance')
    parser.add_argument('--step_interval', type=int, default=1, help='step interval')
    parser.add_argument('--num_steps', type=int, default=20, help='step interval')
    args, _ = parser.parse_known_args()
    if args.config:
        try:
            def override_config(old_config: dict, new_config: dict):
                for k, v in new_config.items():
                    if isinstance(v, dict) and k in old_config:
                        override_config(old_config[k], new_config[k])
                    else:
                        old_config[k] = v
            def load_config(config_fn):
                loaded_config = set()
                # deep first inheritance and avoid the second visit of one node
                if not os.path.exists(config_fn):
                    return {}
                with open(config_fn, 'r', encoding='utf-8') as f:
                    hparams_ = yaml.safe_load(f)
                loaded_config.add(config_fn)
                if 'base_config' in hparams_:
                    ret_hparams = {}
                    if not isinstance(hparams_['base_config'], list):
                        hparams_['base_config'] = [hparams_['base_config']]
                    for c in hparams_['base_config']:
                        if c.startswith('.'):
                            c = f'{os.path.dirname(config_fn)}/{c}'
                            c = os.path.normpath(c)
                        if c not in loaded_config:
                            override_config(ret_hparams, load_config(c))
                    override_config(ret_hparams, hparams_)
                else:
                    ret_hparams = hparams_
                return ret_hparams
            yaml_config = load_config(args.config)
            
            # Update the args namespace directly
            for key, value in yaml_config.items():
                setattr(args, key, value)
            
            # Apply to parser defaults
            parser.set_defaults(**yaml_config)

        except FileNotFoundError:
            print(f"Warning: Configuration file {args.config} not found. Using only command-line arguments or in-code defaults.")
        except Exception as e:
            print(f"Error: Failed to parse configuration file {args.config}: {e}")
            exit(1)
    args, _ = parser.parse_known_args()
    return args

def initialize_kv_cache(args, batch_size, num_heads, depth, dtype, device):
    kv_cache = []

    for _ in range(depth):
        kv_cache.append({
            "k": torch.zeros([batch_size, num_heads, 2048, 512//num_heads], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, num_heads, 2048, 512//num_heads], dtype=dtype, device=device),
            "is_init": False,
        })

    return kv_cache

def generate_audio(args, model, feature_extractor, text_f, text_lens):
    latent_dim = getattr(args, 'latent_dim', 20)
    chunk_size = getattr(args, 'chunk_size', 8)
    batch_size = text_f.shape[0]
    num_steps = args.num_steps
    cfg_text = args.text_cfg

    num_frames = 125

    dtype = next(model.parameters()).dtype
    text_f = text_f.to(dtype=dtype)
    device = text_f.device

    x0 = torch.randn(batch_size, num_frames, latent_dim).to(dtype=dtype, device=device)
    conditions = model.preprocess_conditions(text_f, text_lens)
    empty_conditions = model.get_empty_conditions(batch_size, text_lens)
    id2chunk = get_id2chunk(
        B=batch_size, L=num_frames, chunk_size=chunk_size,
        seq_len=None,
        device=device
    )
    
    steps = torch.linspace(0, 1, num_steps + 1)
    dt = 1.0 / num_steps
    x = x0
    for ti, t in enumerate(steps[:-1]):
        input_t = torch.ones(batch_size, num_frames, device=device, dtype=dtype) * t
        flow_t, eco_logits = model.ode_wrapper_test(input_t, x, conditions, text_lens, id2chunk=id2chunk)
        flow, eco_logits = model.ode_wrapper_test(input_t, x, empty_conditions, text_lens, id2chunk=id2chunk)
        final_flow = cfg_text * (flow_t - flow) + flow
        x = x + dt * final_flow
    output = x
   
    audio_pred = feature_extractor.audiovae.decode(output.transpose(1, 2)).to(torch.float32)

    return audio_pred 

def generate_audio_windows(args, model, feature_extractor, text_f, text_lens):
    num_steps = args.num_steps
    iterval_t = args.step_interval
    cfg_text = args.text_cfg
    cfg_hist = args.hist_cfg

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    text_f = text_f.to(dtype=dtype)
    num_frames = 125
    chunk_size = args.chunk_size
    latent_dim = args.latent_dim
    batch_size = text_f.shape[0]
    x0 = torch.randn(batch_size, num_frames, latent_dim).to(dtype=dtype, device=device)
    output = torch.zeros_like(x0)
    conditions = model.preprocess_conditions(text_f, text_lens)
    empty_conditions = model.get_empty_conditions(text_f.shape[0], text_lens)
    batch_size, num_frames, num_channels = x0.shape
    id2chunk = get_id2chunk(
        B=batch_size, L=num_frames, chunk_size=chunk_size,
        seq_len=None,
        device=device,
    )
    kv_cache_full = initialize_kv_cache(args=args, batch_size=batch_size * 2, num_heads=args.num_heads, depth=args.depth, dtype=x0.dtype, device=x0.device)
    
    input_t = torch.zeros(batch_size, num_frames, device=x0.device, dtype=x0.dtype)
    x0_cat = torch.cat([x0, x0], dim=0)
    id2chunk_cat = torch.cat([id2chunk, id2chunk], dim=0)
    text_lens_cat = torch.cat([text_lens, text_lens], dim=0)
    cond_cat = PreprocessedConditions(
        text_f=torch.cat([conditions.text_f, empty_conditions.text_f], dim=0), 
        text_f_c=torch.cat([conditions.text_f_c, empty_conditions.text_f_c], dim=0), 
        text_lens=torch.cat([conditions.text_lens, empty_conditions.text_lens], dim=0), 
    )

    current_start = 0
    current_end = chunk_size
    dt = 1.0 / num_steps

    all_step = 0
    while current_start < current_end:
        xt = x0[:, current_start:current_end]
        t = input_t[:, current_start:current_end] / num_steps
        
        xt_cat = torch.cat([xt, xt], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        flows_cat, eoc_logits = model.ode_wrapper_flush_cache(t_cat, xt_cat, current_start, current_end, cond_cat, text_lens_cat, id2chunk=id2chunk_cat, kv_cache_con=kv_cache_full)
        flow_t, flow = flows_cat.chunk(2, dim=0)
        flow_final = cfg_text * (flow_t - flow) + flow
       
        xt = xt + dt * flow_final
        input_t[:, current_start:current_end] = input_t[:, current_start:current_end] + 1
        x0[:, current_start:current_end] = xt
        t_end = min(current_start + chunk_size, num_frames)
        if torch.all(input_t[:, current_start:t_end] == num_steps).item():
            current_start = min(current_start + chunk_size, num_frames)
        all_step += 1
        if all_step % iterval_t == 0:
            current_end = min(current_end + chunk_size, num_frames)

    output = x0

    audio_pred = feature_extractor.audiovae.decode(output.transpose(1, 2)).to(torch.float32)

    return audio_pred 

def load_dataset(meta_fn):
    metadata_list = []
    with open(meta_fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                metadata_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {e}")
                continue
                
    return metadata_list

def load_dataset_tsv(meta_fn):
    metadata_list = []
    with open(meta_fn, 'r') as f:
        for idx, line in enumerate(f.readlines()[1:]):
            uniq_id, _, caption, _ = line.strip().split('\t')
            sample = {
                'rank': idx,
                'caption': caption.strip(),
                'gt_audio_path': os.path.join('./data/gt', f'Y{uniq_id}.wav'),
            }
            metadata_list.append(sample)
    return metadata_list

def model_provider(args=None, pre_process=True, post_process=True):
    model = VoxAudio(
        args=args,
        latent_dim=args.latent_dim,
        text_dim=getattr(args, 'text_encoder_dim_for_gen', 2560),
        hidden_dim=512,
        depth=args.depth,
        fused_depth=args.fused_depth,
        num_heads=args.num_heads,
        latent_seq_len=args.latent_seq_len,
        text_seq_len=args.text_seq_len,
        v2=getattr(args, 'v2', False),
        chunk_size=args.chunk_size
    )

    total_params = sum(p.numel() for name, p in model.named_parameters() if 'feature_extractor' not in name)
    trainable_params = sum(p.numel() for name, p in model.named_parameters() if 'feature_extractor' not in name and p.requires_grad)
    print(f"✅ Model built. Total: {total_params:,}, Trainable: {trainable_params:,}")
    return model

def inference_from_dataset(model, feature_extractor, args):
    output_dir = getattr(args, 'output_dir', 'inference_output')
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples_per_prompt = getattr(args, 'num_samples_per_prompt', 1)
    meta_fn = args.meta_fn

    if meta_fn.endswith('.tsv'):
        ds = load_dataset_tsv(meta_fn)
    else:
        ds = load_dataset(meta_fn)

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    local_ds = ds[rank::world_size]
    
    part_jsonl_filename = f"inference_results_rank_{rank}.jsonl"
    part_jsonl_path = os.path.join(output_dir, part_jsonl_filename)
    f_jsonl = open(part_jsonl_path, 'w', encoding='utf-8')

    text_feature_layer = getattr(args, 'text_feature_layer', None)
    batch_size = 32
    sample_rate = getattr(args, 'audio_sample_rate', 24000)
    
    if rank == 0:
        print(f"Starting inference: Total {len(ds)} prompts, split across {world_size} GPUs.")
    print(f"[Rank {rank}] Processing {len(local_ds)} prompts...")

    iterator = tqdm(range(0, len(local_ds), batch_size)) if rank == 0 else range(0, len(local_ds), batch_size)

    with torch.no_grad():
        for i in iterator:
            data_batch = local_ds[i : i + batch_size]
            captions = [item['caption'] for item in data_batch]
            original_ranks = [item['rank'] for item in data_batch]

            for sample_idx in range(num_samples_per_prompt):
                with torch.inference_mode():
                    text_features, text_lens = feature_extractor.encode_t5_text(captions, text_feature_layer)
                    
                    if args.step_interval == 0:
                        audios = generate_audio(
                            args,
                            model, 
                            feature_extractor, 
                            text_f=text_features.to(torch.bfloat16), 
                            text_lens=text_lens
                        )
                    else:
                        audios = generate_audio_windows(
                            args,
                            model, 
                            feature_extractor, 
                            text_f=text_features.to(torch.bfloat16), 
                            text_lens=text_lens
                        )

                for j, audio in enumerate(audios):
                    rank_id = original_ranks[j]
                    uniq_id = f"{rank_id}_{sample_idx}"
                    
                    audio_filename = f"{uniq_id}.wav"
                    audio_path = os.path.join(output_dir, 'audio', audio_filename)
                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                    
                    torchaudio.save(audio_path, audio.cpu().float(), sample_rate)
                    
                    item = data_batch[j]
                    item['audio_path'] = audio_path
                    item['uniq_id'] = uniq_id
                    
                    f_jsonl.write(json.dumps(item, ensure_ascii=False) + '\n')
                    f_jsonl.flush()

    f_jsonl.close()
    
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("Merging result files...")
        final_output_path = os.path.join(output_dir, "inference_results.jsonl")
        
        with open(final_output_path, 'w', encoding='utf-8') as outfile:
            for r in range(world_size):
                part_fn = os.path.join(output_dir, f"inference_results_rank_{r}.jsonl")
                if os.path.exists(part_fn):
                    with open(part_fn, 'r', encoding='utf-8') as infile:
                        shutil.copyfileobj(infile, outfile)
                    os.remove(part_fn)
        
        print(f"✅ All results merged and saved to {final_output_path}")

def main():
    args = add_inference_args()
    model = model_provider(args)
    device = torch.cuda.current_device()
    model = model.to(device)
    model_state_dict = torch.load(args.load_ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(model_state_dict['model'], strict=False)
    model = model.to(torch.bfloat16)
    model.eval()
    model.latent_rot = model.latent_rot.to(device=device)
    feature_extractor = model.feature_extractor

    feature_extractor.audiovae = feature_extractor.audiovae.to(torch.bfloat16)
    feature_extractor.eval()

    inference_from_dataset(model, feature_extractor, args)

    print("🎉 Inference completed.")


if __name__ == "__main__":
    main()
