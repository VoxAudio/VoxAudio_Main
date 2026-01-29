from typing import Optional

import torch


def log_normal_sample(x: torch.Tensor,
                      generator: Optional[torch.Generator] = None,
                      m: float = 0.0,
                      s: float = 1.0) -> torch.Tensor:
    bs = x.shape[0]
    s = torch.randn(bs, device=x.device, generator=generator) * s + m
    return torch.sigmoid(s)

def log_normal_sample_2d(
    x: torch.Tensor,
    chunk_size: int = 1,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0
) -> torch.Tensor:
    """
    Sample a random time step for each chunk, shared across all tokens in the chunk.

    Args:
        x: Input tensor of shape [B, T, ...]
        chunk_size: int, size of each chunk (e.g., 64)
        generator: RNG generator
        m: mean of normal distribution
        s: std of normal distribution

    Returns:
        timesteps: [B, T], same value within each chunk
    """
    B, T = x.shape[:2]

    num_chunks = (T + chunk_size - 1) // chunk_size
    rand = torch.randn(B, num_chunks, device=x.device, generator=generator) * s + m  # [B, num_chunks]
    timesteps = rand.unsqueeze(-1).expand(-1, -1, chunk_size).reshape(B, -1)  # [B, num_chunks * chunk_size]
    timesteps = timesteps[:, :T]

    return torch.sigmoid(timesteps)


def sample_2d(
    x: torch.Tensor,
    chunk_size: int = 1,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Sample a random time step for each chunk, shared across all tokens in the chunk.

    Args:
        x: Input tensor of shape [B, T, ...]
        chunk_size: int, size of each chunk (e.g., 64)
        generator: RNG generator
        m: mean of normal distribution
        s: std of normal distribution

    Returns:
        timesteps: [B, T], same value within each chunk
    """
    B, T = x.shape[:2]

    num_chunks = (T + chunk_size - 1) // chunk_size
    rand = torch.rand(B, num_chunks, device=x.device, generator=generator)
    timesteps = rand.unsqueeze(-1).expand(-1, -1, chunk_size).reshape(B, -1)  # [B, num_chunks * chunk_size]
    timesteps = timesteps[:, :T]

    return timesteps


def log_normal_sample_2d_vectorized(
    x: torch.Tensor,
    id2chunk: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0
) -> torch.Tensor:
    """
    Sample a random time step for each chunk, shared within each chunk.
    Supports per-sample chunk ID starting from 1 (independent across batch).

    Args:
        x: [B, T, ...]
        id2chunk: [B, T], each row is independently numbered from 1,2,3,...
        generator: RNG
        m, s: mean and std of normal

    Returns:
        timesteps: [B, T], same value in each chunk, then sigmoided
    """
    B, T = x.shape[:2]
    device = x.device
    timesteps = torch.zeros(B, T, device=device)

    for b in range(B):
        ids = id2chunk[b]
        unique_ids, inverse_indices = ids.unique(return_inverse=True)

        num_chunks = len(unique_ids)
        rand_times = torch.randn(num_chunks, device=device, generator=generator) * s + m  # [num_chunks]

        timesteps[b] = rand_times[inverse_indices]  # (T,)

    return torch.sigmoid(timesteps)  # [B, T]

def sample_2d_vectorized(
    x: torch.Tensor,
    id2chunk: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0
) -> torch.Tensor:
    """
    Sample a random time step for each chunk, shared within each chunk.
    Supports per-sample chunk ID starting from 1 (independent across batch).

    Args:
        x: [B, T, ...]
        id2chunk: [B, T], each row is independently numbered from 1,2,3,...
        generator: RNG
        m, s: mean and std of normal

    Returns:
        timesteps: [B, T], same value in each chunk, then sigmoided
    """
    B, T = x.shape[:2]
    device = x.device
    timesteps = torch.zeros(B, T, device=device)

    for b in range(B):
        ids = id2chunk[b]
        unique_ids, inverse_indices = ids.unique(return_inverse=True)

        num_chunks = len(unique_ids)
        rand_times = torch.rand(num_chunks, device=device, generator=generator) * s + m  # [num_chunks]

        timesteps[b] = rand_times[inverse_indices]  # (T,)

    return timesteps  # [B, T]

def sample_2d_first(
    x: torch.Tensor,
    id2chunk: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0,
    t_first_chunk_list = None 
) -> torch.Tensor:
    """
    Sample a random time step for each chunk, shared within each chunk.
    Supports per-sample chunk ID starting from 1 (independent across batch).

    Args:
        x: [B, T, ...]
        id2chunk: [B, T], each row is independently numbered from 1, 2, 3,...
        generator: RNG
        m, s: mean and std of normal
        t_first_chunk_list: Optional list. If specified, the timestep for the first
                       chunk (ID=1) will be set to this value for all samples.

    Returns:
        timesteps: [B, T], same value in each chunk.
    """
    B, T = x.shape[:2]
    device = x.device
    timesteps = torch.zeros(B, T, device=device)

    use_fixed_first_t = t_first_chunk_list is not None
    first_chunk_id = 1 

    for b in range(B):
        ids = id2chunk[b]
        
        unique_ids, inverse_indices = ids.unique(return_inverse=True)
        num_chunks = len(unique_ids)

        rand_times = torch.rand(num_chunks, device=device, generator=generator) * s + m  # [num_chunks]

        if use_fixed_first_t and first_chunk_id in unique_ids:
            first_chunk_idx = (unique_ids == first_chunk_id).nonzero(as_tuple=True)[0]
            
            if t_first_chunk_list[b] != None:
                rand_times[first_chunk_idx] = t_first_chunk_list[b]
                
        timesteps[b] = rand_times[inverse_indices]  # (T,)

    return timesteps  # [B, T]

def sample_2d_decrease(
    x: torch.Tensor,
    id2chunk: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0,
    t_first_chunk_list = None,
    overflow_prob: float = 0.5,
    max_overflow_ratio: float = 2.0 
) -> torch.Tensor:
    """
    Samples timesteps:
    1. Generates an initial sequence (t_1..N) that is monotonically decreasing (t_1 >= t_2 >= ...).
    2. Overrides t_1 with a fixed value (t_fixed) if specified, potentially breaking monotonicity t_1 < t_2.
    """
    B, T = x.shape[:2]
    device = x.device
    timesteps = torch.zeros(B, T, device=device)

    if t_first_chunk_list is None:
        t_first_chunk_list = [None] * B

    for b in range(B):
        ids = id2chunk[b]
        unique_ids, inverse_indices = ids.unique(return_inverse=True)
        num_chunks = len(unique_ids)
            
        base_max = s + m
        do_overflow = torch.rand(1, generator=generator).item() < overflow_prob
        
        if do_overflow:
            ratio = torch.rand(1, generator=generator, device=device) * (max_overflow_ratio - 1.0) + 1.0
            current_max = base_max * ratio
        else:
            current_max = torch.rand(1, generator=generator, device=device)
        
        noise = torch.rand(num_chunks, device=device, generator=generator)
        t_raw = noise * current_max
        t_sorted, _ = torch.sort(t_raw, descending=True)
        chunk_times = torch.clamp(t_sorted, max=(s + m))

        fixed_t1 = t_first_chunk_list[b]
        if fixed_t1 is not None:
            chunk_times[0] = float(fixed_t1)

        timesteps[b] = chunk_times[inverse_indices]

    return timesteps