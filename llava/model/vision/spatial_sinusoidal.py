
import torch


def _build_sinusoidal_embedding(positions: torch.Tensor, dim: int, total_dim: int) -> torch.Tensor:
    """Construct a 1D sinusoidal embedding for normalized positions.

    Args:
        positions: Tensor of shape (...,) containing normalized coordinates.
        dim: Number of dimensions allocated for this axis.
        total_dim: Total embedding dimension (D).
    """
    device = positions.device
    dtype = positions.dtype
    positions = positions.unsqueeze(-1)  # (..., 1)

    # Frequencies follow the stand+ard transformer positional encoding schedule.
    freq_exponents = -torch.arange(0, dim, 2, device=device, dtype=dtype) / total_dim
    freq = torch.pow(10000.0, freq_exponents)  # (dim/2,)

    # Compute sinusoidal components.
    sinusoid_inp = positions * freq  # (..., dim/2)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    # Interleave sin and cos to match the desired shape (..., dim).
    embedding = torch.empty(*positions.shape[:-1], dim, device=device, dtype=dtype)
    embedding[..., 0::2] = sin
    embedding[..., 1::2] = cos
    return embedding


def add_2d_sinusoidal_spatial_embedding(vision_tokens: torch.Tensor, grid_size, alpha: float = 0.1) -> torch.Tensor:
    """
    Add a 2D sinusoidal spatial embedding to vision tokens.

    Args:
        vision_tokens: Tensor of shape (B, N, D) or (N, D).
        grid_size: Tuple (H, W) such that H * W == N.
        alpha: Scaling factor for the positional embedding.

    Returns:
        Tensor with the same shape as ``vision_tokens`` augmented with spatial embeddings.
    """
    if vision_tokens.dim() not in (2, 3):
        raise ValueError("vision_tokens must have shape (N, D) or (B, N, D)")

    has_batch = vision_tokens.dim() == 3
    if has_batch:
        batch, num_tokens, dim = vision_tokens.shape
    else:
        num_tokens, dim = vision_tokens.shape
        batch = None

    height, width = grid_size
    if height * width != num_tokens:
        raise ValueError(
            f"grid_size {grid_size} does not match number of tokens {num_tokens}"
        )

    device = vision_tokens.device
    dtype = vision_tokens.dtype

    # Generate normalized coordinates.
    y_coords = torch.arange(height, device=device, dtype=dtype)
    x_coords = torch.arange(width, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    x_norm = x_grid.reshape(-1) / (width - 1 if width > 1 else 1)
    y_norm = y_grid.reshape(-1) / (height - 1 if height > 1 else 1)

    dim_x = dim // 2
    dim_y = dim - dim_x

    x_embedding = _build_sinusoidal_embedding(x_norm, dim_x, dim)
    y_embedding = _build_sinusoidal_embedding(y_norm, dim_y, dim)
    spatial_embedding = torch.cat([x_embedding, y_embedding], dim=-1)  # (N, D)

    if has_batch:
        spatial_embedding = spatial_embedding.unsqueeze(0).expand(batch, -1, -1)

    return vision_tokens + alpha * spatial_embedding
