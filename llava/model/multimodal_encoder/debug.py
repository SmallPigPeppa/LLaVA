import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, Sequence
from itertools import repeat
import collections.abc
import numpy as np
from einops import rearrange
from functorch import vmap
from numpy.distutils.command.install import old_install

from .pos_resize import resize_abs_pos_embed


def to_2tuple(x: Any) -> Tuple[int, int]:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


class CLIPVisionEmbeddingsMSPE(nn.Module):
    """
    CLIP Vision Embeddings with Multi-Scale Patch Embedding (MSPE).
    Supports flexible patch sizes via pseudo-inverse resizing of the conv kernel.
    """

    def __init__(
        self,
        config,
        patch_size_seq: Sequence[int] = None,
        patch_size_probs: Optional[Sequence[float]] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size

        # Base patch parameters
        base_ps = to_2tuple(config.patch_size)
        grid = (self.image_size // base_ps[0], self.image_size // base_ps[1])
        self.num_patches = grid[0] * grid[1]
        self.num_positions = self.num_patches + 1

        # Projection conv and normalization
        self.proj = nn.Conv2d(
            config.num_channels,
            self.embed_dim,
            kernel_size=base_ps,
            stride=base_ps,
            bias=False,
        )

        # Flex settings
        self.interpolation = interpolation
        self.antialias = antialias
        self.base_ps = base_ps
        if patch_size_seq:
            self.patch_size_seq = [to_2tuple(ps) for ps in patch_size_seq]
            if patch_size_probs is None:
                p = 1.0 / len(self.patch_size_seq)
                self.patch_size_probs = [p] * len(self.patch_size_seq)
            else:
                s = sum(patch_size_probs)
                self.patch_size_probs = [p / s for p in patch_size_probs]
            # Precompute pseudo-inverse matrices
            self.pinvs = {}
            for ps in self.patch_size_seq:
                self.pinvs[ps] = self._calc_pinv(self.base_ps, ps)
        else:
            self.patch_size_seq = None
            self.patch_size_probs = []

        # Class and position embeddings
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def _pos_embed(self, x: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
        # Resize position embedding based on current patch size
        old_size = (self.image_size // self.base_ps[0], self.image_size // self.base_ps[1])
        new_size = (self.image_size // patch_size[0], self.image_size // patch_size[1])
        pos_embed = resize_abs_pos_embed(
            self.pos_embed,
            new_size=new_size,
            old_size=old_size,
            num_prefix_tokens=1,
            interpolation=self.interpolation,
            antialias=True
        )

        if self.no_embed_class:
            # Position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # Position embedding has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def _resize(self, x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias
        )
        return x_resized[0, 0, ...]

    def _calc_pinv(self, old_shape: Tuple[int, int], new_shape: Tuple[int, int]) -> torch.Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis = torch.zeros(old_shape, device=self.proj.weight.device)
            basis[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis, new_shape).reshape(-1))
        M = torch.stack(mat, dim=0)
        return torch.linalg.pinv(M)

    def _resize_kernel(self, weight: torch.Tensor, new_ps: Tuple[int, int]) -> torch.Tensor:
        if new_ps == self.base_ps:
            return weight
        if new_ps not in self.pinvs:
            self.pinvs[new_ps] = self._calc_pinv(self.base_ps, new_ps)
        pinv = self.pinvs[new_ps].to(weight.device)

        def _resample(kernel):
            h, w = new_ps
            v = pinv @ kernel.reshape(-1)
            return rearrange(v, '(h w) -> h w', h=h, w=w)

        # apply over output and input channels
        return vmap(vmap(_resample, in_dims=1, out_dims=1), in_dims=0, out_dims=0)(weight)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        '''
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
        '''
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # choose patch size (random if training)
        if self.training and self.patch_size_seq:
            ps = self.patch_size_seq[np.random.choice(len(self.patch_size_seq), p=self.patch_size_probs)]
        else:
            ps = self.base_ps
        # get conv weights
        weight = self._resize_kernel(self.proj.weight, ps)
        patch_embeds = F.conv2d(pixel_values.to(dtype=target_dtype), weight, bias=None, stride=ps)
        # flatten to sequence
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


def setup_mspe(model: Any,
               patch_size_seq: Sequence[int],
               patch_size_probs: Optional[Sequence[float]] = None,
               interpolation: str = "bicubic",
               antialias: bool = True) -> None:
    """
    Replace standard CLIPVisionEmbeddings on model.vision_tower with MSPE version.
    """
    vis = model.vision_tower.vision_model
    old = vis.embeddings
    mspe = CLIPVisionEmbeddingsMSPE(
        old.config,
        patch_size_seq=patch_size_seq,
        patch_size_probs=patch_size_probs,
        interpolation=interpolation,
        antialias=antialias,
    )
    # copy cls & pos embeddings
    mspe.class_embedding.data.copy_(old.class_embedding.data)
    mspe.position_embedding.weight.data.copy_(old.position_embedding.weight.data)
    # copy base conv weights
    mspe.proj.weight.data.copy_(old.patch_embedding.weight.data)
    # swap in
    vis.embeddings = mspe
