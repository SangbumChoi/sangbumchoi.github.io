"""Vanilla PyTorch attention variants: single-head, legacy MHA, MHA, GQA, MQA.

This is a readable reference implementation, not a fused/high-performance kernel.
Run: python scripts/attention_variants.py --benchmark
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class KVCache:
    key: Tensor    # [batch, kv_heads, sequence, head_dim]
    value: Tensor  # [batch, kv_heads, sequence, head_dim]
    next_position: int = 0  # logical position, not the physical cache slot


def _apply_rope(x: Tensor, position_offset: int, base: float = 10_000.0) -> Tensor:
    """Apply split-half RoPE to [B, heads, T, head_dim] query/key vectors."""
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even head_dim")
    half = head_dim // 2
    positions = torch.arange(
        position_offset, position_offset + x.size(-2), device=x.device, dtype=torch.float32
    )
    inv_freq = 1.0 / (base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def _attention(q: Tensor, k: Tensor, v: Tensor, *, causal: bool) -> Tensor:
    """Attention for equal Q/KV head counts; tensors use [B,H,T,D]."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        q_len, k_len = q.size(-2), k.size(-2)
        # Supports both full prefill (q_len == k_len) and cached decode.
        q_positions = torch.arange(k_len - q_len, k_len, device=q.device)
        k_positions = torch.arange(k_len, device=q.device)
        mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    return scores.softmax(dim=-1) @ v


def _grouped_attention(q: Tensor, k: Tensor, v: Tensor, *, causal: bool) -> Tensor:
    """GQA/MQA without expanding K/V to the query-head count.

    q: [B, Hq, T, D], k/v: [B, Hkv, S, D], Hq must be divisible by Hkv.
    """
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.size(1), k.size(2)
    if q_heads % kv_heads:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    group_size = q_heads // kv_heads
    q = q.reshape(batch, kv_heads, group_size, q_len, head_dim)
    scores = torch.einsum("bghqd,bgkd->bghqk", q, k) / math.sqrt(head_dim)
    if causal:
        q_positions = torch.arange(kv_len - q_len, kv_len, device=q.device)
        k_positions = torch.arange(kv_len, device=q.device)
        mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        scores = scores.masked_fill(mask[None, None, None, :, :], torch.finfo(scores.dtype).min)
    probs = scores.softmax(dim=-1)
    out = torch.einsum("bghqk,bgkd->bghqd", probs, v)
    return out.reshape(batch, q_heads, q_len, head_dim)


class SingleHeadAttention(nn.Module):
    """Original single-head scaled dot-product attention, projected to d_model."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model, self.num_heads, self.num_kv_heads = d_model, 1, 1
        self.head_dim = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        shape = (x.size(0), 1, x.size(1), self.d_model)
        return (self.q_proj(x).view(shape), self.k_proj(x).view(shape), self.v_proj(x).view(shape))

    def forward(self, x: Tensor, *, causal: bool = True) -> Tensor:
        q, k, v = self._project(x)
        return self.out_proj(_attention(q, k, v, causal=causal).transpose(1, 2).reshape_as(x))

    def prefill(self, x: Tensor, *, causal: bool = True) -> tuple[Tensor, KVCache]:
        q, k, v = self._project(x)
        y = self.out_proj(_attention(q, k, v, causal=causal).transpose(1, 2).reshape_as(x))
        return y, KVCache(k, v)

    def decode_step(self, x: Tensor, cache: KVCache) -> tuple[Tensor, KVCache]:
        q, k, v = self._project(x)
        cache = KVCache(torch.cat((cache.key, k), 2), torch.cat((cache.value, v), 2))
        y = self.out_proj(_attention(q, cache.key, cache.value, causal=True).transpose(1, 2).reshape_as(x))
        return y, cache


class LegacyMultiHeadAttention(nn.Module):
    """Educational pre-packed MHA: each head owns separate Q/K/V projections."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model, self.num_heads, self.num_kv_heads = d_model, num_heads, num_heads
        self.head_dim = d_model // num_heads
        self.q_heads = nn.ModuleList(nn.Linear(d_model, self.head_dim) for _ in range(num_heads))
        self.k_heads = nn.ModuleList(nn.Linear(d_model, self.head_dim) for _ in range(num_heads))
        self.v_heads = nn.ModuleList(nn.Linear(d_model, self.head_dim) for _ in range(num_heads))
        self.out_proj = nn.Linear(d_model, d_model)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = torch.stack([p(x) for p in self.q_heads], dim=1)
        k = torch.stack([p(x) for p in self.k_heads], dim=1)
        v = torch.stack([p(x) for p in self.v_heads], dim=1)
        return q, k, v

    def forward(self, x: Tensor, *, causal: bool = True) -> Tensor:
        q, k, v = self._project(x)
        z = _attention(q, k, v, causal=causal).transpose(1, 2).reshape_as(x)
        return self.out_proj(z)

    def prefill(self, x: Tensor, *, causal: bool = True) -> tuple[Tensor, KVCache]:
        q, k, v = self._project(x)
        y = self.out_proj(_attention(q, k, v, causal=causal).transpose(1, 2).reshape_as(x))
        return y, KVCache(k, v)

    def decode_step(self, x: Tensor, cache: KVCache) -> tuple[Tensor, KVCache]:
        q, k, v = self._project(x)
        cache = KVCache(torch.cat((cache.key, k), 2), torch.cat((cache.value, v), 2))
        y = self.out_proj(_attention(q, cache.key, cache.value, causal=True).transpose(1, 2).reshape_as(x))
        return y, cache


class MultiHeadAttention(nn.Module):
    """Standard MHA with packed Q/K/V projections."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model, self.num_heads, self.num_kv_heads = d_model, num_heads, num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, t, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        shape = (b, t, self.num_heads, self.head_dim)
        return tuple(a.view(shape).transpose(1, 2) for a in (q, k, v))

    def forward(self, x: Tensor, *, causal: bool = True) -> Tensor:
        q, k, v = self._project(x)
        z = _attention(q, k, v, causal=causal).transpose(1, 2).reshape_as(x)
        return self.out_proj(z)

    def prefill(self, x: Tensor, *, causal: bool = True) -> tuple[Tensor, KVCache]:
        q, k, v = self._project(x)
        y = self.out_proj(_attention(q, k, v, causal=causal).transpose(1, 2).reshape_as(x))
        return y, KVCache(k, v)

    def decode_step(self, x: Tensor, cache: KVCache) -> tuple[Tensor, KVCache]:
        q, k, v = self._project(x)
        cache = KVCache(torch.cat((cache.key, k), 2), torch.cat((cache.value, v), 2))
        y = self.out_proj(_attention(q, cache.key, cache.value, causal=True).transpose(1, 2).reshape_as(x))
        return y, cache


class GroupedQueryAttention(nn.Module):
    """GQA when 1 < num_kv_heads < num_query_heads; 1 gives MQA."""
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        *,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        if d_model % num_query_heads:
            raise ValueError("d_model must be divisible by num_query_heads")
        if num_query_heads % num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        self.d_model = d_model
        self.num_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.use_rope = use_rope
        self.rope_base = rope_base
        if use_rope and self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        shape = (b, t, self.num_kv_heads, self.head_dim)
        return q, k.view(shape).transpose(1, 2), v.view(shape).transpose(1, 2)

    def forward(self, x: Tensor, *, causal: bool = True, position_offset: int = 0) -> Tensor:
        q, k, v = self._project(x)
        if self.use_rope:
            q = _apply_rope(q, position_offset, self.rope_base)
            k = _apply_rope(k, position_offset, self.rope_base)
        z = _grouped_attention(q, k, v, causal=causal).transpose(1, 2).reshape_as(x)
        return self.out_proj(z)

    def prefill(
        self, x: Tensor, *, causal: bool = True, position_offset: int = 0
    ) -> tuple[Tensor, KVCache]:
        q, k, v = self._project(x)
        if self.use_rope:
            q = _apply_rope(q, position_offset, self.rope_base)
            k = _apply_rope(k, position_offset, self.rope_base)
        y = self.out_proj(_grouped_attention(q, k, v, causal=causal).transpose(1, 2).reshape_as(x))
        return y, KVCache(k, v, next_position=position_offset + x.size(1))

    def decode_step(self, x: Tensor, cache: KVCache) -> tuple[Tensor, KVCache]:
        if x.size(1) != 1:
            raise ValueError("decode_step expects exactly one new token")
        position = cache.next_position
        q, k, v = self._project(x)
        if self.use_rope:
            # Cache K after its position rotation; V remains position-unrotated.
            q = _apply_rope(q, position, self.rope_base)
            k = _apply_rope(k, position, self.rope_base)
        cache = KVCache(
            torch.cat((cache.key, k), 2),
            torch.cat((cache.value, v), 2),
            next_position=position + x.size(1),
        )
        y = self.out_proj(_grouped_attention(q, cache.key, cache.value, causal=True).transpose(1, 2).reshape_as(x))
        return y, cache


# MQA is GQA with one KV head; keep a named class for clarity.
class MultiQueryAttention(GroupedQueryAttention):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        *,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
    ):
        super().__init__(
            d_model,
            num_query_heads,
            num_kv_heads=1,
            use_rope=use_rope,
            rope_base=rope_base,
        )


def _measure_ms(fn, device: torch.device, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(3):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            torch.cuda.synchronize(device)
            return start.elapsed_time(end) / iterations
        start_time = time.perf_counter()
        for _ in range(iterations):
            fn()
        return (time.perf_counter() - start_time) * 1000 / iterations


def benchmark(device: torch.device, batch: int, seq_len: int, d_model: int, heads: int, iterations: int):
    if d_model % heads:
        raise ValueError("d_model must be divisible by heads")
    models = [
        ("Single head", SingleHeadAttention(d_model)),
        ("Legacy MHA", LegacyMultiHeadAttention(d_model, heads)),
        ("MHA", MultiHeadAttention(d_model, heads)),
        ("GQA (4 KV)", GroupedQueryAttention(d_model, heads, min(4, heads))),
        ("MQA (1 KV)", MultiQueryAttention(d_model, heads)),
    ]
    dtype_bytes = torch.empty((), dtype=torch.float32).element_size()
    x = torch.randn(batch, seq_len, d_model, device=device)
    next_x = torch.randn(batch, 1, d_model, device=device)
    print(f"device={device}; batch={batch}; prefill_tokens={seq_len}; d_model={d_model}; query_heads={heads}; dtype=float32")
    print("Times are per call; theoretical KV cache is for one layer, excluding temporary activations and parameters.")
    print(f"{'variant':<18} {'params(M)':>10} {'KV cache(MiB)':>14} {'prefill(ms)':>13} {'decode(ms)':>12}")
    for name, model in models:
        model = model.to(device).eval()
        with torch.inference_mode():
            _, cache = model.prefill(x)
        kv_bytes = 2 * batch * seq_len * model.num_kv_heads * model.head_dim * dtype_bytes
        prefill_ms = _measure_ms(lambda: model(x), device, iterations)
        decode_ms = _measure_ms(lambda: model.decode_step(next_x, cache), device, iterations)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name:<18} {params:>10.3f} {kv_bytes / (1024 ** 2):>14.2f} {prefill_ms:>13.3f} {decode_ms:>12.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="run a small inference timing comparison")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not args.benchmark:
        parser.print_help()
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch installation cannot see a CUDA device")
    benchmark(device, args.batch, args.seq_len, args.d_model, args.heads, args.iterations)


if __name__ == "__main__":
    main()
