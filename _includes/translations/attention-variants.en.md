Multi-Head Attention (MHA) splits queries, keys, and values into several independent heads. During token-by-token generation, repeatedly reading cached keys and values can become expensive. Multi-Query Attention (MQA) reduces that cost by sharing a single pair of K/V heads across all query heads. Grouped-Query Attention (GQA) provides an intermediate option: groups of query heads share K/V heads. [Transformer paper](https://arxiv.org/abs/1706.03762), [MQA paper](https://arxiv.org/abs/1911.02150), [GQA paper](https://arxiv.org/abs/2305.13245)

The phrase "multi-groove attention" is not an established name I could identify, so I interpreted it as **Grouped-Query Attention (GQA)** in this context. The examples below are educational reference implementations built from `Linear`, matrix multiplication, `softmax`, and `einsum`, rather than a library attention module.

## Earlier forms: single-head attention and per-head MHA

The basic starting point is **single-head scaled dot-product attention**, using one query, key, and value projection. To illustrate an earlier implementation style as well, I included legacy MHA with a separate `Linear` layer for each head. Computing each head separately and concatenating the results is easy to follow. Modern implementations usually combine the Q/K/V projections into larger linear operations.

The complete implementation is in [`scripts/attention_variants.py`](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/attention_variants.py). The core of standard MHA is:

```python
q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
q = q.view(batch, length, heads, head_dim).transpose(1, 2)
k = k.view(batch, length, heads, head_dim).transpose(1, 2)
v = v.view(batch, length, heads, head_dim).transpose(1, 2)

scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
context = scores.softmax(dim=-1) @ v
```

The causal mask prevents a token from attending to future positions. A full Transformer block also normally contains normalization, residual connections, and an MLP. This comparison isolates the attention operation itself.

## GQA and MQA: sharing K/V heads

GQA specifies the number of query heads separately from the number of K/V heads. With eight query heads and four K/V heads, each pair of query heads shares one K/V pair. The implementation multiplies over the group dimension directly, without expanding K/V to the number of query heads. Its stored cache therefore retains the smaller shared shape.

```python
q = q.reshape(batch, kv_heads, group_size, q_len, head_dim)
scores = torch.einsum("bghqd,bgkd->bghqk", q, k) / math.sqrt(head_dim)
probs = scores.softmax(dim=-1)
out = torch.einsum("bghqk,bgkd->bghqd", probs, v)
```

MQA is the special case with `kv_heads=1`. Relative to eight query heads, its K/V cache is one-eighth the size. Setting the number of K/V heads equal to the number of query heads makes GQA equivalent to MHA for this computation.

## Single-token decoding with a KV cache and RoPE

The implementation's `prefill` method computes K/V for the whole prompt and stores them in the cache. Each subsequent `decode_step` receives one new token. It computes that token's Q/K/V projections, appends the new K/V to the existing cache, and performs attention. It does not recompute the K/V projections for previous tokens.

```python
attn = GroupedQueryAttention(
    d_model=512, num_query_heads=8, num_kv_heads=4, use_rope=True
)

# prompt_hidden: [batch, prompt_length, d_model]
prompt_out, cache = attn.prefill(prompt_hidden)

# next_hidden: [batch, 1, d_model]
token_out, cache = attn.decode_step(next_hidden, cache)
```

RoPE is applied to Q and K after their linear projections and before computing attention scores. The cache stores K **after rotation**. The next step applies rotations corresponding to the next logical position to the new Q and K. V is not rotated: standard RoPE introduces position information through the Q/K dot product while preserving the value representation used in the weighted sum.

This example tracks the next logical position with `cache.next_position`. In a production rolling buffer, physical slots and logical token positions may differ. Resetting the RoPE position to the physical slot when the buffer wraps would produce incorrect relative positions. Positions must remain consistent, or the cached K rotations must be adjusted to a new position scheme. This educational implementation uses `torch.cat`, which copies the cache on every step; it does not use a preallocated KV buffer like a production inference engine.

## KV cache capacity

Autoregressive generation caches previous keys and values to avoid recomputing them. For one layer:

```text
KV bytes = 2 × batch × sequence_length × num_kv_heads × head_dim × bytes_per_element
```

The leading 2 accounts for keys and values. The table below uses `batch=2`, context length 256, `d_model=512`, eight query heads where applicable, and FP32. These are **single-layer** cache sizes; the full model cache grows with the number of layers.

| Method | Query heads | K/V heads | KV cache |
| --- | ---: | ---: | ---: |
| Single-head | 1 | 1 | 2.00 MiB |
| Legacy MHA | 8 | 8 | 2.00 MiB |
| MHA | 8 | 8 | 2.00 MiB |
| GQA | 8 | 4 | 1.00 MiB |
| MQA | 8 | 1 | 0.25 MiB |

Single-head attention uses only one head, but that head spans the full model dimension. It therefore stores as many K/V elements as MHA in this configuration. BF16 or FP16 halves the bytes per element and the cache size. These figures exclude model weights, temporary activations, and allocator overhead.

## Comparing CPU inference times

The script measures prefill over the whole context and decoding of one new token with an existing cache:

```bash
python scripts/attention_variants.py --benchmark
```

The defaults are CPU execution, `batch=2`, `sequence_length=256`, `d_model=512`, eight query heads, and FP32. One run in the working environment produced:

| Method | Parameters | KV cache | Prefill | Decode one token |
| --- | ---: | ---: | ---: | ---: |
| Single-head | 1.051M | 2.00 MiB | 2.514 ms | 0.395 ms |
| Legacy MHA | 1.051M | 2.00 MiB | 5.059 ms | 0.716 ms |
| MHA | 1.051M | 2.00 MiB | 4.574 ms | 0.786 ms |
| GQA (4 K/V heads) | 0.788M | 1.00 MiB | 3.936 ms | 0.480 ms |
| MQA (1 K/V head) | 0.591M | 0.25 MiB | 3.454 ms | 0.380 ms |

These timings illustrate one run of a plain PyTorch implementation. They do not establish a guaranteed speed ranking. This is not an optimized kernel, and results depend on the CPU or GPU, PyTorch version, dtype, and batch size. GQA and MQA are especially useful when GPU token generation repeatedly reads a long-context cache. This reference implementation is not a fused CUDA kernel that reads shared heads together, so it does not fully measure that benefit. Deployment decisions require measurements on the target hardware and optimized runtime.

Prefill attention computation remains approximately `O(batch × sequence_length² × d_model)` for MHA, GQA, and MQA. Sharing K/V reduces projection work and cache size, but preserves the query-head computation. Prefill time therefore does not decrease in proportion to cache memory. GQA can offer a point between MHA quality and MQA efficiency, but the effect of head sharing on quality must be evaluated for the model and training setup.

## What each variant changes

- **Single-head attention** is a useful starting point for explaining the computation.
- **Legacy MHA** calls separate per-head projections. It is usually slower than packed projections, but makes the mechanism easier to follow.
- **MHA** gives query, key, and value heads independent projections and uses the largest KV cache among the multi-head variants compared here.
- **GQA** shares K/V within groups of query heads to reduce the cache.
- **MQA** shares one K/V pair across all query heads, giving the smallest cache.

The main distinction is the number of K/V heads. A production choice should account for model quality, cache limits, and kernel support on the target GPU.
