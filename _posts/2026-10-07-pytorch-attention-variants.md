---
title: "Single-head, multi-head, grouped-query, and multi-query attention in PyTorch"
title_ko: "PyTorch로 직접 구현한 Single-Head, Multi-Head, Grouped-Query, Multi-Query Attention"
english_translation: "attention-variants.en.md"
excerpt: "Reference PyTorch implementations of attention variants, with KV cache capacity, RoPE, and CPU prefill/decode measurements."
permalink: /posts/pytorch-attention-variants/
date: 2026-10-07
eyebrow: "FIELD NOTE / DEEP LEARNING"
dek: "Implementing attention variants in plain PyTorch and comparing KV cache size, RoPE behavior, and CPU inference under the same conditions."
dek_ko: "바닐라 PyTorch로 attention 변형을 구현하고, KV 캐시 크기와 CPU 추론 시간을 같은 조건에서 비교한다."
read_time: true
comments: false
share: false
related: false
---

Transformer 어텐션에는 Query, Key, Value를 여러 개의 독립된 head로 나누는 Multi-Head Attention(MHA)이 널리 쓰인다. 이후 토큰을 하나씩 생성하는 추론에서 Key와 Value를 반복해서 읽는 비용을 낮추기 위해 Multi-Query Attention(MQA)이 나왔고, 둘 사이의 선택지를 제공하는 Grouped-Query Attention(GQA)이 제안됐다. MQA는 모든 query head가 한 쌍의 K/V head를 공유하고, GQA는 query head를 여러 그룹으로 나눠 그룹마다 K/V를 공유한다. [Transformer 원 논문](https://arxiv.org/abs/1706.03762), [MQA 논문](https://arxiv.org/abs/1911.02150), [GQA 논문](https://arxiv.org/abs/2305.13245)

여기서 “멀티 그루브 어텐션”은 통용되는 표준 명칭으로 확인하기 어려워, 문맥상 **Grouped-Query Attention (GQA)**을 뜻하는 것으로 해석했다. 아래 코드는 라이브러리의 attention 모듈 대신 `Linear`, 행렬 곱, `softmax`, `einsum`으로 구성한 학습용 기준 구현이다.

## 이전 방식: Single-Head와 head별 MHA

Multi-Head Attention보다 앞선 기본형은 하나의 Query/Key/Value 투영으로 계산하는 **Single-Head Scaled Dot-Product Attention**이다. “이전 버전”을 구현 코드의 역사적 형태로도 볼 수 있도록, 각 head가 별도의 `Linear` 층을 갖는 구식 MHA도 함께 넣었다. head마다 따로 계산해 결과를 이어 붙이는 방식은 이해하기 쉽지만, 현대 구현은 보통 Q/K/V 투영을 하나의 큰 선형층으로 묶어 계산한다.

전체 구현은 [`scripts/attention_variants.py`](https://github.com/SangbumChoi/sangbumchoi.github.io/blob/master/scripts/attention_variants.py)에 있다. 표준 MHA의 핵심은 다음과 같다.

```python
q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
q = q.view(batch, length, heads, head_dim).transpose(1, 2)
k = k.view(batch, length, heads, head_dim).transpose(1, 2)
v = v.view(batch, length, heads, head_dim).transpose(1, 2)

scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
context = scores.softmax(dim=-1) @ v
```

인과 마스크는 현재 토큰이 미래 위치를 보지 못하게 한다. 실제 Transformer 블록에는 보통 이 attention 앞뒤로 정규화, residual 연결, MLP가 더해진다. 이 글의 비교에서는 attention 연산 자체를 분리해서 다룬다.

## GQA와 MQA: K/V head를 공유하기

GQA는 query head 수와 K/V head 수를 따로 받는다. 예를 들어 query head 8개와 K/V head 4개라면, query 두 개가 K/V 한 쌍을 공유한다. K/V head를 query head 수만큼 복제하지 않고 그룹 차원으로 직접 곱하므로, 코드가 저장하는 캐시도 공유 전 크기를 유지한다.

```python
q = q.reshape(batch, kv_heads, group_size, q_len, head_dim)
scores = torch.einsum("bghqd,bgkd->bghqk", q, k) / math.sqrt(head_dim)
probs = scores.softmax(dim=-1)
out = torch.einsum("bghqk,bgkd->bghqd", probs, v)
```

MQA는 같은 구현에서 `kv_heads=1`인 특수한 경우다. 따라서 K/V 캐시가 query head 8개 기준에서 1/8로 줄어든다. GQA에서 K/V head 수를 query head 수와 같게 놓으면 계산은 MHA와 동등하다.

## KV 캐시를 쓰는 단일 토큰 디코드와 RoPE

구현의 `prefill`은 프롬프트 전체의 K/V를 한 번 계산해 캐시에 넣는다. 이후 `decode_step`에는 새 토큰 하나만 넣는다. 새 토큰의 Q/K/V 투영은 매번 계산하지만, attention은 새 K/V를 기존 캐시에 이어 붙인 뒤 수행한다. 따라서 과거 토큰의 K/V 투영은 반복하지 않는다.

```python
attn = GroupedQueryAttention(
    d_model=512, num_query_heads=8, num_kv_heads=4, use_rope=True
)

# prompt_hidden: [batch, prompt_length, d_model]
prompt_out, cache = attn.prefill(prompt_hidden)

# next_hidden: [batch, 1, d_model]
token_out, cache = attn.decode_step(next_hidden, cache)
```

RoPE는 Q와 K에 선형 투영을 적용한 직후, attention 점수 `QKᵀ`를 계산하기 전에 적용한다. K는 **회전된 상태로** 캐시에 저장하고, 다음 단계의 새 Q와 K에는 다음 논리 위치의 회전을 적용한다. V는 회전하지 않는다. 표준 RoPE는 위치 정보를 Q와 K의 내적에 반영하므로, attention 점수는 위치 정보를 갖고 가중합에 쓰이는 V는 원래 표현을 유지한다.

이 예제는 `cache.next_position`으로 다음 논리 위치를 추적한다. 실제 배포용 rolling buffer에서는 물리 슬롯 번호와 논리 토큰 위치가 다를 수 있다. 버퍼가 감겼다고 RoPE 위치를 슬롯 번호로 되돌리면 상대 위치가 틀어지므로, 위치 번호를 일관되게 이어가거나 K의 RoPE 회전을 새 위치 체계에 맞게 바꿔야 한다. 이 구현은 이해를 위한 예제로, `torch.cat`이 매번 캐시를 복사하므로 실제 추론 엔진처럼 사전 할당된 KV 버퍼를 사용하지는 않는다.

## KV 캐시 용량

자동 회귀 생성은 이전 토큰의 K/V를 캐시해 재계산을 피한다. 한 레이어에서 캐시에 필요한 바이트 수는 다음과 같다.

```text
KV bytes = 2 × batch × sequence_length × num_kv_heads × head_dim × bytes_per_element
```

여기서 앞의 2는 Key와 Value를 뜻한다. 아래는 `batch=2`, 문맥 길이 256, `d_model=512`, query head 8개, FP32로 계산한 **한 레이어**의 캐시다. 모델 전체 캐시는 레이어 수만큼 늘어난다.

| 방식 | Query head | K/V head | KV 캐시 |
| --- | ---: | ---: | ---: |
| Single-Head | 1 | 1 | 2.00 MiB |
| Legacy MHA | 8 | 8 | 2.00 MiB |
| MHA | 8 | 8 | 2.00 MiB |
| GQA | 8 | 4 | 1.00 MiB |
| MQA | 8 | 1 | 0.25 MiB |

Single-Head는 head 수가 하나지만 head 차원이 `d_model` 전체이므로 이 설정의 K/V 총 원소 수는 MHA와 같다. BF16이나 FP16은 원소당 바이트 수가 절반이어서 캐시도 절반이다. 표는 캐시만 계산하며 모델 가중치, 임시 활성값, allocator 여유분은 포함하지 않는다.

## CPU 추론 시간 비교

같은 환경에서 스크립트를 실행하면 prefill(문맥 전체 처리)과 decode(캐시가 있을 때 다음 토큰 하나 처리) 시간을 확인할 수 있다.

```bash
python scripts/attention_variants.py --benchmark
```

기본값은 CPU에서 `batch=2`, `sequence_length=256`, `d_model=512`, query head 8개, FP32다. 현재 작업 환경의 한 번의 실행 결과는 다음과 같다.

| 방식 | 파라미터 수 | KV 캐시 | Prefill | Decode 1토큰 |
| --- | ---: | ---: | ---: | ---: |
| Single-Head | 1.051M | 2.00 MiB | 2.514 ms | 0.395 ms |
| Legacy MHA | 1.051M | 2.00 MiB | 5.059 ms | 0.716 ms |
| MHA | 1.051M | 2.00 MiB | 4.574 ms | 0.786 ms |
| GQA (4 K/V heads) | 0.788M | 1.00 MiB | 3.936 ms | 0.480 ms |
| MQA (1 K/V head) | 0.591M | 0.25 MiB | 3.454 ms | 0.380 ms |

여기서 시간은 바닐라 PyTorch 구현의 실행 예시이지, 각 구조의 보장된 상대 속도는 아니다. 이 코드는 최적화된 커널이 아니고, CPU, GPU, PyTorch 버전, dtype, 배치 크기에 따라 순위와 차이가 바뀐다. 특히 GQA/MQA의 이점은 보통 캐시를 반복해서 읽는 긴 문맥의 GPU 토큰 생성에서 중요하다. 기준 구현은 head를 묶어 읽는 fused CUDA 커널이 아니므로 그 이점을 온전히 측정하지 못한다. 실제 배포 속도를 판단하려면 목표 하드웨어와 최적화된 런타임에서 측정해야 한다.

Prefill의 attention 계산량은 세 방식 모두 대략 `O(batch × sequence_length² × d_model)`로 남는다. GQA/MQA는 K/V 투영과 캐시 크기를 줄이지만 query head의 계산은 유지하기 때문이다. 그래서 KV 캐시 메모리 감소 폭만큼 prefill 시간이 빨라지지는 않는다. GQA는 MHA 품질과 MQA 속도 사이를 조절할 수 있지만, head 공유가 모델 품질에 미치는 영향은 학습 조건과 모델에 따라 확인해야 한다.

## 요약

- **Single-Head**는 attention 원리를 설명하기 좋은 기준형이다.
- **Legacy MHA**는 head별 투영을 따로 호출한다. 현대의 packed 투영보다 일반적으로 느리지만 동작을 이해하기 쉽다.
- **MHA**는 query, key, value 각각의 head가 독립된 투영을 갖고, KV 캐시를 가장 많이 사용한다.
- **GQA**는 여러 query head가 K/V를 그룹 단위로 공유해 캐시를 절감한다.
- **MQA**는 모든 query head가 K/V 한 쌍을 공유해 캐시를 가장 작게 만든다.

세 변형의 차이는 주로 K/V head 수다. 실제 서비스에서의 선택은 품질 검증, KV 캐시 한도, 목표 GPU의 커널 지원을 함께 보고 결정해야 한다.
