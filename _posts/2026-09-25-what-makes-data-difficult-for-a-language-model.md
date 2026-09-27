---
title: "What makes data difficult for a language model?"
title_ko: "언어 모델에게 데이터가 어렵다는 것은 무엇일까?"
permalink: /posts/what-makes-data-difficult-for-a-language-model/
date: 2026-09-25
last_modified_at: 2026-09-25
eyebrow: "FIELD NOTE / PRE-TRAINING"
dek: "My first small pre-training run showed code and algebra converging far below web text. Instead of treating that as a conclusion, I want to turn it into a better question: how should we define difficulty for next-token prediction?"
dek_ko: "처음 돌려본 작은 pre-training run에서 code와 algebra는 web text보다 훨씬 낮은 loss로 수렴했습니다. 이를 결론으로 받아들이기보다 더 나은 질문으로 바꿔보고 싶습니다. next-token prediction에서 difficulty는 어떻게 정의해야 할까요?"
read_time: true
comments: false
share: false
related: false
---

<div class="lang-block" lang="en" markdown="1">

I recently pre-trained a small language model from scratch for the first time.

The model was trained on a mixture of data sources. Besides the overall loss, I
tracked the loss of each source separately, and one pattern was hard to miss:
structured, repetitive domains such as code and algebra converged quickly to
low loss, while heterogeneous web text stayed noticeably higher.

![Per-source pre-training loss curves and loss-token counts for a small model]({{ '/assets/images/posts/what-makes-data-difficult-for-a-language-model/loss-by-source.png' | relative_url }})
*Left: training loss per source (200-step running mean) over 16k steps of my small pre-training run; the dashed line is the mixed loss a single training curve would report. Right: how many loss tokens each source contributed, which is not the same as its input mixture share.*

A few things stand out in this first run:

- **Code and algebra go lowest.** `math_algebraic_stack` ends near 1.2 and
  `code_github_code_clean` near 1.3, and both drop below 2 within roughly the
  first 1,500 steps.
- **Web text stays highest.** The three web sources, which together make up
  about 62% of loss tokens, plateau somewhere around 3.1-3.4.
- **"Math" is not one difficulty.** `math_openwebmath`, which is math written
  as web prose, stays near 2.7, far above `math_algebraic_stack`. Same topic,
  very different loss, which already hints that format matters more than
  subject.
- **Templated data is easy too.** `know_flan`, built from instruction
  templates, sits near 1.4, close to code.
- **The mixed loss hides all of this.** The single dashed curve (about 2.8)
  is dominated by web text simply because web contributes most of the tokens.

These are training-loss curves from one run, not held-out evaluations, so I
treat them as a hint rather than a measurement.

It is tempting to read this as "the model is good at code and bad at web text."
I do not think that reading is justified yet. This post is a note on why, and
on the experiment I want to run next.

## Human difficulty is not predictive difficulty

What is difficult for a person and what is difficult for next-token prediction
can be very different things.

Math and code are often conceptually hard for us. But their syntax and regular
structure constrain what can come next. After `for i in range(`, the plausible
continuations are few. A closing bracket, an indentation level, or the next
step of a routine derivation is often nearly determined by the context.

Ordinary web text looks easy to read, yet it leaves many more doors open. After
"I went to the store and", a large number of continuations are reasonable, and
no amount of modeling skill can make that choice certain.

So a low loss on code may say less about how well the model *understands* code
and more about how *narrow* the next-token distribution of code is.

## Loss mixes two different things

A small identity makes this concrete. For a context $$x$$, let $$p(\cdot \mid x)$$
be the true next-token distribution and $$q(\cdot \mid x)$$ the model's
prediction. The expected cross-entropy loss splits into two parts:

$$
\underbrace{\mathbb{E}_{y \sim p}\left[-\log q(y \mid x)\right]}_{\text{loss we measure}}
= \underbrace{H\big(p(\cdot \mid x)\big)}_{\text{irreducible entropy}}
+ \underbrace{D_{\mathrm{KL}}\big(p(\cdot \mid x)\,\|\,q(\cdot \mid x)\big)}_{\text{what the model has not learned yet}}
$$

The first term is a property of the data. The second term is the gap that
training can close. A per-domain loss curve reports only their sum.

> **In simple words - Entropy vs. loss:** Entropy is how uncertain the next token *really* is. Loss is how surprised the model *was*. A domain can have high loss simply because its next tokens are genuinely unpredictable, even when the model has learned almost everything that can be learned from it.
{: .notice--info}

Under that view, "code converged lower than web text" has at least two
explanations that a raw loss curve cannot separate:

1. code has lower intrinsic entropy, so its floor is lower; or
2. the model genuinely closed more of the learnable gap on code.

Both may be true at once. The interesting quantity is the second one, and it is
the one I have not measured.

## Existing work points in the same direction

This is not a new concern, and several lines of work already treat it
seriously.

- **DoReMi** ([Xie et al., 2023](https://arxiv.org/abs/2305.10429)) optimizes
  domain weights with *excess loss*: the loss of a proxy model minus the loss of
  a reference model on the same domain. Subtracting the reference roughly
  removes each domain's intrinsic difficulty, so the optimizer attends to
  domains where there is still something to learn instead of domains that are
  simply noisy.
- **Rho-1** ([Lin et al., 2024](https://arxiv.org/abs/2404.07965)) looks at loss
  at the token level and finds that tokens follow very different trajectories
  during training: some are learned early, some stay hard, and some fluctuate.
  Averaging them into one number hides most of that structure.
- **Entropy-Guided Token Dropout**
  ([arXiv:2512.23422](https://arxiv.org/abs/2512.23422)) reports that, in
  multi-epoch training on limited domain data, low-entropy tokens are learned
  quickly and come to dominate optimization, while generalization on
  high-entropy tokens degrades with continued training.
- **Data Mixing Laws** ([Ye et al., 2024](https://arxiv.org/abs/2403.16952))
  show that domain-level loss can be predicted as a function of the mixture
  proportions, which suggests these differences are systematic enough to model
  rather than noise.

Together they suggest that "difficulty" is better defined relative to what is
learnable, and better measured at the token level than at the domain level.

## The next experiment

Instead of comparing raw loss across domains, I want to measure the
relationship between **token-level predictability** and **how fast each token
is learned**.

### 1. Estimate predictability with a reference model

The true entropy $$H(p)$$ is not observable. As a proxy, I will score a held-out
set with a larger, well-trained reference model and record, for every token:

- **predictive entropy**: the entropy of the reference model's full next-token
  distribution, $$H(q_{\text{ref}}(\cdot \mid x))$$, which says how open the
  context is;
- **reference surprisal**: $$-\log q_{\text{ref}}(y \mid x)$$ for the actual
  token, which says how surprising that particular token was.

These answer different questions. A context can be open (high entropy) while
the observed token happens to be the most likely one, and vice versa.

### 2. Bucket tokens within each domain

Comparing domains mixes entropy with everything else that differs between
them. So I will split tokens into entropy quantiles *within* each domain: the
lowest-entropy tokens of web text are compared with the highest-entropy tokens
of web text, and the same for code and algebra.

### 3. Track learning speed per bucket

For every saved checkpoint of my own small model, I will compute the mean loss
in each bucket and summarize each curve with:

- **time to converge**: the number of training tokens needed to close, say, 90%
  of the gap between the initial and final bucket loss;
- **excess loss**: the small model's loss minus the reference model's loss,
  following the DoReMi idea, at every checkpoint.

If difficulty is mostly entropy, the curves should line up by entropy bucket
regardless of domain. If they do not, something about the domain itself is
doing work that entropy does not explain.

### 4. Controls I need to watch

- **Tokenization:** code and math are tokenized very differently from prose, and
  a "token" may carry different amounts of information in each domain.
- **Duplication and boilerplate:** licenses, imports, and templated pages can
  produce near-zero loss by memorization rather than learning.
- **Position in the sequence:** tokens late in a long context are usually
  easier, and domains differ in document length.
- **Reference-model bias:** the reference model has its own training mixture, so
  its entropy estimates are not a neutral ground truth.

## Questions I want the data to answer

- Within a single domain, are low-entropy tokens learned faster, and by how
  much?
- Does the gap in learning speed between domains shrink once entropy is held
  fixed?
- Is there a group of high-entropy tokens whose excess loss keeps falling late
  in training? Those may be where the "hard" learning actually happens.
- Would a mixture chosen by excess loss rather than raw loss change what my
  small model ends up being good at?

I do not have answers yet. The per-domain curve was only the first hint that
"difficulty" needs a sharper definition.

It is striking that pre-training has become accessible enough for an
individual to run a model from scratch and then ask questions like these with a
relatively small experiment. I will follow up with the results.

## Key terms

- **Per-domain loss:** the average next-token loss measured on data from a
  single source.
- **Entropy:** how spread out the next-token distribution is; the part of the
  loss no model can remove.
- **Surprisal:** $$-\log q(y \mid x)$$ for the token that actually appeared.
- **Excess loss:** a model's loss minus a reference model's loss on the same
  data, used as a proxy for what is still learnable.
- **Data mixture:** the proportion of each source in the pre-training corpus.

</div>

<div class="lang-block" lang="ko" markdown="1">

처음으로 작은 언어 모델을 처음부터 pre-training 해봤습니다.

여러 데이터 소스를 섞어 학습하면서 전체 loss뿐 아니라 source별 loss를 따로
추적해봤는데, 꽤 뚜렷한 차이가 보였습니다. Code나 algebra처럼 구조적이고
반복적인 패턴이 있는 데이터는 빠르게 낮은 loss로 수렴한 반면, 이질적인 web
text는 눈에 띄게 높은 loss를 유지했습니다.

![작은 모델의 source별 pre-training loss 곡선과 loss token 수]({{ '/assets/images/posts/what-makes-data-difficult-for-a-language-model/loss-by-source.png' | relative_url }})
*왼쪽: 작은 pre-training run의 16k step 동안 source별 training loss(200-step 이동 평균). 점선은 하나의 training curve가 보여주는 mixed loss입니다. 오른쪽: 각 source가 기여한 loss token 수로, 입력 mixture 비율과는 다릅니다.*

이번 첫 run에서 눈에 띄는 점이 몇 가지 있습니다.

- **Code와 algebra가 가장 낮게 내려갑니다.** `math_algebraic_stack`은 약 1.2,
  `code_github_code_clean`은 약 1.3에서 끝나며, 둘 다 대략 처음 1,500 step
  안에 2 아래로 떨어집니다.
- **Web text는 가장 높게 머뭅니다.** loss token의 약 62%를 차지하는 세 개의
  web source는 3.1-3.4 근처에서 정체됩니다.
- **"수학"이 하나의 난이도는 아닙니다.** web 문서 형태로 쓰인 수학인
  `math_openwebmath`는 약 2.7에 머물러 `math_algebraic_stack`보다 훨씬 높습니다.
  같은 주제인데 loss는 크게 다르다는 점에서, 이미 주제보다 **형식**이 더
  중요하다는 힌트를 줍니다.
- **템플릿 기반 데이터도 쉽습니다.** instruction 템플릿으로 만든 `know_flan`은
  1.4 근처로 code와 비슷합니다.
- **Mixed loss는 이 모든 것을 가립니다.** 하나의 점선 곡선(약 2.8)은 단지 web이
  가장 많은 token을 차지한다는 이유로 web text에 의해 좌우됩니다.

이는 held-out 평가가 아니라 한 번의 run에서 나온 training loss 곡선이므로,
측정값이 아니라 힌트로 보고 있습니다.

이를 "모델이 code는 잘하고 web text는 못한다"로 읽고 싶어지지만, 아직 그렇게
해석할 근거는 없다고 생각합니다. 이 글은 그 이유와, 다음에 해보려는 실험에 대한
기록입니다.

## 사람에게 어려운 것과 예측이 어려운 것은 다르다

사람에게 어려운 내용과 next-token prediction 관점에서 어려운 데이터는 전혀
다를 수 있습니다.

수학이나 코드는 개념적으로 사람에게 어려운 경우가 많습니다. 하지만 syntax와
규칙적인 structure가 다음에 올 수 있는 것을 제약합니다. `for i in range(` 뒤에
올 수 있는 그럴듯한 continuation은 몇 개 되지 않습니다. 닫는 괄호, indentation
수준, 혹은 정형화된 유도 과정의 다음 단계는 context에 의해 거의 결정되는
경우가 많습니다.

평범한 web text는 읽기에는 쉬워 보이지만, 훨씬 많은 가능성을 열어 둡니다.
"I went to the store and" 뒤에는 합리적인 continuation이 매우 많고, 모델링
능력이 아무리 뛰어나도 그 선택을 확실하게 만들 수는 없습니다.

따라서 code에서 loss가 낮다는 것은 모델이 code를 얼마나 잘 *이해*하는지보다,
code의 next-token distribution이 얼마나 *좁은지*를 더 많이 말해주는 것일 수
있습니다.

## Loss는 서로 다른 두 가지를 섞고 있다

간단한 항등식 하나로 이를 구체적으로 볼 수 있습니다. context $$x$$에 대해
$$p(\cdot \mid x)$$를 실제 next-token distribution, $$q(\cdot \mid x)$$를 모델의
예측이라고 하겠습니다. 기대 cross-entropy loss는 두 부분으로 나뉩니다.

$$
\underbrace{\mathbb{E}_{y \sim p}\left[-\log q(y \mid x)\right]}_{\text{loss we measure}}
= \underbrace{H\big(p(\cdot \mid x)\big)}_{\text{irreducible entropy}}
+ \underbrace{D_{\mathrm{KL}}\big(p(\cdot \mid x)\,\|\,q(\cdot \mid x)\big)}_{\text{what the model has not learned yet}}
$$

첫 번째 항은 데이터 자체의 성질입니다. 두 번째 항은 학습으로 줄일 수 있는
gap입니다. domain별 loss 곡선은 이 둘의 합만을 보여줍니다.

> **쉽게 말하면 - Entropy vs. loss:** Entropy는 다음 token이 *실제로* 얼마나 불확실한지입니다. Loss는 모델이 얼마나 *놀랐는지*입니다. 어떤 domain은 모델이 그 데이터에서 배울 수 있는 것을 거의 다 배웠더라도, 다음 token이 본질적으로 예측 불가능하다는 이유만으로 loss가 높을 수 있습니다.
{: .notice--info}

이 관점에서 보면 "code가 web text보다 낮게 수렴했다"는 사실에는 raw loss
곡선만으로는 구분할 수 없는 설명이 적어도 두 가지 있습니다.

1. code의 intrinsic entropy가 더 낮아서 바닥 자체가 더 낮거나,
2. 모델이 실제로 code에서 학습 가능한 gap을 더 많이 줄였거나.

둘 다 동시에 참일 수도 있습니다. 흥미로운 것은 두 번째 양이고, 제가 아직
측정하지 않은 것도 바로 그것입니다.

## 기존 연구도 같은 방향을 가리킨다

새로운 문제의식은 아니며, 이미 여러 연구 흐름이 이를 진지하게 다루고 있습니다.

- **DoReMi** ([Xie et al., 2023](https://arxiv.org/abs/2305.10429))는 *excess
  loss*, 즉 같은 domain에서 proxy model의 loss에서 reference model의 loss를 뺀
  값으로 domain weight를 최적화합니다. reference를 빼면 각 domain의 intrinsic
  difficulty가 대략 제거되므로, optimizer는 단지 noisy한 domain이 아니라 아직
  배울 것이 남아 있는 domain에 집중하게 됩니다.
- **Rho-1** ([Lin et al., 2024](https://arxiv.org/abs/2404.07965))은 token
  수준에서 loss를 들여다보고, 학습 중 token마다 궤적이 크게 다르다는 것을
  발견합니다. 어떤 token은 일찍 학습되고, 어떤 token은 계속 어렵고, 어떤 token은
  요동칩니다. 이를 하나의 숫자로 평균 내면 그 구조 대부분이 가려집니다.
- **Entropy-Guided Token Dropout**
  ([arXiv:2512.23422](https://arxiv.org/abs/2512.23422))은 제한된 domain
  데이터로 여러 epoch 학습할 때, low-entropy token이 빠르게 학습되어
  optimization을 지배하게 되고, 학습이 계속될수록 high-entropy token에 대한
  generalization은 나빠진다고 보고합니다.
- **Data Mixing Laws** ([Ye et al., 2024](https://arxiv.org/abs/2403.16952))는
  domain 수준의 loss를 mixture 비율의 함수로 예측할 수 있음을 보여주며, 이는
  이런 차이가 noise가 아니라 모델링할 수 있을 만큼 체계적이라는 것을
  시사합니다.

종합하면, "difficulty"는 학습 가능한 부분을 기준으로 정의하는 것이 더 낫고,
domain 수준보다는 token 수준에서 측정하는 것이 더 낫다는 것을 시사합니다.

## 다음 실험

domain 간 raw loss를 비교하는 대신, **token-level predictability**와 **각
token이 얼마나 빨리 학습되는지** 사이의 관계를 측정해보려고 합니다.

### 1. Reference model로 predictability 추정하기

실제 entropy $$H(p)$$는 관측할 수 없습니다. 그 proxy로, 더 크고 잘 학습된
reference model로 held-out set을 채점하고, 모든 token에 대해 다음을
기록합니다.

- **predictive entropy**: reference model의 전체 next-token distribution의
  entropy인 $$H(q_{\text{ref}}(\cdot \mid x))$$로, context가 얼마나 열려
  있는지를 말해줍니다.
- **reference surprisal**: 실제 token에 대한 $$-\log q_{\text{ref}}(y \mid x)$$로,
  그 특정 token이 얼마나 놀라웠는지를 말해줍니다.

이 둘은 서로 다른 질문에 답합니다. context는 열려 있지만(high entropy) 관측된
token이 마침 가장 확률이 높은 token일 수도 있고, 그 반대일 수도 있습니다.

### 2. 각 domain 안에서 token을 구간으로 나누기

domain끼리 비교하면 entropy와 domain 사이에 달라지는 다른 모든 요인이 함께
섞입니다. 그래서 각 domain *안에서* token을 entropy quantile로 나누려고 합니다.
web text의 가장 낮은 entropy token을 web text의 가장 높은 entropy token과
비교하고, code와 algebra에서도 마찬가지로 합니다.

### 3. 구간별 학습 속도 추적하기

제 작은 모델의 저장된 checkpoint마다 각 구간의 평균 loss를 계산하고, 각 곡선을
다음 지표로 요약합니다.

- **time to converge**: 초기 구간 loss와 최종 구간 loss 사이 gap의 예컨대 90%를
  줄이는 데 필요한 training token 수;
- **excess loss**: DoReMi의 아이디어를 따라, 매 checkpoint에서 작은 모델의
  loss에서 reference model의 loss를 뺀 값.

difficulty가 대부분 entropy로 설명된다면, 곡선들은 domain과 상관없이 entropy
구간별로 정렬되어야 합니다. 그렇지 않다면, entropy로 설명되지 않는 domain 자체의
무언가가 작용하고 있는 것입니다.

### 4. 주의해야 할 통제 변수

- **Tokenization:** code와 수학은 산문과 매우 다르게 tokenize되며, 하나의
  "token"이 담는 정보량이 domain마다 다를 수 있습니다.
- **중복과 boilerplate:** license, import, 템플릿 페이지는 학습이 아니라
  암기로 0에 가까운 loss를 만들 수 있습니다.
- **Sequence 내 위치:** 긴 context의 뒤쪽 token은 보통 더 쉽고, domain마다
  문서 길이가 다릅니다.
- **Reference model의 bias:** reference model도 자체 training mixture를 가지고
  있으므로, 그 entropy 추정치는 중립적인 ground truth가 아닙니다.

## 데이터로 답하고 싶은 질문

- 하나의 domain 안에서 low-entropy token이 더 빨리 학습될까? 그렇다면 얼마나
  더 빠를까?
- entropy를 고정하면 domain 간 학습 속도 차이는 줄어들까?
- 학습 후반까지 excess loss가 계속 줄어드는 high-entropy token 그룹이 있을까?
  그곳이 실제로 "어려운" 학습이 일어나는 곳일지도 모릅니다.
- raw loss가 아니라 excess loss로 고른 mixture를 쓰면, 제 작은 모델이 결국
  잘하게 되는 것이 달라질까?

아직 답은 없습니다. domain별 곡선은 "difficulty"에 더 명확한 정의가 필요하다는
첫 번째 힌트였을 뿐입니다.

개인이 직접 모델을 처음부터 pre-training 해보고, 비교적 작은 실험으로 이런
질문까지 던져볼 수 있을 만큼 pre-training이 접근 가능해졌다는 것이 꽤
인상적입니다. 결과가 나오면 이어서 정리해보겠습니다.

## 핵심 용어

- **Per-domain loss:** 하나의 source에서 온 데이터로 측정한 평균 next-token
  loss.
- **Entropy:** next-token distribution이 얼마나 퍼져 있는지를 나타내며, 어떤
  모델도 제거할 수 없는 loss의 부분.
- **Surprisal:** 실제로 등장한 token에 대한 $$-\log q(y \mid x)$$.
- **Excess loss:** 같은 데이터에서 모델의 loss에서 reference model의 loss를 뺀
  값으로, 아직 학습 가능한 부분의 proxy로 사용됩니다.
- **Data mixture:** pre-training corpus에서 각 source가 차지하는 비율.

</div>
