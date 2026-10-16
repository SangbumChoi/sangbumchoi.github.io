---
title: "A red pen is fast, a pass/fail grade outgrows the teacher: RLVR, OPD and MOPD on a toy model"
title_ko: "빨간 펜(OPD)은 빠르고, O/X 채점(RLVR)은 선생을 넘어요"
permalink: /posts/rlvr-opd-mopd-toy-experiment/
date: 2026-10-15
last_modified_at: 2026-10-16
eyebrow: "FIELD NOTE / POST-TRAINING"
dek: "I taught a tiny model three-digit addition and subtraction four ways under identical conditions. Per-character feedback (OPD) is fast but capped by the teacher, pass/fail rewards (RLVR) are slow but can beat the teacher, and with two subjects only per-subject teachers (MOPD) kept both. Scaled up to million-sized arithmetic with near-perfect teachers, the distillation methods tie and RL barely learns."
dek_ko: "세 자리 덧셈·뺄셈을 푸는 작은 모델을 같은 조건에서 네 가지 방법으로 가르쳐 봤어요. 글자마다 고쳐 주는 OPD 는 빠르지만 선생 실력이 천장이고, 정답 여부만 알려 주는 RLVR 은 느려도 선생을 넘으며, 과목이 둘이면 과목별 선생을 둔 MOPD 만 두 과목을 다 지켰어요. 선생이 거의 완벽한 백만 단위 사칙연산으로 키우면 증류 방법끼리는 비슷하고 RL 은 거의 못 배워요."
read_time: true
comments: false
share: false
related: false
---

<link rel="stylesheet" href="{{ '/assets/css/posts/rl-toy.css' | relative_url }}?v={{ site.jarvis_asset_version }}">
<script src="{{ '/assets/js/posts/rl-toy.js' | relative_url }}?v={{ site.jarvis_asset_version }}" defer></script>

<div class="lang-block" lang="en" markdown="1">

*A toy experiment: CPU only, about 35 minutes, three seeds per run. The full
code is [`rlsim.py`]({{ '/files/rl-toy/rlsim.py' | relative_url }}), a single
364-line file that needs only PyTorch and NumPy.*

## At a glance {#glance}

There are six experiments. Experiments 1–5 use the same small student model,
and every method writes the same 256 answers per training step. Numbers are
accuracy on 2,000 held-out evaluation problems (drawn separately from training
problems, though an overlap is possible; see [known gaps](#code-gaps)).
Experiment 6 scales up to million-sized numbers and all four operations on a
GPU.

| Experiment | Setup | Result (accuracy) | What it means |
| --- | --- | --- | --- |
| 1. Start at 34% | addition, 1,000 steps | SFT 99.9% · OPD 98.6% · RLVR 81.0% | Per-character feedback is far faster than pass/fail |
| 2. Start at 0% | addition, 1,000 steps | OPD 96.4% · RLVR 0.0% | Pass/fail only starts working once the model is sometimes right |
| 3. Weak teacher (73%) | addition, 3,000 steps | RLVR 91.8% · SFT 79.4% · OPD 78.9% | Distillation is capped by the teacher; RLVR is anchored to the truth |
| 4. Two subjects | add/subtract, 1,000 steps | MOPD 98/97% · Mix-RL 45/45% · Cascade-RL 3/7% · addition teacher only 99/0% | Each subject needs its own teacher |
| 5. RLVR from 0% | addition, 1,000–3,000 steps | true-answer SFT 100% · injected answer 70% · per-digit grading 24% · partial credit and easy-first 0% | A 0% student has to be shown the answer |
| 6. Million-scale + − × ÷ | dev GPU, 10,000 steps, 3 seeds | true-answer SFT 82.5% · SFT distillation 82.1% · MOPD 81.2% · single-teacher OPD 81.1% · MOPD → RL 78.0% · Mix-RL 65.6% · Cascade-RL 50.8% | With near-perfect teachers the distillation methods barely differ, and RL learns little because its signal is rare |

<div class="rl-callout is-decision" markdown="1">
<span class="rl-callout__title">Why modern post-training looks the way it does</span>
Build an expert teacher per subject with RL (RL can surpass a teacher), then
gather those teachers' red pens into a single student (MOPD). Mixing subjects
in one RL run makes them interfere, and running RL subject by subject makes
the model forget the earlier subjects.
</div>

## Why I ran this: the shape of LLM post-training {#why}

Open models released in 2025–2026 mostly follow the pipeline below. This note
shrinks the blue part, **RL and distillation (OPD, MOPD)**, until it runs on a
laptop, and runs it piece by piece.

<div class="rl-flow">
  <div class="rl-flow__step"><b>Pre-training</b><span>Predict the next word of internet text</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step"><b>SFT</b><span>Imitate good example answers. This experiment's "34% start"</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step is-here"><b>Per-subject RL</b><span>Practise each subject (math, code, …) against a grader to build expert teachers</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step is-here"><b>MOPD</b><span>Merge the teachers into one student with their red pens</span></div>
</div>
<p class="rl-caption">Figure 1. A typical modern pipeline. The blue steps are what this experiment runs.</p>

Xiaomi's MiMo-V2-Flash, for example, describes three stages: SFT, per-subject
RL to build teachers, then multi-teacher OPD. DeepSeek-V4 reportedly replaced
V3.2's mixed multi-subject RL stage entirely with OPD (I took that from a blog
analysis and did not check the original). Experiment 4 compares exactly these
two approaches.

## The setup: problems, grader, models {#stage}

### Problems: three-digit addition and subtraction, ones digit first

The model sees a problem like `699+684=` and writes the answer one character
at a time: four digits **starting from the ones place**, then `$` (end of
answer). 1383 becomes `3831$`.

Ones-first is the order of long addition by hand. You need the ones digit to
know the carry, and the carry to write the tens digit. So these five
characters are a tiny **reasoning trace**: a mistake early on shakes the
digits after it.

The vocabulary is 14 characters: `0`–`9`, `+`, `-`, `=` and `$`.
Subtraction always takes the smaller number from the larger, so answers are
never negative.

### The verifier: one point only if all five characters match

This grader is the "V" in RLVR (reinforcement learning with *verifiable*
rewards). There is no partial credit: writing 1183 for 1383 scores 0, even
though three of four digits are right.

```python
# rlsim.py: the verifier
def correct(seq, answers):
    return (seq[:, PROMPT_LEN:] == answers).all(1).float()   # 1 if every character matches, else 0
```

### Models: one small student, three larger teachers

Both are GPT-style decoder transformers shrunk down. The teacher is about 8×
the student.

| Model | Size | Parameters | Role |
| --- | --- | ---: | --- |
| Student | 2 layers, width 64 | 102,734 | The learner in every experiment |
| Teacher | 4 layers, width 128 | 798,606 | Teaches in SFT distillation, OPD and MOPD |

The three teachers were trained with SFT on true answers ([details](#teachers)).
Each learned only its own subject, so each scores close to 0% on the other one.
That matters in experiment 4.

| Teacher | Addition | Subtraction | How it was made |
| --- | ---: | ---: | --- |
| Addition teacher | 99.8% | 0.2% | SFT on addition only, 500 steps (16 s) |
| Subtraction teacher | 0.0% | 100% | SFT on subtraction only, 600 steps (18 s) |
| Weak addition teacher | 73.0% | 0.1% | Addition SFT stopped as soon as it passed 70% (275 steps) |

### How the teachers learned: SFT on the true answers {#teachers}

The teachers use neither RL nor distillation. This toy problem computes the
true answer (a + b, a − b) for free, so the teachers simply **copy correct
answers**. In code that is `load_or_train_teachers()` calling `sft_until()`
([walkthrough 12](#code-teachers)).

```python
# name: (subjects to learn, target accuracy, seed)
specs = {"add": (["+"], 0.995, 1), "sub": (["-"], 0.995, 2), "add-weak": (["+"], 0.70, 3)}

# repeat until the target is reached
prompts, answers, _ = mixed(BATCH, ops, rng)   # 256 random problems of its own subject + true answers
step_sft(model, opt, prompts, answers)         # copy the answer character by character (cross-entropy)
```

1. **Model:** a randomly initialized 4-layer, width-128 TinyGPT, 798,606
   parameters (about 8× the student).
2. **Training:** each step draws 256 fresh problems of its subject and trains
   it to reproduce the five answer characters with teacher forcing. Adam,
   learning rate 0.001.
3. **Stopping:** every few steps it is scored on 1,000 validation problems
   (seed 999) and stops once it beats the target. These are separate from the
   evaluation problems used for the charts (seed 12345).
4. **Saving:** stored as `results/teachers/*.pt` and only loaded afterwards.
   **Teachers never train during the experiments.**

| Teacher | Target (check interval) | Stopped at (time) | Validation accuracy |
| --- | --- | ---: | ---: |
| Addition teacher | 99.5% (every 100 steps) | 500 (16 s) | 99.9% |
| Subtraction teacher | 99.5% (every 100 steps) | 600 (18 s) | 100% |
| Weak addition teacher | 70% (every 25 steps) | 275 (9 s) | 75.2% |

- **The weak teacher was taught less on purpose.** Same recipe as the addition
  teacher, stopped as soon as it crossed 70%. Its mistakes are scattered across
  problems, which is why the student first overtakes it in experiment 3.
- **Outside its own subject it scores 0%.** The addition teacher never saw a
  `-` during training. That character's embedding never received a gradient and
  is still its random initial value. Given a subtraction problem it answers
  548 − 286 with 834: it adds.
- **The student's 34% starting point uses the same function**, with the target
  lowered to 30%.

This is the **same function** (`step_sft`) as the experiments' "SFT
distillation". Only the target answer differs.

| | Answer being copied | If the teacher is wrong |
| --- | --- | --- |
| Building a teacher | `answers`, the true answer | not applicable: the truth is the target |
| SFT distillation in the experiments | `teacher_tokens(...)`, the teacher's answer | the student learns the mistake as is |

<div class="rl-callout is-warn" markdown="1">
<span class="rl-callout__title">The teachers only ever saw correct prefixes</span>
In OPD the teacher reads the student's prefix, which may be wrong: a situation
it never saw in training. That was fine here. In addition, every digit's
correct value is fixed by the problem alone, so even after the student wrote 1
in the hundreds place, the teacher still gave the correct 1 for the thousands
place 99.3% of the time. In real LLM reasoning, a wrong early step changes what
is correct afterwards. How well a teacher handles the student's mistaken states
decides OPD's quality.
</div>

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">Teachers in real LLM training</span>
Correct labels aren't free, so teachers come from their own pipeline:
pre-training → SFT → per-subject RLVR. MiMo-V2-Flash says it built a teacher
per subject with RL from the same starting model and then merged them with
MOPD; Qwen3's small models were reported to be distilled from larger Qwen3
models. This toy's "SFT on true answers" stands in for "grow a teacher with RL".
</div>

### The student's starting point: about 34%

A real LLM starts RL after pre-training and SFT. So the student was also given
a little SFT on true answers, stopping once it passed 30%. That took 380–1,115
steps depending on the seed, and it starts at 34% on average on the evaluation
problems. Experiment 2 is the exception: it starts from an untrained 0% student.

### Kept equal for a fair comparison

| Item | Value | Meaning |
| --- | --- | --- |
| Answers per step | 256 | Same for every method. RLVR: 32 problems × 8; others: 256 problems × 1 |
| Learning rate | 0.0005 (Adam) | How far parameters move per update. Same for every method |
| Sampling | temperature 1 | During training, sample from the model's probabilities as they are |
| Evaluation | 2,000 fixed held-out problems | Greedy decoding (always the most likely character), then graded |
| Repeats | 3 seeds | Lines in the charts are means; shaded bands are min–max |

## The four methods on one problem {#methods}

One problem shows what each method teaches: **699 + 684 = 1383**. The 34%
student (seed 0) writes <span class="ans">1<span class="bad">1</span>83</span>:
only the hundreds digit is wrong.

### ① SFT distillation: copy the model answer

The teacher solves the problem first and writes a model answer. The student
copies it one character at a time, and while doing so it sees **the teacher's
prefix**, not its own. The student never solves anything itself.

The loss grows as the probability the student gives the teacher's character
shrinks (−log probability, i.e. cross-entropy). Copying the teacher's `1383`:

| Place | Teacher wrote | Student's probability for it | Loss (−log p) |
| --- | --- | ---: | ---: |
| ones | <span class="ans">3</span> | 99.7% | 0.00 |
| tens | <span class="ans">8</span> | 74.4% | 0.30 |
| hundreds | <span class="ans">3</span> | 10.6% | 2.25 |
| thousands | <span class="ans">1</span> | 99.8% | 0.00 |
| end | <span class="ans">$</span> | 99.8% | 0.00 |

In the hundreds place the student gives the correct 3 only 10.6%, the largest
loss. Training pushes the student toward "3 goes here".

```python
# step_sft in rlsim.py, condensed
targets = teacher_tokens(prompts, subjects, teachers)      # the teacher's model answer
seq = torch.cat([prompts, targets], 1)
loss = F.cross_entropy(answer_logits(model, seq), targets)  # −log p per character
```

<p class="rl-caption">Line-by-line notes and tensor shapes: <a href="#code-sft">walkthrough 7</a>.</p>

### ② RLVR (GRPO): solve the same problem 8 times, get only pass/fail

The student solves the same problem **8 times** itself. The verifier returns a
single pass/fail per answer. Each answer is then compared with the average of
the eight: answers better than average get their characters' probabilities
raised, worse ones lowered. That is GRPO (RL that compares answers to the same
problem against each other).

How much to raise or lower is the **advantage**: (that answer's score − mean of
the 8) ÷ standard deviation of the 8.

**Case A: 699 + 684 attempted 8 times, all wrong.**

<div class="chips"><span class="chip"><span class="ans">1<span class="bad">0</span><span class="bad">7</span>3</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">5</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">2</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">0</span><span class="bad">7</span>3</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">0</span><span class="bad">7</span>3</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">2</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">1</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">1</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span></div>

All eight score 0, so the mean is 0 and every difference is 0. Every advantage
is 0, so **nothing is learned from this problem**, not even which digit was
wrong.

**Case B: 227 + 189 (answer 416), right 2 times out of 8.**

<div class="chips"><span class="chip"><span class="ans"><span class="bad">5</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip is-ok"><span class="ans">416</span><span class="mark-ok">O</span><span class="adv">+1.62</span></span><span class="chip"><span class="ans"><span class="bad">6</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip"><span class="ans"><span class="bad">2</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip is-ok"><span class="ans">416</span><span class="mark-ok">O</span><span class="adv">+1.62</span></span><span class="chip"><span class="ans"><span class="bad">3</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip"><span class="ans"><span class="bad">8</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip"><span class="ans"><span class="bad">9</span><span class="bad">0</span>6</span><span class="mark-no">X</span><span class="adv">−0.54</span></span></div>

**Where +1.62 and −0.54 come from.** They measure how much better than the
average of its group of eight each answer did.

| Step | What it does | Computation | Result |
| --- | --- | --- | --- |
| ① Score | 1 if right, 0 if wrong | — | <span class="ans">0 1 0 0 1 0 0 0</span> |
| ② Mean | mean of the 8 scores | 2 ÷ 8 | **0.25** |
| ③ Difference | score − mean | right 1 − 0.25 / wrong 0 − 0.25 | **+0.75** / **−0.25** |
| ④ Std | square ③, sum, divide by (8 − 1), take √ | 0.75² × 2 + 0.25² × 6 = 1.50; ÷ 7 = 0.214 → √ | **0.463** |
| ⑤ Advantage | ③ ÷ ④ | +0.75 ÷ 0.463 / −0.25 ÷ 0.463 | **+1.62** / **−0.54** |

The sign comes from ③ and the size from ⑤. Dividing by 7 rather than 8 in ④
is PyTorch's `std()` default; dividing by 8 gives a std of 0.433 and
+1.73 / −0.58.

So the two right answers are pushed up with weight +1.62 and the six wrong ones
down with −0.54. But the wrong <span class="ans">516</span> got its last two
digits, "16", right, and the whole answer still gets −0.54, dragging the right
digits down too. A single pass/fail can't tell characters apart. Summed over
all eight answers, though, the picture changes a little
([walkthrough 8](#code-rlvr)).

Flip the marks yourself below and watch the learning signal change with the
number of right answers.

<div class="rl-grpo" data-rl-grpo></div>

**By number right.** Scores are only 0 or 1, so the number right out of 8
determines everything. Same formula as the calculator.

| Right out of 8 | Mean · std | Right answer | Wrong answer |
| --- | ---: | ---: | ---: |
| 0 | 0.000 · 0.000 | — | 0 |
| 1 | 0.125 · 0.354 | +2.47 | −0.35 |
| 2 (case B) | 0.250 · 0.463 | +1.62 | −0.54 |
| 3 | 0.375 · 0.518 | +1.21 | −0.72 |
| 4 | 0.500 · 0.535 | +0.94 | −0.94 |
| 5 | 0.625 · 0.518 | +0.72 | −1.21 |
| 6 | 0.750 · 0.463 | +0.54 | −1.62 |
| 7 | 0.875 · 0.354 | +0.35 | −2.47 |
| 8 | 1.000 · 0.000 | 0 | — |

- **The rare outcome moves most.** If only one of eight is right, it is pushed
  up by +2.47; if seven are right, the one wrong answer is pushed down by −2.47.
- **The advantages always sum to 0.** With two right: +1.62 × 2 − 0.54 × 6 = 0.
  Up and down balance, so characters that all eight answers share cancel out
  ([walkthrough 8](#code-rlvr)).
- **All right or all wrong gives 0.** Every difference in ③ is 0; there is
  nothing to compare. So RLVR learns nothing from problems that are **too hard
  or too easy**.
- **How wrong doesn't matter.** 516 (one digit off) and 906 (two digits off)
  both get −0.54, because scores are only 0/1. A grader with partial credit
  would give each answer a different value.

**What the number actually does.** The loss is −advantage × log p(answer).
Minimizing it raises the probability of that answer with strength 1.62 and
lowers others with strength 0.54. It is then scaled by the learning rate and
averaged over 256 answers × 5 characters, so it is a relative weight between
answers, not "the probability moves by 1.62".

Dividing by the std in ⑤ is debated. A very hard problem with one right answer,
or a very easy one with seven, has a small std, so dividing concentrates large
values there. Dr. GRPO drops the division and uses the raw differences from ③
(+0.75 / −0.25).

```python
# step_rlvr in rlsim.py, condensed
seq = write(model, prompts.repeat_interleave(8, 0))          # solve each problem 8 times
reward = correct(seq, answers).view(-1, 8)                   # pass/fail
advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
logp = log_prob_of_written_tokens(model, seq)                # log p of the characters written
loss = -(advantage.view(-1, 1) * logp).mean()                # push good answers up, bad ones down
```

<p class="rl-caption">Line-by-line notes, these eight answers' real log-probabilities, and where the signal survives once the eight are summed: <a href="#code-rlvr">walkthrough 8</a>.</p>

### ③ OPD: a red pen on the student's own answer

In OPD (on-policy distillation) the student solves the problem **itself**.
The teacher then reads the student's answer from the start and, at each
character, gives its own probabilities for "what I would write after this
prefix". The student learns to make its probabilities match. The teacher is
the larger model SFT-trained on true answers ([how the teachers learned](#teachers)).

Below is the teacher's red pen on the student's
<span class="ans">1<span class="bad">1</span>83</span>. Each row shows the
student's and the teacher's top probabilities at that position.

| Place: student wrote | Student (top) | Teacher (top) | Loss (reverse KL) |
| --- | --- | --- | ---: |
| ones: <span class="ans">3</span> | <span class="ans">3</span> 99.7% · <span class="ans">1</span> 0.1% · <span class="ans">2</span> 0.1% | <span class="ans">3</span> 99.9% | **0.01** |
| tens: <span class="ans">8</span> | <span class="ans">8</span> 74.4% · <span class="ans">7</span> 22.3% · <span class="ans">9</span> 2.3% | <span class="ans">8</span> 99.9% | **1.34** |
| hundreds: <span class="ans"><span class="bad">1</span></span> (true 3) | <span class="ans">1</span> 35.8% · <span class="ans">0</span> 22.4% · <span class="ans">2</span> 20.3% | <span class="ans">3</span> 99.2% · <span class="ans">4</span> 0.7% | **9.48** |
| thousands: <span class="ans">1</span> | <span class="ans">1</span> 99.8% · <span class="ans">2</span> 0.1% · <span class="ans">3</span> 0.1% | <span class="ans">1</span> 99.3% · <span class="ans">2</span> 0.6% | **0.01** |
| end: <span class="ans">$</span> | <span class="ans">$</span> 99.8% · <span class="ans">1</span> 0.1% | <span class="ans">$</span> 100% | **0.01** |

The loss is **reverse KL**: for every character the student puts probability
on, how much does the teacher put there? The more the student bets on
characters the teacher almost never writes, the larger the loss. As a formula:
Σ p_student(c) × log(p_student(c) ÷ p_teacher(c)).

- **Hundreds (9.48):** the student bets 36% on 1, the teacher almost 0% (and
  99% on 3). The wrong digit is pinpointed.
- **Tens (1.34):** written correctly as 8, but still penalized: the student also
  bet 22% on 7, which the teacher gives 0%. OPD fixes **right but unsure**,
  which RLVR cannot.
- **Other places (0.01):** student and teacher agree; almost no loss.

"On-policy" means the student learns from **answers it actually wrote**. SFT
distillation only ever sees the teacher's prefix; OPD also teaches "from here,
continue like this" after the student's own mistakes. The cost is one extra
teacher pass per answer.

```python
# step_opd in rlsim.py, condensed
seq = write(model, prompts)                                   # the student solves it (once)
s = answer_logits(model, seq).log_softmax(-1)                 # student probabilities per character
t = answer_logits(teacher, seq).log_softmax(-1)               # the teacher reading the same answer
loss = (s.exp() * (s - t)).sum(-1).mean()                     # reverse KL over all 14 characters
```

<p class="rl-caption">The gradient, and where probability moves in the hundreds place: <a href="#code-opd">walkthrough 9</a>.</p>

### ④ MOPD: each subject gets its own teacher's red pen

MOPD (multi-teacher OPD) is OPD that routes each problem by **subject tag to
that subject's teacher**: addition to the addition teacher, subtraction to the
subtraction teacher.

The two-subject starting student writes
<span class="ans">2<span class="bad">5</span>2</span> for **548 − 286**
(answer 262). The same answer under each teacher's red pen:

| Place | Student wrote | Subtraction teacher would | Addition teacher would |
| --- | --- | --- | --- |
| ones | <span class="ans">2</span> | <span class="ans">2</span> | <span class="ans">4</span> |
| tens | <span class="ans"><span class="bad">5</span></span> (true 6) | <span class="ans">6</span> | <span class="ans">3</span> |
| hundreds | <span class="ans">2</span> | <span class="ans">2</span> | <span class="ans">8</span> |
| thousands | <span class="ans">0</span> | <span class="ans">0</span> | <span class="ans">1</span> |

The subtraction teacher fixes only the wrong tens digit. The addition teacher
rewrites every digit: on its own it would write <span class="ans">834</span>,
i.e. 548 + 286, ignoring a minus sign it has never seen. Hand it subtraction and
it teaches the student to add (experiment 4).

```python
# rlsim.py: MOPD is OPD with a per-subject teacher map
step_opd(model, opt, prompts, subjects, teachers={"+": add_teacher, "-": sub_teacher})
```

| Method | Does the student solve it? | Feedback unit | Needs |
| --- | --- | --- | --- |
| SFT distillation | No | per character (vs the teacher's answer) | a teacher |
| RLVR | Yes, 8 times per problem | one pass/fail per answer | a grader |
| OPD | Yes, once per problem | a probability distribution per character | a teacher |
| MOPD | Yes, once per problem | per character, from the subject's teacher | a teacher per subject |

## The training environment: what each method sees, writes and gets back {#env}

All methods play the same "game": read a problem, write five characters, done.
Two things differ: **who writes the answer** and **what comes back**.

### The game, in RL terms

| RL term | In this experiment | Code |
| --- | --- | --- |
| state | the problem plus characters written so far, e.g. `699+684=` then `3`, `8` | prefix of `seq` |
| action | the next character, one of 14 | `nxt` in `write()` |
| episode | ends after five characters: ones → tens → hundreds → thousands → `$` | `ANSWER_LEN = 5` |
| policy | the student model: state in, 14 probabilities out | `TinyGPT` |
| reward | differs per method (table below) | `correct()` or the teacher |

### Per-method environment

| Item | SFT distillation | RLVR (GRPO) | OPD · MOPD |
| --- | --- | --- | --- |
| Problems per step | 256 random | 32 random, 8 attempts each | 256 random |
| Who writes the answer | **the teacher**, greedy | **the student**, sampled (temperature 1) | **the student**, sampled, once per problem |
| Prefix when predicting the next character | the teacher's characters (teacher forcing) | its own | its own |
| What comes back | the teacher's character at each position | 0 or 1 per answer, once at the end | the teacher's 14 probabilities at each position |
| As a reward | none: imitate a demonstration (behavior cloning) | sparse: one number for five characters | dense: −reverse KL at every character |
| Needs true answers? | No; the teacher's answer is treated as truth | **Yes**, the `correct()` grader | No |
| Needs a teacher? | Yes, writes 256 new answers every step | No | Yes, reads 256 student answers once (MOPD splits by subject) |
| Loss | cross-entropy | −advantage × log p | reverse KL (summed over all 14 characters) |
| Model passes per step | teacher ×5 (writing) + student ×1 (+backprop) | student ×5 (writing) + student ×1 (+backprop) | student ×5 (writing) + student ×1 (+backprop) + teacher ×1 (reading) |
| 1,000 steps, wall clock (exp. 1, incl. eval) | 45 s | 22 s | 31 s |
| Code | [`step_sft`](#code-sft) + `teacher_tokens` | [`step_rlvr`](#code-rlvr) | [`step_opd`](#code-opd) |

- **Why SFT is slowest:** a teacher 8× the student's size *writes* an answer
  every step, which takes one pass per character (5). OPD's teacher only *reads*
  an answer the student already wrote, producing all five positions in one pass.
  In real LLMs too, the OPD teacher's cost is reading (prefill), not generation,
  which is relatively cheap.
- **SFT and OPD never use the true answers.** `mixed()` returns them, and both
  methods throw them away. They only need a teacher, which is why a wrong
  teacher passes its mistakes on (experiment 3). RLVR is the reverse: no teacher,
  just the truth (the grader).
- **Only RLVR solves the same problem several times.** One pass/fail says
  nothing about "good"; it has to be compared with other answers to the same
  problem. So the same 256 answers cover only 32 problems.

### What every method shares

- **Starting weights:** each seed builds one starting student, and every method
  gets its own `copy.deepcopy(start)`. All start from exactly the same weights.
- **Optimizer:** a fresh Adam per method, learning rate 0.0005, gradients
  clipped to total norm 1 (`clip_grad_norm_`).
- **Problem order:** each method rebuilds its RNG from the same seed. SFT and OPD
  draw 256 problems per step, so they see **the same problems in the same
  order**. RLVR draws 32 per step, so its order differs.
- **Evaluation:** every 20 steps, 2,000 fixed problems per subject (seed 12345),
  greedy decoding. Evaluation never affects training.

### What changed per experiment

| Experiment | Starting student | Subjects · steps | Teacher |
| --- | --- | --- | --- |
| 1 | addition SFT to 34% (410 · 1,115 · 380 steps by seed) | addition · 1,000 | SFT/OPD: addition teacher (99.8%) |
| 2 | random init (0%) | addition · 1,000 | OPD: addition teacher |
| 3 | the same 34% student as experiment 1 | addition · 3,000 | SFT/OPD: weak addition teacher (73%) |
| 4 | two-subject SFT: addition 34% · subtraction 30% (1,585 · 1,220 · 400 steps) | half and half · 1,000; Cascade does addition 500 → subtraction 500 | MOPD: per-subject teachers; control: the addition teacher for both |

### Hardware

CPU only on an Apple M4 Max laptop (no GPU). Python 3.11, PyTorch 2.10, NumPy
2.3, with `torch.set_num_threads(4)`. Building the three teachers takes under a
minute; the four experiments about 35 minutes. No external data and no
pre-trained weights: the code generates the problems on the spot.

## Experiment 1: a student starting at 34% {#exp1}

**Question:** from the same starting point, which method learns fastest?<br>
**Setup:** the 34% student, addition only, 1,000 steps. SFT distillation and
OPD learn from the addition teacher (99.8%).

<figure class="rl-chart" data-rl-chart="single"></figure>
<p class="rl-caption">Figure 2. Addition accuracy. Hover a line to see that step's values.</p>

| Method | After 1,000 steps (seed range) | Reached 90% | CPU time |
| --- | ---: | ---: | ---: |
| SFT distillation | 99.9% (99.8–99.9) | 180 steps | 45 s |
| OPD | 98.6% (97.4–100) | 280 steps | 31 s |
| RLVR | 81.0% (78.5–83.4) | never | 22 s |

**Why:** the information per step differs. RLVR writes 256 answers and learns
only "how many of the 8 were right" for each of 32 problems. OPD gets a full
14-way probability distribution for each of the five characters of every
answer.

On top of that, RLVR ran out of problems worth learning from. At the start 61%
of problems carried a signal; by step 1,000 only 24%. Early on, hard problems
with 8 wrong answers carry no signal; later, easy problems with 8 right answers
don't either (figure 3, right).

Here SFT distillation was slightly faster than OPD. The answers are only five
characters, so there is little "continue from your own mistake" to learn. More
on that in [what carries over to real LLMs](#real).

## Experiment 2: a student that knows nothing {#exp2}

**Question:** can a blank student learn with each method?<br>
**Setup:** a randomly initialized student (0%), addition only, 1,000 steps.

<div class="rl-pair">
<div><div class="rl-pair__title">Accuracy from 0%</div><figure class="rl-chart" data-rl-chart="cold"></figure></div>
<div><div class="rl-pair__title">Problems RLVR could learn from</div><figure class="rl-chart" data-rl-chart="signal"></figure></div>
</div>
<p class="rl-caption">Figure 3. Left: accuracy of the 0% student. Right: share of problems where RLVR's 8 answers were not all the same (i.e. carried a signal).</p>

| Method | After 1,000 steps (seed range) | Reached 90% | Problems with signal |
| --- | ---: | ---: | ---: |
| OPD | 96.4% (93.4–98.4) | 740 steps | always 100% |
| RLVR | 0.0% (0.0–0.0) | never | 0% |

**Why:** a blank student gets all five characters right by chance with
probability about (1/14)⁵, roughly 1 in 540,000. Over 1,000 steps × 256 answers
= 256,000 answers, the expected number of right answers is 0.5. Every group of
8 is all wrong, the signal stays 0, and the student never changes.

OPD tells the student "this digit goes here" at every character even when the
answer is wrong, so it learns from scratch.

### Why OPD suddenly jumps around step 400–500 {#cold-jump}

Accuracy sits near 0% for 400 steps and then shoots up around step 500. I re-ran
seed 0 (the accuracy curve reproduces exactly) and measured accuracy per digit.

| Step | Per-digit accuracy % (ones · tens · hundreds · thousands · $) | Exact match | KL |
| ---: | --- | ---: | ---: |
| 0 | <span class="ans">11 · 10 · 0 · 0 · 0</span> | 0.0% | — |
| 100 | <span class="ans">10 · 11 · 10 · 71 · 100</span> | 0.2% | 4.57 |
| 200 | <span class="ans">12 · 10 · 11 · 75 · 100</span> | 0.3% | 4.45 |
| 300 | <span class="ans">30 · 10 · 10 · 73 · 100</span> | 0.5% | 4.30 |
| 400 | <span class="ans">90 · 26 · 10 · 78 · 100</span> | 2.3% | 3.19 |
| 500 | <span class="ans">99 · 90 · 81 · 98 · 100</span> | 73.4% | 0.92 |
| 600 | <span class="ans">99 · 98 · 98 · 100 · 100</span> | 95.3% | 0.12 |
| 1,000 | <span class="ans">99 · 99 · 99 · 100 · 100</span> | 97.4% | 0.04 |

<p class="rl-caption">2,000 evaluation problems, greedy. KL is the average loss over the preceding 20 steps.</p>

- **Easy positions come first.** The end marker `$` hits 100% within 100 steps,
  and the thousands digit (always 0 or 1) quickly reaches the 70s. Neither needs
  much arithmetic.
- **Positions that need arithmetic sit at 10% for a while.** Ones, tens and
  hundreds hover around 10% until step 300, guessing among ten digits. KL barely
  moves from 4.3–4.6. The model seems not to have found which input positions to
  add yet.
- **Addition unlocks in long-addition order.** The ones digit goes 30% → 90%
  between steps 300 and 400; then between 400 and 500 the tens go 26% → 90% and
  the hundreds 10% → 81%. The ones digit (no incoming carry) comes first, the
  digits that need a carry follow. Subtraction goes in a different order
  ([below](#cold-sub)).
- **Exact match needs all five.** At step 400 the ones digit is at 90% but exact
  match is 2.3%, because the other digits are still guessing. Exact match is
  close to the product of the per-digit accuracies (step 500: product 71%,
  actual 73%).

Two things overlap. One: the model takes a while to find the digit rule and then
learns it fast; per-digit accuracy really does jump. Two: a 5-for-5 grade makes
digit-by-digit progress look like "flat, then a leap". Feeding the correct
prefix (teacher forcing) gives almost the same per-digit accuracy, so this is
not early mistakes propagating. Experiment 1 starts at 34% and already knows
part of the rule, so it has no plateau.

### Subtraction? The order follows how hard each digit's arithmetic is {#cold-sub}

Subtraction does not go ones-first, and it doesn't go in reading order either.
All six runs (OPD and true-answer SFT, three seeds each) behaved like this:

- **The hundreds digit passes 50% first**, at steps 200–220, and then stalls in
  the 60s–70s for a while.
- **The ones digit is the first to finish (90%)**, at steps 420–940, followed by
  the hundreds. The tens digit is last; under OPD it never reached 90% within
  1,000 steps.
- **Subtraction is harder than addition.** After 1,000 OPD steps, exact match is
  72.3–83.1% for subtraction versus 93.5–98.4% for addition.

| Run | Step past 50% (ones · tens · hundreds) | Step past 90% (ones · tens · hundreds) | Final exact match (seed range) |
| --- | --- | --- | ---: |
| Addition OPD (1,000 steps) | 433 · 540 · 560 | 507 · 647 · 653 | 96.4% (93.5–98.4) |
| Subtraction OPD (1,000 steps) | 340 · 620 · 213 | 827 · never · 970 (2/3) | 79.3% (72.3–83.1) |
| Addition true-answer SFT (2,000 steps) | 433 · 580 · 780 | 473 · 667 · 873 | 100.0% (100.0–100.0) |
| Subtraction true-answer SFT (2,000 steps) | 493 · 880 · 213 | 573 · 1,100 · 887 | 99.8% (99.5–100.0) |

<p class="rl-caption">Mean of three seeds. "(2/3)" means only two of the three seeds got there; the mean uses those seeds only.</p>

Splitting up the arithmetic each digit needs shows why. Below, a "wrap" is the
step that goes past 10 or below 0 and comes back around (7 + 5 → 2, 3 − 8 → 5).

| Digit | Addition | Subtraction (larger − smaller) |
| --- | --- | --- |
| Ones | one wrap | one wrap |
| Tens | wrap + carry (needs the ones digit) | wrap + borrow (needs the ones digit) |
| Hundreds | wrap + carry (needs the tens and ones) | no wrap + borrow (needs the tens and ones) |

Subtraction here always takes the smaller number from the larger, so
subtracting the hundreds digits gives 0–9 directly, with no wrap. Ignoring the
borrow and just writing "hundreds of a − hundreds of b" is already right more
than half the time. The stall in the 60s–70s looks like that stage. Getting the
borrow right needs the lower digits, so the hundreds only pass 90% after the
ones digit is learned. The tens digit needs both a wrap and a borrow, so it's
last.

Addition's ones → tens → hundreds order also looks less like "long-addition
order" and more like the order of easiest arithmetic, which happens to coincide.
The model learns the easy computation first, not the writing order.

### What if the answer is written biggest digit first? It learns far worse {#cold-order}

So does writing order not matter? I switched the answer to reading order
(thousands digit first) and compared it with the current ones-first format.
There is no teacher for that format, so this uses true-answer SFT from a 0%
student for 2,000 steps (three seeds, `digit_order.py`).

<figure class="rl-chart" data-rl-bars="order"></figure>
<p class="rl-caption">Figure 3-3. True-answer SFT after 2,000 steps, per-digit and exact-match accuracy (mean of three seeds). The thousands digit is 100% in every format and is left out.</p>

| Format | Exact match at 2,000 steps (seed range) | Per-digit % (ones · tens · hundreds · thousands) | Seeds that reached 90% (ones · tens · hundreds · thousands) |
| --- | ---: | --- | --- |
| Addition · ones first | 100.0% (100.0–100.0) | 100 · 100 · 100 · 100 | 3/3 · 3/3 · 3/3 · 3/3 |
| Addition · thousands first | 14.1% (7.8–22.7) | 31 · 71 · 95 · 100 | 0/3 · 1/3 · 3/3 · 3/3 |
| Subtraction · ones first | 99.8% (99.5–100.0) | 100 · 100 · 100 · 100 | 3/3 · 3/3 · 3/3 · 3/3 |
| Subtraction · thousands first | 13.7% (2.5–33.1) | 40 · 30 · 96 · 100 | 1/3 · 0/3 · 3/3 · 3/3 |

- **Ones first:** after 2,000 steps addition is at 100% and subtraction at
  99.5–100%.
- **Thousands first:** addition 8–23%, subtraction 3–33%. The thousands and
  hundreds digits come quickly (90% reached at steps 0–100 for thousands,
  420–920 for hundreds). The tens and ones digits reached 90% within 2,000 steps
  in only one of six seeds each.

Writing ones first turns the low digits the model has already written into a
scratchpad for the next digit. If a's ones digit is 7 and the model wrote 2 as
the answer's ones digit, it can see there was a carry (7 + 5 = 12). Writing
thousands first means computing every carry in its head in one go. The hundreds
digit stalling at 93–97% in that format also looks like a carry guessed from the
tens digit alone. For addition, the ones digit only decides the carry when the
tens digits sum to exactly 9, which is about 4.5% of problems, so guessing gets
you to about 95%. This matches the finding that small transformers learn
addition much better when the answer is written reversed, ones digit first
(Lee et al., 2023).

What I could not explain is why, in the thousands-first format, the ones digit
(the easiest arithmetic) is learned last.

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">In real LLM training</span>
This is the same reason we have LLMs write out their reasoning (chain of
thought). Writing intermediate results first makes the next step easier. The
five ones-first characters are this experiment's smallest "reasoning trace"
([the setup](#stage)).
</div>

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">In real LLM training</span>
This is why RL is preceded by SFT that gets the model to "sometimes right"
(cold-start SFT). This experiment's starting student (random → 30% after
380–1,115 steps of SFT) and teachers (random → 99.9% after 500 steps) were made
exactly this way. Small models often skip RL and are distilled from a larger
one instead; Qwen3's small models were reported to be built that way. Whether
anything other than SFT gets RL started is compared in [experiment 5](#exp5).
</div>

## Experiment 3: when the teacher is weak {#exp3}

**Question:** how far does a student get with a teacher that makes mistakes?<br>
**Setup:** the 34% student, addition only, 3,000 steps. SFT distillation and OPD
learn from the weak addition teacher (73%). RLVR uses only the grader.

<figure class="rl-chart" data-rl-chart="weak"></figure>
<p class="rl-caption">Figure 4. Addition accuracy. The dashed line is the weak teacher's skill (73%).</p>

| Method | Best (at step) | After 3,000 steps (seed range) | Reached 90% |
| --- | ---: | ---: | ---: |
| OPD (weak teacher) | 84.5% (840) | 78.9% (78.8–79.1) | never |
| SFT distillation (weak teacher) | 82.5% (680) | 79.4% (79.1–79.9) | never |
| RLVR | 92.4% (2,920) | 91.8% (88.3–98.5) | 1,660 steps |

**At first the student beat the teacher.** Taught by a 73% teacher, the student
reached 84%. The teacher's mistakes are scattered across problems, so the
student picks up the shared rule before the noise.

**With more training it came back down.** As the student matches the teacher's
probabilities ever more closely, it copies the teacher's mistakes too, and falls
to 79%. Distillation means "become the teacher", so the teacher is the ceiling.

**RLVR is slow but has no ceiling.** Its reference is the truth, not a teacher.
It catches up with distillation around step 1,000 and crosses 90% at step 1,660.

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">In real LLM training</span>
Hence "grow the teacher with RL, then transfer it with distillation".
Distillation alone can't beat the best teacher; RL alone is too slow and
expensive.
</div>

## Experiment 4: two subjects {#exp4}

**Question:** how do you make one student good at two subjects?<br>
**Setup:** a student with a little of both (addition 34%, subtraction 30%),
1,000 steps, four methods.

- **Mix-RL:** each step mixes 16 addition and 16 subtraction problems, 8
  attempts each, pass/fail.
- **Cascade-RL:** 500 steps of RL on addition, then 500 on subtraction.
- **MOPD:** each step, the addition teacher red-pens 128 addition problems and
  the subtraction teacher 128 subtraction problems.
- **Addition teacher only:** the same OPD, but the addition teacher grades both
  subjects.

<div class="rl-pair">
<div><div class="rl-pair__title">Addition accuracy</div><figure class="rl-chart" data-rl-chart="multi-add"></figure></div>
<div><div class="rl-pair__title">Subtraction accuracy</div><figure class="rl-chart" data-rl-chart="multi-sub"></figure></div>
</div>
<p class="rl-caption">Figure 5. Accuracy on both subjects. At the vertical dashed line (step 500) Cascade-RL switches from addition to subtraction.</p>

| Method | Addition (seed range) | Subtraction (seed range) | In one line |
| --- | ---: | ---: | --- |
| MOPD | 98.3% (97.2–99.2) | 97.5% (95.6–98.7) | kept both |
| Addition teacher only | 99.1% (98.8–99.4) | 0.2% (0.1–0.2) | learned subtraction as addition |
| Mix-RL | 44.6% (42.5–48.4) | 44.5% (40.0–50.2) | dips, then climbs slowly |
| Cascade-RL | 2.6% (0.5–5.8) | 6.7% (0.7–17.6) | both collapsed |

### MOPD: the right teacher for each subject

Both subjects pass 97% by step 1,000, reaching 90% at step 480 (addition) and
540 (subtraction). Each subject is corrected character by character by a teacher
who knows it, so the subjects don't interfere.

### Addition teacher only: you can't teach what you don't know

Addition hits 99%, but subtraction falls from 30% to 3% within 20 steps and
then to 0%. As shown above, the addition teacher writes 834 for 548 − 286. The
student faithfully follows that red pen and adds on subtraction problems too.
This is why MOPD keeps a separate teacher per subject.

### Mix-RL: mixing makes the subjects shake each other

In the first 80 steps addition dropped from 34% to 18% and subtraction from 30%
to 21%. Reinforcing right answers in one subject shook the other through shared
parameters (the see-saw effect). Each subject also gets only 16 problems per
step, half the signal. It then climbs slowly to 45% by step 1,000.

### Cascade-RL: learn one, forget the other

- **Steps 0–500 (addition only):** addition rose 34% → 67%, but subtraction fell
  30% → 3.5% without ever being trained: catastrophic forgetting.
- **Step 500 (switch to subtraction):** subtraction is already almost gone, so
  most groups of 8 are all wrong. The share of problems with a signal fell from
  58.5% in the addition phase to 4.9%.
- **Steps 500–600:** addition then collapsed too, 67% → 6.5%. Rare subtraction
  signals kept moving the parameters and seem to have shaken addition as well; I
  didn't isolate the exact cause.

<div class="rl-callout is-warn" markdown="1">
<span class="rl-callout__title">The size of the cascade collapse may be exaggerated</span>
There is no KL constraint ("don't drift too far from the original model") and
no replay of earlier-subject data. Real pipelines use exactly these to limit
forgetting. Forgetting in this direction is a real, reported problem, but a
collapse down to 3% owes a lot to this setup.
</div>

## Experiment 5: getting RLVR started on a 0% student {#exp5}

**Question:** in experiment 2, RLVR never received a single signal from the 0%
student. What has to change for it to start learning?<br>
**Setup:** the same random 0% student, addition only, 256 answers per step, 3
seeds. Five methods for 1,000 steps; the three that started learning were run
to 3,000.

| Method | What changed | In code |
| --- | --- | --- |
| Easy problems first | 0–9 operands for 300 steps → 0–99 for 300 → 0–999 for 400. Still pass/fail | only the problems change |
| Partial credit (per answer) | score = fraction of the five positions right (0, 0.2, …, 1), one number per answer | `hit.mean(1)` |
| Per-digit grading | grade each position separately; advantages computed per position across the 8 answers | `hit` per position |
| Injected answer | in each group of 8, replace one answer with the true answer; the student writes the other 7 | `seq[::8] = answers` |
| True-answer SFT (baseline) | no RL: copy 256 true answers | `step_sft(answers)` |

```python
def step_rlvr_variant(model, opt, prompts, answers, method: str) -> dict:
    """The same GRPO step as rlsim.step_rlvr, plus a switch for the variants."""
    prompts, answers = prompts.repeat_interleave(s.GROUP, 0), answers.repeat_interleave(s.GROUP, 0)
    seq = s.write(model, prompts)
    if method == "inject":  # replace the first answer of each group with the true answer
        seq[:: s.GROUP, s.PROMPT_LEN :] = answers[:: s.GROUP]
    hit = (seq[:, s.PROMPT_LEN :] == answers).float()  # (256, 5): right at each position?
    exact = hit.all(1).float()
    logp = s.answer_logits(model, seq).log_softmax(-1).gather(-1, seq[:, s.PROMPT_LEN :, None]).squeeze(-1)
    if method == "token":  # per position: mean and std over the same position of the 8 answers
        reward = hit.view(-1, s.GROUP, s.ANSWER_LEN)
        advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
        loss = -(advantage.reshape(-1, s.ANSWER_LEN) * logp).mean()
        signal = (reward.std(1) > 0).float().mean().item()
    else:
        reward = (hit.mean(1) if method == "partial" else exact).view(-1, s.GROUP)
        advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
        loss = -(advantage.view(-1, 1) * logp).mean()
        signal = (reward.std(1) > 0).float().mean().item()
    s.update(opt, model, loss)
    own = exact.view(-1, s.GROUP)[:, 1:] if method == "inject" else exact.view(-1, s.GROUP)  # student-written answers only
    return {"signal": signal, "own_exact": own.mean().item()}
```

<p class="rl-caption"><code>exp5.py</code> (lines 50–71) adds a switch to <code>rlsim.py</code>'s <code>step_rlvr</code>; <code>s</code> is <code>rlsim</code>, left unchanged.</p>

### After 1,000 steps

| Method | Accuracy at 1,000 (seed range) | Per-digit accuracy % (ones · tens · hundreds · thousands · $) | Share with signal |
| --- | ---: | --- | ---: |
| plain RLVR (exp. 2) | 0.0% (0.0–0.0) | unchanged | 0% |
| easy problems first | 0.0% (0.0–0.0) | unchanged | 0% |
| partial credit (per answer) | 0.0% (0.0–0.1) | <span class="ans">14 · 11 · 10 · 50 · 100</span> | 86% |
| per-digit grading | 0.2% (0.1–0.2) | <span class="ans">44 · 15 · 18 · 66 · 100</span> | 32% (per position) |
| injected answer | 0.6% (0.2–1.4) | <span class="ans">37 · 14 · 10 · 76 · 100</span> | 100% |
| true-answer SFT | 72.3% (16.9–100.0) | <span class="ans">100 · 100 · 72 · 94 · 100</span> | — |
| for reference: OPD (exp. 2) | 96.4% (93.4–98.4) | — | 100% |

<p class="rl-caption">Per-digit accuracy is the mean of three seeds. "Unchanged" means the gradient stayed 0 and the weights never moved from their initial values.</p>

### To 3,000 steps: the three that started

<figure class="rl-chart" data-rl-chart="exp5"></figure>
<p class="rl-caption">Figure 6. Addition accuracy of the 0% student. Hover a line to see that step's values.</p>

| Method | Accuracy at 3,000 (seed range) | Reached 90% (seeds 0 · 1 · 2) | Per-digit accuracy % (ones · tens · hundreds · thousands · $) |
| --- | ---: | ---: | --- |
| true-answer SFT | 100.0% (100.0–100.0) | 820 · 1,140 · 680 | <span class="ans">100 · 100 · 100 · 100 · 100</span> |
| injected answer | 69.7% (9.7–100.0) | 1,800 · never · 1,480 | <span class="ans">100 · 100 · 70 · 94 · 100</span> |
| per-digit grading | 23.9% (5.5–36.3) | never · never · never | <span class="ans">82 · 69 · 49 · 80 · 100</span> |

### Why

- **Easy problems first: the weights never moved.** Even 0–9 problems still have
  five-character answers (7 + 5 → `2100$`). A student that doesn't know the
  format never gets all five right, so every group scores 0, advantages are 0
  and so is the gradient. Per-digit accuracy after 1,000 steps is identical to
  the start. Easy-first only helps a student that is already sometimes right.
- **Partial credit (per answer): only the easy positions.** The end marker `$`
  reached 100%, and the thousands digit stopped at exactly 50.2% in all three
  seeds: 50.2% of evaluation sums are ≥ 1,000, so it appears to always write 1.
  Ones, tens and hundreds stay at 10%, pure guessing. The score is one number
  for the whole answer, so +0.2 for a lucky ones digit is spread equally over all
  five characters and drowned out by the noise of other positions being right or
  wrong by chance. It can't tell which position earned it.
- **Per-digit grading: learns per position, but slowly.** An answer with the
  right ones digit raises only the ones character. Unlike partial credit,
  per-digit accuracy really does rise (ones, tens, hundreds at 82 · 69 · 49% by
  step 3,000). But it learns by trial and error, only when it happens to write
  the right digit (10%), so exact match is just 24% at step 3,000. It is a tiny
  version of a process reward.
- **Injected answer: works, but about 2× slower than SFT.** By step 3,000 two
  seeds are at 99–100% (90% at steps 1,480 and 1,800). One seed is at 9.7%; that
  seed was also the slowest under true-answer SFT. At first all 7 student answers
  are wrong and each gets −0.35. Uniformly lowering answers sampled from the
  student's own distribution points nowhere on average (the expected gradient of
  the log-probability of a sampled answer is 0). So what is actually learned is
  the 32 injected answers at +2.47: an eighth of SFT's data (where all 256 are
  correct), plus noise. Once the student starts getting answers right itself, it
  becomes real RLVR; by the end, 99% of the answers it wrote were right.
- **True-answer SFT: the fastest.** At step 1,000 it ranges from 17% to 100% by
  seed; by step 3,000 all three are at 100%. It reached 90% at steps 680–1,140,
  similar to OPD in experiment 2 (520–840).

<div class="rl-callout is-decision" markdown="1">
<span class="rl-callout__title">A 0% student has to be shown the answer</span>
Only methods that show "what the answer is" learned properly: true-answer SFT,
OPD and answer injection. Changing only the grading helps a little when the unit
is small enough to be hit by chance (per digit), and not at all when all five
characters must be right, however easy the problems. In practice, cold-start
SFT first gets the model to "sometimes right", then RL takes over. Answer
injection is the same idea as mixing a stronger model's answers into RL rollout
groups (e.g. LUFFY), and per-digit grading is the same idea as a process reward
model (PRM).
</div>

<div class="rl-callout is-warn" markdown="1">
<span class="rl-callout__title">Limits of this experiment</span>
- Only 3 seeds and 3,000 steps. Whether the slow injection seed and per-digit
  grading go further with more steps is untested.
- Injected answers were used as if the student had written them. Real work
  corrects for the fact that they weren't (importance weighting and similar).
- Per-digit grading was possible only because this toy knows the right answer
  at every position. Real reasoning has no per-step ground truth, so a process
  reward model has to be trained separately.
</div>

## Experiment 6: million-scale arithmetic, four teachers {#exp6}

**Question:** experiments 1–5 use three-digit addition and subtraction, which is
very simple, and the right answer comes for free. If the problems grow to
million-scale arithmetic with all four operations and one teacher per
operation, does MOPD clearly beat the other methods?<br>
**Setup:** 61 jobs on a dev GPU box (two H100 NVL), about 3 hours 40 minutes.
Each student run is 10,000 steps, three seeds. The code is a single file,
`big/bigsim.py` (not published here).

| | Experiments 1–5 | Experiment 6 |
| --- | --- | --- |
| Numbers | 0–999 | 0–999,999; each operand's digit count drawn uniformly from 1–6 |
| Operations | + − | + − × ÷ (division gives quotient and remainder) |
| Answer | four digits + `$` | 12 digits + `$`. +, − and × are written ones first; ÷ writes the quotient, then the remainder, biggest digit first (long-division order) |
| Student · teacher | 100K · 800K parameters | 3.2M · 10.7M parameters |
| Evaluation | 2,000 problems | operation × digit count (1–6): 24 cells × 250 = 6,000 problems |

### Making "the answer is free" less free

- **True-answer SFT baseline:** the result when the correct worked answer can be
  handed over directly. Other methods have to beat this to mean anything.
- **RL-polished teachers:** only 10,000 steps of true-answer SFT, then 20,000
  more steps of RLVR. Closer to how real teachers are made.
- **A systematically wrong teacher:** an addition teacher that always drops the
  carry into the thousands digit. Not random errors: the same mistake every
  time.
- **A grader that's sometimes wrong:** during RL, each pass/fail mark is flipped
  with 10% probability.

### The teachers are nearly perfect

| Teacher | Accuracy % (own operation; for the four-operation teacher + · − · × · ÷) | By digit count % (1–6) | Training time |
| --- | --- | --- | ---: |
| Addition (true-answer SFT) | 100.0 | 100 · 100 · 100 · 100 · 100 · 100 | 35 min |
| Subtraction (true-answer SFT) | 100.0 | 100 · 100 · 100 · 100 · 100 · 100 | 35 min |
| Multiplication (true-answer SFT) | 99.8 | 100 · 100 · 100 · 100 · 100 · 99 | 35 min |
| Division (true-answer SFT) | 97.1 | 100 · 100 · 100 · 100 · 98 · 86 | 35 min |
| Four-operation teacher (4× the training) | 100.0 · 100.0 · 100.0 · 97.3 | — | 138 min |
| Multiplication (RL-polished) | 99.6 | 100 · 100 · 100 · 100 · 100 · 98 | 142 min |
| Division (RL-polished) | 90.5 | 100 · 100 · 99 · 95 · 84 · 65 | 69 min |
| Addition (carry bug) | 80.4 | 100 · 100 · 80 · 70 · 71 · 61 | 36 min |

A 10.7M-parameter teacher gets 6-digit × 6-digit multiplication right 99.8% of
the time with no worked steps. That looks like the ones-first format plus an
effectively unlimited supply of problems. The teacher that learned all four
operations (4× the training) is as good as the specialists: 100% on +, − and ×,
97.3% on ÷. For the RL-polished teachers, 20,000 RL steps did not raise
accuracy (multiplication 99.9 → 99.6%, division 91.8 → 90.5%).

The starting student is 1,000 steps of true-answer SFT. It is already at
99.9–100% on addition and subtraction, and starts at 28–32% on multiplication
and 23–32% on division.

### Results: the distillation methods are about the same, and RL barely learns

| Method (10,000 steps) | Overall accuracy (seed range) | + · − % | × · ÷ % |
| --- | ---: | --- | --- |
| Starting student | 64.3% (63.4–65.5) | 100.0 · 100.0 | 30.2 · 27.0 |
| True-answer SFT (baseline) | 82.5% (80.6–83.6) | 99.9 · 99.9 | 57.7 · 72.3 |
| SFT distillation | 82.1% (80.6–83.6) | 100.0 · 100.0 | 58.2 · 70.3 |
| MOPD | 81.2% (80.4–81.9) | 100.0 · 100.0 | 55.1 · 69.6 |
| Single-teacher OPD | 81.1% (79.8–81.9) | 100.0 · 99.9 | 58.4 · 66.1 |
| MOPD → RL | 78.0% (77.4–78.9) | 100.0 · 100.0 | 48.1 · 63.8 |
| Mix-RL | 65.6% (64.5–66.8) | 100.0 · 100.0 | 31.8 · 30.4 |
| Cascade-RL | 50.8% (43.5–55.0) | 69.8 · 76.1 | 26.9 · 30.6 |

<figure class="rl-chart" data-rl-bars="big"></figure>
<p class="rl-caption">Figure 6-1. The student's multiplication and division accuracy after 10,000 steps (mean of three seeds).</p>

- **All four distillation methods match true-answer SFT.** True-answer SFT
  82.5%, SFT distillation 82.1%, MOPD 81.2%, single-teacher OPD 81.1%. With a
  near-perfect teacher, copying the teacher and copying the truth are the same
  thing. MOPD did not come out ahead.
- **Why MOPD didn't win:** here even the teacher that learned all four
  operations is as good as the specialists, so a teacher per subject buys
  nothing. MOPD won experiment 4 because the addition teacher knew nothing about
  subtraction. Answers are also short (13 positions), so OPD's strength,
  learning to continue after one's own mistakes, has little room to show.
- **The ceiling is the student.** The teacher is at 99.8% on multiplication;
  the student sits near 58% whatever the method, stuck on large multiplications
  (table below). The distillation curves were still rising 0.9–2.3 points over
  the last 2,000 steps, so 10,000 steps is not converged.
- **Mix-RL barely moves.** Multiplication 30 → 32%, division 27 → 30%. Only
  about 3% of problems have some but not all of the 8 answers right, because a
  point needs all 13 positions correct and the student rarely gets large ×
  or ÷ right.
- **Cascade-RL forgets the earlier subjects.** Running + → − → × → ÷ for 2,500
  steps each dropped addition 100 → 69.8% and subtraction 100 → 76.1%.
  Multiplication and division stay near the start.
- **MOPD → RL ends below MOPD alone** (78.0 vs 81.2%). After 5,000 MOPD steps,
  5,000 RL steps added 1.0–1.4 points; the same 5,000 steps spent on more MOPD
  added 4.0–4.9.

| Accuracy by digit count % (longer operand, 1–6 digits) | Multiplication | Division |
| --- | --- | --- |
| Teacher (true-answer SFT) | 100 · 100 · 100 · 100 · 100 · 99 | 100 · 100 · 100 · 100 · 98 · 86 |
| Starting student | 100 · 39 · 18 · 11 · 9 · 4 | 91 · 39 · 14 · 9 · 5 · 3 |
| True-answer SFT (baseline) | 100 · 99 · 65 · 35 · 27 · 20 | 100 · 98 · 85 · 73 · 42 · 35 |
| SFT distillation | 100 · 99 · 66 · 38 · 26 · 20 | 100 · 98 · 85 · 67 · 40 · 32 |
| MOPD | 100 · 93 · 60 · 36 · 25 · 17 | 100 · 98 · 85 · 65 · 40 · 29 |
| Single-teacher OPD | 100 · 94 · 69 · 41 · 27 · 20 | 100 · 97 · 80 · 58 · 35 · 26 |
| MOPD → RL | 100 · 84 · 46 · 25 · 19 · 14 | 100 · 97 · 75 · 54 · 32 · 25 |
| Mix-RL | 100 · 41 · 22 · 13 · 10 · 5 | 96 · 45 · 19 · 12 · 7 · 4 |
| Cascade-RL | 95 · 33 · 15 · 9 · 6 · 3 | 97 · 45 · 19 · 12 · 6 · 4 |

### Realism conditions

| Condition | Method | This condition (seed range) | Baseline condition |
| --- | --- | ---: | ---: |
| RL-polished teachers | MOPD | 82.2% (81.1–83.1) | 81.2% (80.4–81.9) |
| | SFT distillation | 82.0% (80.9–83.2) | 82.1% (80.6–83.6) |
| | MOPD → RL | 78.8% (77.5–79.5) | 78.0% (77.4–78.9) |
| Bug teacher (addition only) | MOPD | addition 80.4% (80.4–80.4) | addition 100.0% (99.9–100.0) |
| | SFT distillation | addition 80.3% (80.1–80.4) | addition 100.0% (99.9–100.0) |
| | MOPD → RL | addition 86.9% (80.4–100.0); per seed 80.4 · 100.0 · 80.4 | addition 100.0% (100.0–100.0) |
| Grader wrong 10% of the time | Mix-RL | 63.1% (62.0–64.1) | 65.6% (64.5–66.8) |
| | Cascade-RL | 18.4% (10.8–29.3) | 50.8% (43.5–55.0) |
| | MOPD → RL | 77.1% (76.5–77.8) | 78.0% (77.4–78.9) |

<figure class="rl-chart" data-rl-bars="real"></figure>
<p class="rl-caption">Figure 6-2. Baseline condition versus the bug teacher (addition accuracy) and the 10%-wrong grader (overall accuracy). Mean of three seeds.</p>

- **RL-polished teachers: the student ends up the same.** The teacher's division
  is 90.5%, below the true-answer SFT teacher's 97.1%, but the student's
  division is around 70%, so the teacher difference only exists above the
  student's ceiling.
- **Bug teacher: distillation copies the mistake.** The starting student was
  at 99.9–100% on addition, and both MOPD and SFT distillation brought it down
  to 80.4%, exactly the bug teacher's score. The teacher's mistake overwrote
  something the student already knew.
- **Bug teacher + RL: only one seed of three recovered.** For seed 1, addition
  was back to 99.8% within 1,000 RL steps and ended at 100%; seeds 0 and 2
  stayed at 80.4% to the end. Once the student writes the buggy answer
  confidently, all 8 answers are identically wrong and there is no signal. It
  only gets fixed when an answer happens to get the carry right. Same reason as
  experiments 2 and 5.
- **A sometimes-wrong grader: Cascade-RL collapsed.** During the addition and
  subtraction stages (already ~100%), "problems with something to learn" showed
  up as 57%. With a clean grader it was 0.7–2.1%, so nearly all of it was fake
  signal. In the multiplication stage, one mostly-wrong answer flipped to a pass
  gets pushed hard with +2.47, and around steps 5,250–6,000 even addition
  collapsed below 50%. Mix-RL didn't collapse, but multiplication and division
  ended below the starting point. MOPD → RL mostly held because of what it had
  already learned from OPD.

### RL learning rate: at 3e-4 the model falls apart

At first I ran RL with the same learning rate as OPD, 3e-4. A multiplication
teacher that was at 99.9% after SFT dropped into the 80s within 1,000 RL steps.
Only a few percent of problems have anything to learn, so the RL gradient is
small and noisy, and Adam normalizes the step size per parameter, which turns
that noise into steps of size 3e-4 as well. OPD's signal is dense (every
character), so the same learning rate was fine there.

<figure class="rl-chart" data-rl-bars="lr"></figure>
<p class="rl-caption">Figure 6-3. Accuracy after RL with learning rate 3e-4 versus 3e-5. Teachers: after 20,000 RL steps that followed 10,000 SFT steps (SFT ended at 99.9% for multiplication and 91.8% for division). Students: seed 0.</p>

| Accuracy % | RL learning rate 3e-4 | RL learning rate 3e-5 |
| --- | --- | --- |
| Multiplication teacher: end of SFT → after 20,000 RL steps | 99.9 → 77.5 | 99.9 → 99.6 |
| Division teacher: end of SFT → after 20,000 RL steps | 91.8 → 58.3 | 91.8 → 90.5 |
| Mix-RL student (seed 0): × · ÷ | 14.9 · 18.3 | 32.3 · 35.0 |
| Cascade-RL student (seed 0): + · − | 0.1 · 2.3 | 75.3 · 82.8 |
| Short sweep: Mix-RL multiplication after 1,200 steps (start 30.3) | 16.3 (1e-4: 26.9) | 29.5 (1e-5: 30.5) |

<p class="rl-caption">The last row is from the <code>big/rl_lr_test.py</code> log (seed 0 starting student, 1,200 steps).</p>

Lowering just the learning rate to 3e-5 stopped the collapse. Every RL run in
the results above was re-run at 3e-5. It's the same reason real LLM RL uses
learning rates around 1e-6, far smaller than SFT.

<div class="rl-callout is-decision" markdown="1">
<span class="rl-callout__title">Does MOPD pull far ahead? Not in this setup</span>
MOPD's advantage grows under three conditions: ① no single teacher is good at
every subject (experiment 4; in practice, teachers RL-trained per subject such
as math, code and agents have different strengths), ② answers are long enough
that a student's mistakes propagate, and ③ the student is big enough to get
close to the teacher. All three were weak in experiment 6. The next things to
scale are longer answers with written-out reasoning (CoT), a bigger student, or
a smaller four-operation teacher so the subjects compete.
</div>

<div class="rl-callout is-warn" markdown="1">
<span class="rl-callout__title">Limits of this experiment</span>
- The student is small (3.2M) and 10,000 steps is not converged. Longer runs
  could widen or narrow the gaps between methods.
- The format writes only the answer, with no worked steps, so it's unlike real
  long LLM reasoning.
- Each teacher was made once, with one seed. Only the students have three seeds.
- RL runs started at 3e-4 and were switched to 3e-5 and re-run partway through.
  The 3e-4 results are used only in the learning-rate comparison above.
</div>

## What carries over to real LLM training, and what doesn't {#real}

This is a toy for seeing the concepts. Take the direction, not the numbers.

| Item | This experiment | Real LLMs |
| --- | --- | --- |
| Answer length | five characters | hundreds to tens of thousands of reasoning tokens |
| SFT distillation vs OPD | about the same (SFT slightly faster): no room to continue from one's own mistakes | OPD is widely reported to do better on long reasoning, because the student learns in the mistaken states it actually produces |
| Safeguards | no KL constraint, one fixed learning rate | KL constraints, low learning rates, data mixing |
| Most expensive method | SFT (the teacher writes every step); RLVR is cheapest with only a grader | RL: many long generations per problem; SFT data is generated once up front |
| OPD requirements | teacher and student share the 14-character vocabulary | same tokenizer, and access to the teacher's per-token log-probabilities |
| Grader | exact character match | math: answer equivalence; code: tests pass; writing: rubric + LLM judge |

## Code walkthrough: rlsim.py, line by line {#code}

[`rlsim.py`]({{ '/files/rl-toy/rlsim.py' | relative_url }}) is a single
364-line file using only PyTorch and NumPy. The snippets below are cut straight
from it, and the captions give the file's line numbers. Tensor shapes and values
were extracted by running the same code.

### 0. Map of the file {#code-map}

<div class="rl-flow">
  <div class="rl-flow__step"><b>main()</b><span>parse arguments</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step"><b>load_or_train_teachers()</b><span>build (or load) the three teachers</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step"><b>experiment()</b><span>one experiment = starting student × methods × 3 seeds</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step is-here"><b>post_train()</b><span>each step: step_sft / step_rlvr / step_opd</span></div>
</div>
<p class="rl-caption">Figure 7. Call order. The methods split inside the blue box.</p>

| Lines | Functions | What they do |
| --- | --- | --- |
| 30–38 | constants | vocabulary, lengths, batch, learning rate |
| 45–66 | `problems` · `mixed` · `correct` | generate problems, mix subjects, grade |
| 72–118 | `TinyGPT` · `answer_logits` · `write` · `accuracy` | model, answer-position predictions, writing, accuracy |
| 124–174 | `update` · `step_sft` · `teacher_tokens` · `step_rlvr` · `step_opd` | one step of each method |
| 180–220 | `sft_until` · `post_train` | SFT to a target, the training loop |
| 226–347 | `test_sets` · `load_or_train_teachers` · `warm_student` · `experiment` | evaluation sets, teachers, starting student, experiment definitions |

### 1. Characters and problems: `VOCAB`, `problems()`, `mixed()` {#code-data}

```python
VOCAB = list("0123456789+-=$")  # "$" = end of answer
STOI = {c: i for i, c in enumerate(VOCAB)}
PROMPT_LEN, ANSWER_LEN = 8, 5  # after "123+045=": four digits ones-first, then "$"
SEQ_LEN = PROMPT_LEN + ANSWER_LEN
BATCH = 256  # answers the student writes per step; identical for every method
GROUP = 8  # RLVR: answers per problem (32 problems x 8)
LR = 5e-4
LOG_EVERY = 20
WARM_TARGET = 0.30  # student starting point: 30% accuracy (mimics pre-training + a little SFT)
```

<p class="rl-caption">rlsim.py, lines 30–38</p>

```python
def problems(n: int, op: str, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """n problems (prompts) and answers as token ids. Subtraction always subtracts the smaller number from the larger."""
    a, b = rng.integers(0, 1000, n), rng.integers(0, 1000, n)
    if op == "-":
        a, b = np.maximum(a, b), np.minimum(a, b)
    c = a + b if op == "+" else a - b
    prompts = [[STOI[ch] for ch in f"{x:03d}{op}{y:03d}="] for x, y in zip(a, b)]
    answers = [[STOI[ch] for ch in f"{z:04d}"[::-1] + "$"] for z in c]
    return torch.tensor(prompts), torch.tensor(answers)


def mixed(n: int, ops: list[str], rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """n problems split evenly across subjects. The third value is each row's subject."""
    per = n // len(ops)
    parts = [problems(per, op, rng) for op in ops]
    subjects = [op for op in ops for _ in range(per)]
    return torch.cat([p for p, _ in parts]), torch.cat([a for _, a in parts]), subjects
```

<p class="rl-caption">rlsim.py, lines 45–61</p>

- `VOCAB` has 14 characters and `STOI` maps character → id: `0`–`9` are 0–9,
  `+` 10, `-` 11, `=` 12, `$` 13.
- `f"{x:03d}{op}{y:03d}="` pads to three digits: 45 + 7 becomes `045+007=`, so
  a problem is always 8 characters.
- `f"{z:04d}"[::-1] + "$"` pads to four digits and reverses, so an answer is
  always 5 characters.
- Subtraction puts the larger number first with `np.maximum` / `np.minimum`.
- `mixed()` splits n across subjects, addition in the first half and
  subtraction in the second. The third value, `subjects`, is each row's subject
  tag; MOPD uses it to pick a teacher.

```text
699+684=  →  [6, 9, 9, 10, 6, 8, 4, 12]   one row of prompts (8)
1383      →  reverse "1383" → "3831" + "$"
          →  [3, 8, 3, 1, 13]              one row of answers (5)
```

### 2. The verifier: `correct()` {#code-verifier}

```python
def correct(seq: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    """Verifier: 1 if the written answer matches the reference character for character, else 0."""
    return (seq[:, PROMPT_LEN:] == answers).all(1).float()
```

<p class="rl-caption">rlsim.py, lines 64–66</p>

```text
seq                   (256, 13)  8 prompt positions + 5 answer positions
seq[:, PROMPT_LEN:]   (256, 5)   the 5 answer positions
== answers            (256, 5)   True / False per position
.all(1)               (256,)     are all five True?
.float()              (256,)     1.0 or 0.0
```

The end marker `$` is compared too: four correct digits followed by anything
other than `$` score 0.

### 3. The model: `TinyGPT` {#code-model}

```python
class TinyGPT(nn.Module):
    def __init__(self, layers: int, width: int, heads: int = 4):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), width)
        self.pos = nn.Embedding(SEQ_LEN, width)
        block = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, len(VOCAB))
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(SEQ_LEN), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.tok(x) + self.pos.weight[:t]
        h = self.blocks(h, mask=self.mask[:t, :t], is_causal=True)
        return self.head(self.norm(h))


def student() -> TinyGPT:
    return TinyGPT(layers=2, width=64)


def teacher() -> TinyGPT:
    return TinyGPT(layers=4, width=128)
```

<p class="rl-caption">rlsim.py, lines 72–97</p>

- `self.tok` maps a character id to a 64-dim vector (token embedding);
  `self.pos` is a learned vector for each of positions 0–12 (position
  embedding). Their sum is the input.
- `nn.TransformerEncoderLayer` with a causal mask
  (`generate_square_subsequent_mask`) lets each position see only itself and
  earlier positions. Despite the "Encoder" name, this makes it a GPT-style
  decoder-only model.
- `norm_first=True` places LayerNorm before attention and the MLP (pre-LN, the
  standard since GPT-2). Dropout is 0.
- `self.head` outputs 14 scores (logits) per position: shape (batch, positions,
  14).
- Student and teacher are the same class at different sizes. Sharing the
  vocabulary lets the teacher's probabilities be compared with the student's
  position by position and character by character, which is what makes OPD
  possible.

| Part | Student (2 layers, width 64) | Teacher (4 layers, width 128) | Student formula |
| --- | ---: | ---: | --- |
| token embedding | 896 | 1,792 | 14 × 64 |
| position embedding | 832 | 1,664 | 13 × 64 |
| attention | 33,280 | 264,192 | 2 × (64×192+192 + 64×64+64) |
| MLP | 66,176 | 526,848 | 2 × (64×256+256 + 256×64+64) |
| LayerNorm (in layers) | 512 | 2,048 | 2 × 2 × (64+64) |
| final LayerNorm | 128 | 256 | 64 + 64 |
| output head | 910 | 1,806 | 64 × 14 + 14 |
| **total** | **102,734** | **798,606** | |

### 4. Picking out the answer predictions: `answer_logits()` {#code-slice}

```python
def answer_logits(model: TinyGPT, seq: torch.Tensor) -> torch.Tensor:
    """The prediction made just before each of the five answer characters (batch, 5, vocab)."""
    return model(seq)[:, PROMPT_LEN - 1 : -1]
```

<p class="rl-caption">rlsim.py, lines 100–102</p>

A GPT's output at position i predicts the character at position i+1. So the
prediction for the first answer character (position 8) lives in the output at
position 7 (`=`). The one-position shift is the key.

```text
  0  1  2  3  4  5  6  7  8  9 10 11 12   ← position
  6  9  9  +  6  8  4  =  3  8  3  1  $   ← input character
  9  9  +  6  8  4  =  3  8  3  1  $  -   ← what this position's output predicts
                       ^  ^  ^  ^  ^      ← [:, PROMPT_LEN-1 : -1] = positions 7–11
```

`model(seq)` is (batch, 13, 14), and slicing `[:, 7:12]` gives (batch, 5, 14).
Position 12 (after `$`) has nothing to predict and is dropped. SFT, RLVR and OPD
all get "the student's probabilities at the five answer positions" from this one
function.

### 5. Writing an answer: `write()` {#code-write}

```python
@torch.no_grad()
def write(model: TinyGPT, prompts: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    """The model writes the answer one character at a time. greedy=False samples from its probabilities (temperature 1)."""
    seq = prompts
    for _ in range(ANSWER_LEN):
        logits = model(seq)[:, -1]
        nxt = logits.argmax(-1) if greedy else torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return seq


def accuracy(model: TinyGPT, test: tuple[torch.Tensor, torch.Tensor]) -> float:
    prompts, answers = test
    return correct(write(model, prompts, greedy=True), answers).mean().item()
```

<p class="rl-caption">rlsim.py, lines 105–118</p>

- **`@torch.no_grad()`:** no gradients are recorded while writing. The
  probabilities needed for training are computed by **reading the finished
  answer again** with `answer_logits()`. Real RL frameworks do the same: an
  inference engine (vLLM, etc.) generates rollouts and the trainer re-reads them
  to compute probabilities.
- **Five iterations:** each feeds the whole `seq` so far and uses only the last
  position's logits (`[:, -1]`). There is no KV cache, so the prefix is
  recomputed every time, which is fine at 13 positions.
- **`greedy=True` → `argmax`:** always the top character. Used for evaluation and
  for teacher answers.
- **`greedy=False` → `torch.multinomial(softmax)`:** sample by probability
  (temperature 1). Used for student answers in RLVR and OPD. A character the
  student believes at 30% appears 30% of the time, which is why eight attempts
  at the same problem disagree.

### 6. The update: `update()` {#code-update}

```python
def update(opt: torch.optim.Optimizer, model: TinyGPT, loss: torch.Tensor) -> None:
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

<p class="rl-caption">rlsim.py, lines 124–128</p>

All three methods move parameters with these four lines; only the loss differs.
If the total gradient norm exceeds 1 it is scaled down to 1, so no single step
jumps too far.

### 7. One step of SFT distillation: `step_sft()`, `teacher_tokens()` {#code-sft}

```python
def step_sft(model: TinyGPT, opt, prompts: torch.Tensor, targets: torch.Tensor) -> dict:
    """Copy the given answer character by character (cross-entropy)."""
    seq = torch.cat([prompts, targets], 1)
    loss = F.cross_entropy(answer_logits(model, seq).reshape(-1, len(VOCAB)), targets.reshape(-1))
    update(opt, model, loss)
    return {}


def teacher_tokens(prompts: torch.Tensor, subjects: list[str], teachers: dict[str, TinyGPT]) -> torch.Tensor:
    """For each subject, the model answer written by that subject's teacher (greedy)."""
    targets = torch.empty(len(prompts), ANSWER_LEN, dtype=torch.long)
    for op, model in teachers.items():
        rows = torch.tensor([s == op for s in subjects])
        if rows.any():
            targets[rows] = write(model, prompts[rows], greedy=True)[:, PROMPT_LEN:]
    return targets
```

<p class="rl-caption">rlsim.py, lines 131–146</p>

| Line | Code | Shape | Meaning |
| --- | --- | ---: | --- |
| 213 | `mixed(BATCH, ops, rng)` | <span class="ans">(256, 8)</span> | 256 problems × 8 characters |
| 145 | `teacher_tokens(prompts, subjects, teachers)` | <span class="ans">(256, 5)</span> | 256 teacher answers × 5 characters |
| 133 | `torch.cat([prompts, targets], 1)` | <span class="ans">(256, 13)</span> | the problem followed by the teacher's answer |
| 134 | `answer_logits(model, seq)` | <span class="ans">(256, 5, 14)</span> | 5 answer positions × 14 character scores |
| 134 | `.reshape(-1, len(VOCAB))` | <span class="ans">(1,280, 14)</span> | 1,280 characters flattened for `cross_entropy` |

- **Line 143, `rows`:** a mask that is True for rows of this teacher's subject.
  Only those rows are written by this teacher (greedy). With two teachers the
  rows are split and written in two passes.
- **Line 133, `torch.cat`:** the teacher's answer is appended to the problem, and
  the student predicts each next character while reading it. The prefix is always
  the teacher's (teacher forcing), so the student never actually writes; it never
  gets to learn how to continue after its own mistake.
- **Line 134, `F.cross_entropy`:** the mean of −log p(teacher's character) over
  256 × 5 = 1,280 characters.
- **No true answers:** line 215 of `post_train()` passes
  `teacher_tokens(...)` instead of `answers`.

### 8. One step of RLVR: `step_rlvr()` {#code-rlvr}

```python
def step_rlvr(model: TinyGPT, opt, prompts: torch.Tensor, answers: torch.Tensor) -> dict:
    """GRPO: write 8 answers per problem; raise the probability of the characters in correct answers, lower it in wrong ones.
    If all 8 are right or all 8 are wrong there is nothing to compare, so that problem teaches nothing."""
    prompts, answers = prompts.repeat_interleave(GROUP, 0), answers.repeat_interleave(GROUP, 0)
    seq = write(model, prompts)
    reward = correct(seq, answers).view(-1, GROUP)
    advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
    logp = answer_logits(model, seq).log_softmax(-1).gather(-1, seq[:, PROMPT_LEN:, None]).squeeze(-1)
    update(opt, model, -(advantage.view(-1, 1) * logp).mean())
    return {"signal": (reward.std(1) > 0).float().mean().item(), "reward": reward.mean().item()}
```

<p class="rl-caption">rlsim.py, lines 149–158</p>

| Line | Code | Shape | Meaning |
| --- | --- | ---: | --- |
| 210 | `mixed(BATCH // GROUP, ops, rng)` | <span class="ans">(32, 8)</span> | 32 problems |
| 152 | `prompts.repeat_interleave(GROUP, 0)` | <span class="ans">(256, 8)</span> | 8 copies of each problem, 256 rows |
| 153 | `write(model, prompts)` | <span class="ans">(256, 13)</span> | 256 answers sampled by the student |
| 154 | `correct(seq, answers).view(-1, GROUP)` | <span class="ans">(32, 8)</span> | row = problem, column = attempt, value 0/1 |
| 155 | `advantage` | <span class="ans">(32, 8)</span> | per problem: (score − mean) ÷ std |
| 156 | `….gather(-1, written).squeeze(-1)` | <span class="ans">(256, 5)</span> | log p of the 5 characters each answer wrote |
| 157 | `advantage.view(-1, 1)` | <span class="ans">(256, 1)</span> | multiplied by logp, spread equally over 5 characters |

**Line 152, `repeat_interleave`: eight copies of the same problem side by side.**
Easy to confuse with `repeat`:

```text
problems           [A, B, C]
repeat_interleave  [A, A, B, B, C, C]   ← this one
repeat             [A, B, C, A, B, C]
```

Line 154's `.view(-1, 8)` groups 8 consecutive rows as one problem, so this
layout is what makes "one row = eight attempts at one problem". With `repeat`,
answers to different problems would be averaged together: a bug that fails
silently, without an error.

**Line 155, advantage: mean and std per problem.** `mean(1)` and `std(1)` run
across the eight attempts (columns). PyTorch's `std` divides by n − 1 by
default. With 2 of 8 right, as in 227 + 189, the mean is 0.25 and the std 0.463,
so right answers get +1.62 and wrong ones −0.54; dividing by n would give a std
of 0.433 and +1.73 / −0.58. If all eight are equal the std is 0; `1e-6` avoids
dividing by zero, and the numerator is 0 too, so the advantage is exactly 0.

**Line 156, logp: pick out the log-probability of each written character.**
`log_softmax(-1)` gives 14 log-probabilities per position, (256, 5, 14), and
`gather` keeps only the character actually written: (256, 5).

**Line 157, the loss.** Multiplying the advantage (256, 1) by logp (256, 5)
applies one answer's advantage equally to its five characters. The minus sign is
there because the optimizer *reduces* the loss: reducing it means increasing
advantage × log p, so characters in answers with positive advantage gain
probability and those with negative advantage lose it.

That is the policy gradient. The log-probability of a whole answer is the sum of
its five characters' log-probabilities. The gradient of the expected score is the
average of score × ∇log p, and subtracting the group's mean score (a baseline)
leaves its expectation unchanged while reducing noise.

**Real values: the eight answers to 227 + 189**

| Answer | Written (ones first) | Grade · advantage | Per-position log p (ones · tens · hundreds · thousands · $) |
| --- | --- | --- | --- |
| <span class="ans"><span class="bad">5</span>16</span> | <span class="ans">6 1 5 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−2.09</b> −0.00 −0.00</span> |
| <span class="ans">416</span> | <span class="ans">6 1 4 0 $</span> | <span class="mark-ok">O</span> +1.62 | <span class="lp">−0.01 −0.13 <b>−1.56</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">6</span>16</span> | <span class="ans">6 1 6 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−2.16</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">2</span>16</span> | <span class="ans">6 1 2 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−3.34</b> −0.00 −0.00</span> |
| <span class="ans">416</span> | <span class="ans">6 1 4 0 $</span> | <span class="mark-ok">O</span> +1.62 | <span class="lp">−0.01 −0.13 <b>−1.56</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">3</span>16</span> | <span class="ans">6 1 3 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−2.65</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">8</span>16</span> | <span class="ans">6 1 8 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−1.65</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">9</span><span class="bad">0</span>6</span> | <span class="ans">6 0 9 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 <b>−2.31</b> <b>−1.77</b> −0.00 −0.00</span> |

A log-probability of −0.00 is a character the student is already almost 100%
sure of. The gradient of log p is proportional to (1 − p), so such positions
barely move however large the advantage. What moves is the hundreds place, where
the student hesitated.

**Where the signal survives once the eight are summed.** Earlier I said the wrong
516 drags its right digits down too. For one answer in isolation, that's true.
But the loss sums all eight, and answers that wrote **the same character after
the same prefix** add their advantages, because the same prefix means the same
student probability.

| Position | Prefix (as written) | Character × count | Sum of advantages |
| --- | --- | --- | ---: |
| ones | — | <span class="ans">6</span> × 8 | 0.00 |
| tens | <span class="ans">6</span> | <span class="ans">1</span> × 7 | **+0.54** |
| | <span class="ans">6</span> | <span class="ans">0</span> × 1 | −0.54 |
| hundreds | <span class="ans">6 1</span> | <span class="ans">4</span> × 2 | **+3.24** |
| | <span class="ans">6 1</span> | <span class="ans">2</span> × 1 | −0.54 |
| | <span class="ans">6 1</span> | <span class="ans">3</span> × 1 | −0.54 |
| | <span class="ans">6 1</span> | <span class="ans">5</span> × 1 | −0.54 |
| | <span class="ans">6 1</span> | <span class="ans">6</span> × 1 | −0.54 |
| | <span class="ans">6 1</span> | <span class="ans">8</span> × 1 | −0.54 |
| | <span class="ans">6 0</span> | <span class="ans">9</span> × 1 | −0.54 |

- **Ones:** all eight wrote 6. +1.62 × 2 + (−0.54) × 6 = 0: exact cancellation.
  Advantages always sum to 0, so a character everyone wrote gets no signal.
- **Tens:** seven wrote 1 and only 906 wrote 0. 1 gets +0.54 and 0 gets −0.54.
- **Hundreds (the seven with prefix "61"):** 4 gets +3.24 and each of the other
  five digits −0.54. The signal concentrates where the answers diverge.
- **Thousands and end marker:** prefixes differ between answers so nothing
  cancels, but the student is already almost 100% sure, so they barely move.

So even a single pass/fail per answer, compared across several answers, reveals
to some extent which position was the problem. But when all eight are wrong, as
for 699 + 684, there is nothing to compare and this is 0 too.

**Line 158, signal:** the share of problems whose 8 scores have a nonzero std
(i.e. were not all the same). That is the "problems with signal" in figure 3.
It is only logged, never used for training.

### 9. One step of OPD and MOPD: `step_opd()` {#code-opd}

```python
def step_opd(model: TinyGPT, opt, prompts: torch.Tensor, subjects: list[str], teachers: dict[str, TinyGPT]) -> dict:
    """OPD: the student writes; keeping the student's own prefix, shrink the gap between the teacher's and the student's
    distribution at every character (reverse KL over the whole vocabulary). A different teacher per subject makes it MOPD."""
    seq = write(model, prompts)
    s = answer_logits(model, seq).log_softmax(-1)
    t = torch.empty_like(s)
    with torch.no_grad():
        for op, model_t in teachers.items():
            rows = torch.tensor([x == op for x in subjects])
            if rows.any():
                t[rows] = answer_logits(model_t, seq[rows]).log_softmax(-1)
    kl = (s.exp() * (s - t)).sum(-1)
    update(opt, model, kl.mean())
    return {"signal": 1.0, "kl": kl.mean().item()}
```

<p class="rl-caption">rlsim.py, lines 161–174</p>

| Line | Code | Shape | Meaning |
| --- | --- | ---: | --- |
| 213 | `mixed(BATCH, ops, rng)` | <span class="ans">(256, 8)</span> | 256 problems |
| 164 | `write(model, prompts)` | <span class="ans">(256, 13)</span> | 256 answers sampled by the student |
| 165 | `answer_logits(model, seq).log_softmax(-1)` | <span class="ans">(256, 5, 14)</span> | student log-probabilities per position (with gradient) |
| 171 | `answer_logits(model_t, seq[rows]).log_softmax(-1)` | <span class="ans">(256, 5, 14)</span> | the teacher reading the same answers (no gradient) |
| 172 | `(s.exp() * (s - t)).sum(-1)` | <span class="ans">(256, 5)</span> | one reverse KL per position, summed over 14 characters |
| 173 | `kl.mean()` | <span class="ans">scalar</span> | mean over 1,280 positions = loss |

- **Line 164:** the student samples one answer per problem; no gradient recorded.
- **Line 165:** the student's log-probabilities on re-reading its own answer.
  This is where the gradient flows.
- **Lines 166–171:** the teacher's log-probabilities when reading **the
  student's answer**. It only reads, so all five positions come out in one pass.
  Routing rows to a different teacher per subject with the `rows` mask makes it
  MOPD.
- **Line 172:** reverse KL = Σ over 14 characters of p_student ×
  (log p_student − log p_teacher). `s.exp()` is the student's probability and
  `s - t` the log-ratio. This is the exact sum over all 14 characters, not a
  one-sample estimate.
- **Lines 173–174:** the mean over 256 × 5 positions is the loss. Every position
  gets a signal, so `signal` is always 1.

OPD, MOPD and experiment 4's control are the same function with a different
`teachers` map:

```text
OPD   (exp. 1, 2)   {"+": add}
OPD   (exp. 3)      {"+": add_weak}
MOPD  (exp. 4)      {"+": add, "-": sub}
control (exp. 4)    {"+": add, "-": add}   # subtraction problems read by the addition teacher too
```

**The shape of the gradient.** Differentiating by hand, the gradient of one
position's reverse KL with respect to its logit z_c is below. It matched
autograd to within 1.2 × 10⁻⁷.

```text
∂KL/∂z_c = p_student(c) × ( log p_student(c) − log p_teacher(c) − KL )
```

Gradient descent moves against the gradient, so in logit space **characters whose
log-ratio exceeds that position's KL go down, and those below it go up.** SFT's
gradient is p_student(c) − [c is the teacher's character]: it raises only the
teacher's character and lowers the rest in proportion to the student's
probability.

**Real values: the hundreds place of 699 + 684** (prefix 3, 8). Nudging the logits
a tiny step along the gradient, here is the share of probability leaving (−) and
arriving at (+) each character. The four non-digit characters are at 0% and
omitted.

| Character | Student → teacher probability | Share moved by SFT | Share moved by OPD |
| --- | --- | ---: | ---: |
| <span class="ans">0</span> | 22.4% → < 0.001% | <span class="down">−18%</span> | <span class="up">+2.5%</span> |
| <span class="ans">1</span> | 35.8% → < 0.001% | <span class="down">**−69%**</span> | <span class="down">**−100%**</span> |
| <span class="ans">2</span> | 20.3% → 0.004% | <span class="down">−13%</span> | <span class="up">**+34%**</span> |
| <span class="ans">3</span> | 10.6% → 99.2% | <span class="up">**+91%**</span> | <span class="up">**+47%**</span> |
| <span class="ans">4</span> | 2.3% → 0.7% | <span class="up">+2.0%</span> | <span class="up">+3.9%</span> |
| <span class="ans">5</span> | 1.7% → 0.002% | <span class="up">+1.5%</span> | <span class="up">+2.2%</span> |
| <span class="ans">6</span> | 0.7% → 0.003% | <span class="up">+0.7%</span> | <span class="up">+0.8%</span> |
| <span class="ans">7</span> | 0.9% → < 0.001% | <span class="up">+0.9%</span> | <span class="up">+1.2%</span> |
| <span class="ans">8</span> | 1.1% → 0.006% | <span class="up">+1.0%</span> | <span class="up">+1.4%</span> |
| <span class="ans">9</span> | 4.2% → 0.016% | <span class="up">+3.0%</span> | <span class="up">+6.9%</span> |

- **SFT:** 91% of the incoming probability goes to the correct 3. The outflow is
  proportional to the student's probability: 1 (69%), 0 (18%), 2 (13%). The other
  9% goes in small amounts to 4–9, which had little probability. Their logits go
  down too, but 1, 0 and 2 go down much more, so in probability they rise a
  little. Probability is shared among characters, so it is always relative.
- **OPD:** the outflow comes **entirely from 1**, the character the student
  actually wrote; the teacher gives it almost 0, so its log-ratio (13.75) is the
  largest. On the incoming side 3 gets the most (47%), but **2 rises too (34%)**.
  0's logit goes down, but 1 goes down so much more that 0's probability barely
  changes (+2.5%).

Why does 2, which the teacher gives only 0.004%, go up? Its log-ratio (8.53) is
below this position's KL (9.48). OPD is moving probability off 1, the biggest
problem right now, and 2, which already has a lot, takes a big share. Once 1
shrinks and the KL drops, 2 flips to going down. Below is a thought experiment
moving only this position's 14 logits directly, step size 0.5.

| Step | SFT: probability of 1 · 2 · 3 (%) | OPD: probability of 1 · 2 · 3 (%) |
| --- | --- | --- |
| 0 | <span class="ans">36 · 20 · 11</span> | <span class="ans">36 · 20 · 11</span> |
| 1 | <span class="ans">31 · 19 · 17</span> | <span class="ans">19 · 25 · 22</span> |
| 2 | <span class="ans">27 · 17 · 26</span> | <span class="ans">9 · 18 · 50</span> |
| 3 | <span class="ans">22 · 15 · 36</span> | <span class="ans">3 · 6 · 81</span> |
| 4 | <span class="ans">18 · 13 · 46</span> | <span class="ans">2 · 3 · 90</span> |
| 5 | <span class="ans">15 · 11 · 54</span> | <span class="ans">1 · 2 · 93</span> |

OPD clears 1 out first (36% → 19%) while 2 briefly grows (20% → 25%); from the
second step 2 shrinks too and 3 rises. SFT raises 3 steadily from the start. The
two losses have different gradient scales, so compare only the direction, not the
speed per step. Real training also moves parameters with Adam, so logits don't
move independently like this.

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">Real-LLM OPD differs at line 172</span>
With a vocabulary of around 150,000 tokens, summing over every token at every
position is expensive. Some implementations instead use the log-probability
difference of the single token the student actually wrote
(log p_student − log p_teacher) as that token's penalty and train it with a
policy gradient like RLVR (Thinking Machines' 2025 write-up on on-policy
distillation takes this approach). Then only the written token's log-probability
is needed from the teacher. Either way, teacher and student must share a
tokenizer.
</div>

### 10. The training loop: `post_train()` {#code-loop}

```python
def post_train(method: str, model: TinyGPT, plan: list[tuple[int, list[str]]], teachers: dict, tests: dict, seed: int) -> dict:
    """plan = [(steps, subjects), ...]; two stages make a cascade (finish one subject, then the next)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    curve, signals, step, started = [], [], 0, time.time()
    schedule = [ops for steps, ops in plan for _ in range(steps)]
    for step in range(len(schedule) + 1):
        if step % LOG_EVERY == 0:
            point = {"step": step, **{op: accuracy(model, tests[op]) for op in tests}}
            point["signal"] = float(np.mean(signals)) if signals else None
            curve.append(point)
            signals = []
        if step == len(schedule):
            break
        ops = schedule[step]
        if method == "rlvr":
            prompts, answers, _ = mixed(BATCH // GROUP, ops, rng)
            stats = step_rlvr(model, opt, prompts, answers)
        else:
            prompts, answers, subjects = mixed(BATCH, ops, rng)
            if method == "sft":
                stats = step_sft(model, opt, prompts, teacher_tokens(prompts, subjects, teachers))
            else:
                stats = step_opd(model, opt, prompts, subjects, teachers)
        if "signal" in stats:
            signals.append(stats["signal"])
    return {"curve": curve, "seconds": round(time.time() - started, 1)}
```

<p class="rl-caption">rlsim.py, lines 193–220</p>

- **Line 199:** expands `plan` into a per-step subject list. `[(1000, ["+", "-"])]`
  is 1,000 entries of `["+", "-"]`; Cascade's `[(500, ["+"]), (500, ["-"])]` is
  500 of `["+"]` followed by 500 of `["-"]`. This one line is the whole difference
  between Mix-RL and Cascade-RL.
- **Lines 201–205:** every 20 steps (including step 0 and the last), measure
  greedy accuracy per subject and record the mean signal collected since. Each is
  one point on a chart: 51 points for 1,000 steps.
- **Lines 209–211:** RLVR draws `BATCH // GROUP` = 32 problems (16 per subject
  with two subjects).
- **Lines 213–217:** the others draw 256. SFT builds teacher answers for
  `step_sft`; everything else goes to `step_opd`. OPD and MOPD share the line.

### 11. Experiment definitions: `experiment()` {#code-exp}

```python
    if name == "multi":
        runs = {
            "mix-rlvr": ("rlvr", [(steps, ["+", "-"])], {}),
            "cascade-rlvr": ("rlvr", [(steps // 2, ["+"]), (steps // 2, ["-"])], {}),
            "mopd": ("opd", [(steps, ["+", "-"])], {"+": add, "-": sub}),
            "opd-add-teacher-only": ("opd", [(steps, ["+", "-"])], {"+": add, "-": add}),
        }
    else:
        runs = {
            "single": {
                "sft": ("sft", [(steps, ["+"])], {"+": add}),
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd": ("opd", [(steps, ["+"])], {"+": add}),
            },
            "cold": {
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd": ("opd", [(steps, ["+"])], {"+": add}),
            },
            "weak": {
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd-weak": ("opd", [(steps, ["+"])], {"+": weak}),
                "sft-weak": ("sft", [(steps, ["+"])], {"+": weak}),
            },
        }[name]
```

<p class="rl-caption">rlsim.py, lines 303–326</p>

A run is three things: (method, plan, teacher map). This table is the entire
experimental design.

```python
    for seed in range(seeds):
        if name == "cold":
            torch.manual_seed(seed)
            start, info = student(), {"steps": 0, "+": accuracy(student(), tests["+"])}
        else:
            start, info = warm_student(ops, seed)
        result["warm"].append(info)
        if name == "single" and seed == 0:
            result["example"] = example_card(start, add)
        for run, (method, plan, run_teachers) in runs.items():
            out = post_train(method, copy.deepcopy(start), plan, run_teachers, tests, seed)
            result["runs"].setdefault(run, []).append(out)
```

<p class="rl-caption">rlsim.py, lines 331–342</p>

Each seed builds one starting student (a random model for experiment 2), and
every method gets its own `copy.deepcopy(start)` for `post_train()`. Methods
never share weights.

### 12. Teachers and the starting student: `sft_until()`, `load_or_train_teachers()`, `warm_student()` {#code-teachers}

```python
def sft_until(model: TinyGPT, ops: list[str], target: float, rng, val: dict, check_every: int, max_steps: int) -> dict:
    """SFT on the true answers until accuracy reaches target (used to build teachers and the student's starting point)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(max_steps + 1):
        if step % check_every == 0:
            accs = {op: accuracy(model, val[op]) for op in ops}
            if np.mean(list(accs.values())) >= target or step == max_steps:
                return {"steps": step, **accs}
        prompts, answers, _ = mixed(BATCH, ops, rng)
        step_sft(model, opt, prompts, answers)
    raise AssertionError
```

<p class="rl-caption">rlsim.py, lines 180–190</p>

```python
def load_or_train_teachers() -> dict[str, TinyGPT]:
    """An addition specialist, a subtraction specialist (each learns only its own subject), and an under-trained addition teacher (~70%)."""
    folder = OUT / "teachers"
    folder.mkdir(parents=True, exist_ok=True)
    specs = {"add": (["+"], 0.995, 1), "sub": (["-"], 0.995, 2), "add-weak": (["+"], 0.70, 3)}
    teachers, report = {}, {}
    for name, (ops, target, seed) in specs.items():
        torch.manual_seed(seed)
        model = teacher()
        path = folder / f"{name}.pt"
        if path.exists():
            model.load_state_dict(torch.load(path))
        else:
            started = time.time()
            info = sft_until(model, ops, target, np.random.default_rng(seed), val_sets(), 25 if target < 0.9 else 100, 30000)
            torch.save(model.state_dict(), path)
            print(f"teacher {name}: {info} in {time.time() - started:.0f}s", flush=True)
        teachers[name] = model
        report[name] = {op: accuracy(model, t) for op, t in test_sets().items()}
    (folder / "accuracy.json").write_text(json.dumps(report, indent=1))
    print("teacher test accuracy:", report, flush=True)
    return teachers


def warm_student(ops: list[str], seed: int) -> tuple[TinyGPT, dict]:
    torch.manual_seed(seed)
    model = student()
    info = sft_until(model, ops, WARM_TARGET, np.random.default_rng(1000 + seed), val_sets(), 5, 5000)
    return model, info
```

<p class="rl-caption">rlsim.py, lines 236–264</p>

- **`sft_until()`:** SFT on true answers (learning rate 0.001). Every
  `check_every` steps it is scored on 1,000 validation problems (seed 999) and
  stops at the target. The stopping set and the charts' evaluation set
  (seed 12345) are different.
- **The three teachers:** addition only to 99.5% (seed 1), subtraction only to
  99.5% (seed 2), and the weak one: addition only to 70% (seed 3), checked every
  25 steps, which first passed 70% at step 275. It scores 73% on the evaluation
  set.
- **Saving:** teachers are saved as `results/teachers/*.pt` and loaded instead of
  retrained on later runs.
- **`warm_student()`:** target 30%, checked every 5 steps. Weights are initialized
  with the experiment seed and problems drawn with 1000 + seed. Experiments 1 and
  3 call the same function with the same seeds, so their starting students are
  identical.

### 13. How these lines change in real LLM training code {#code-real}

| This code | Real LLM training (TRL, verl, …) | Why |
| --- | --- | --- |
| the loop in `write()` | an inference engine (vLLM, SGLang) generates with a KV cache, and the trained weights are synced to it every step | answers run to thousands of tokens; generation dominates wall time |
| one `answer_logits()` | log-probabilities from three models: current, the one that generated the rollouts (old), and a reference (ref) | needed for the probability ratio and KL below |
| `correct()` | math-equivalence checks (Math-Verify), code-test sandboxes, rubric + LLM judge | answers take many forms |
| line 155, advantage | the same GRPO formula; Dr. GRPO drops the std division, and DAPO discards groups with zero signal and resamples (dynamic sampling) | these target exactly the signal decay seen here |
| line 157, `-(A × logp).mean()` | several updates per rollout batch, so the ratio π / π_old is applied and clipped to 0.8–1.2 (PPO clip); a KL(π ‖ π_ref) penalty is sometimes added | here each batch is used for one update, so the ratio is always 1 |
| mean over 256 × 5 | answer lengths vary, so averaging per token vs per answer changes results (DAPO's token-level loss) | every answer here is 5 characters, so it makes no difference |
| line 172, sum over 14 characters | only the written token's log-probability difference, or the top few tokens | the vocabulary is around 150,000 |
| learning rate 5e-4 | around 1e-6 | large models need small, stable steps |

### 14. Known gaps {#code-gaps}

- **Evaluation problems may have appeared in training.** There are 1,000 × 1,000
  = 1 million addition problems. SFT and OPD draw 256,000 at random over 1,000
  steps, so a given evaluation problem appears at least once with probability
  22.6% (3.1% for RLVR's 32,000). A 100k-parameter model can't memorize a million
  problems, so accuracy still measures whether the rule was learned, but strictly
  these aren't "unseen" problems. Filtering evaluation problems out in
  `problems()` would fix it.
- **Experiment 2's recorded starting accuracy:** line 334 calls `student()` twice,
  so the recorded 0% comes from a second, different random model rather than the
  actual starting model. Both are random models at 0%, so the results are
  unaffected.
- **Subjects without a teacher:** `step_opd()` would compute with uninitialized
  rows (`torch.empty_like`) for any subject missing from `teachers`. Every subject
  had a teacher here, so it never happened. A fix would stop with an error when a
  subject is missing.
- **No safeguards:** no KL(π ‖ π_ref) constraint, no replay of earlier subjects,
  no learning-rate schedule. This is why the cascade collapse may be exaggerated.

## Glossary {#terms}

| Term | Plain meaning | Here |
| --- | --- | --- |
| policy | the model being trained; it decides "which character to write here" | the student |
| rollout | an answer the model wrote itself | 256 per step |
| reward | the score of one answer | 1 if right, 0 if wrong |
| verifier | a program that grades answers mechanically | `correct()` |
| RLVR | RL on verifier scores (RL with Verifiable Rewards) | pass/fail grading |
| GRPO | RLVR that solves a problem several times and compares the answers | 8 per problem |
| advantage | how much better an answer is than the average for the same problem | (score − mean) ÷ std |
| on-policy | learning from answers the student wrote | RLVR, OPD, MOPD |
| off-policy | learning from answers someone else (the teacher) wrote | SFT distillation |
| distillation | transferring a large model's (teacher's) skill to a small one (student) | SFT distillation, OPD, MOPD |
| KL divergence | how different two probability distributions are; 0 if identical | OPD loss |
| reverse KL | KL measured where the student puts probability; betting on characters the teacher avoids is penalized hard | OPD loss |
| temperature | how adventurous sampling is; 1 means sample the probabilities as they are | 1 during training |
| greedy | always pick the most likely character | evaluation |
| step | one parameter update | one per 256 answers |
| seed | the starting point of the random numbers; change it and results shift a little | 3 repeats |
| logit | the pre-softmax score per character; softmax turns it into a probability | output of `model(seq)` |
| teacher forcing | feeding a fixed answer (the teacher's) as the prefix instead of the model's own characters | SFT distillation |
| behavior cloning | imitating demonstrations without rewards; SFT in RL vocabulary | SFT distillation |
| policy gradient | the gradient that raises the log-probability of answers that scored well | the `step_rlvr` loss |
| baseline | a reference value subtracted from scores; keeps the gradient's mean, reduces noise | mean score of the 8 answers |
| KV cache | storing the prefix's computation while generating and reusing it | not used (13 positions) |
| process reward | a score for each intermediate step rather than the whole answer | per-digit grading in exp. 5 |
| off-policy guidance | mixing answers written by someone else (teacher, truth) into RL rollout groups | answer injection in exp. 5 |
| curriculum | teaching easy problems first, then harder ones | easy-first in exp. 5 |
| cold start | starting from knowing nothing | experiment 2 |
| catastrophic forgetting | forgetting an old subject while learning a new one | Cascade-RL |
| see-saw effect | when subjects are mixed, one goes up as another goes down | Mix-RL |

## Run it yourself {#rerun}

Python 3, PyTorch and NumPy are enough; it takes about 35 minutes on a CPU.
Download [`rlsim.py`]({{ '/files/rl-toy/rlsim.py' | relative_url }}).

```bash
# train the three teachers → experiments 1–4 (1,000 steps, 3 seeds) → experiment 3 again at 3,000 steps
python3 rlsim.py all --steps 1000
python3 rlsim.py weak --steps 3000

# quick check (1 seed, 100 steps, under a minute)
python3 rlsim.py single --seeds 1 --steps 100
```

`rlsim.py` reproduces experiments 1–4, and results land in `results/*.json`.
Experiment 5, the per-digit traces and the charts came from small helper scripts
(`exp5.py`, `cold_digits.py`, `digit_order.py`, `trace.py`, `summary.py`,
`charts.py`) that build on `rlsim.py` and aren't published here. The core of
experiment 5 is the `step_rlvr_variant` function shown above. Experiment 6 is a
separate, larger script (`big/bigsim.py`, run with `big/orchestrate.py`; about
3 h 40 min on two H100s), also not published. Note that the CLI's default is
`--steps 500`; the experiments here used 1,000.

---

*A three-digit addition/subtraction toy, built to see the direction of real LLM
post-training with your own eyes. The numbers hold only for this setup.*

</div>

<div class="lang-block" lang="ko" markdown="1">

*장난감 실험이에요. CPU 만으로 약 35분, 실행마다 시드 3개예요. 전체 코드는
PyTorch 와 NumPy 만 쓰는 364줄짜리 파일 하나,
[`rlsim.py`]({{ '/files/rl-toy/rlsim.py' | relative_url }}) 예요.*

## 한눈에 보기 {#glance-ko}

실험은 여섯 개예요. 실험 1~5 는 모두 같은 작은 학생 모델로, 한 스텝에 답안
256장을 쓰는 같은 조건에서 비교했어요. 숫자는 평가용으로 따로 고정한 문제
2,000개에서 잰 정답률이에요(학습 문제와 따로 뽑지만 우연히 겹칠 수는 있어요.
[허술한 점](#code-gaps-ko)). 실험 6 은 GPU 에서 백만 단위 사칙연산으로 키운
거예요.

| 실험 | 조건 | 결과 (정답률) | 뜻 |
| --- | --- | --- | --- |
| 1. 34% 에서 출발 | 덧셈, 1,000스텝 | SFT 99.9% · OPD 98.6% · RLVR 81.0% | 글자마다 피드백이 O/X 보다 훨씬 빨라요 |
| 2. 0% 에서 출발 | 덧셈, 1,000스텝 | OPD 96.4% · RLVR 0.0% | O/X 채점은 가끔이라도 맞혀야 시작돼요 |
| 3. 약한 선생 (73%) | 덧셈, 3,000스텝 | RLVR 91.8% · SFT 79.4% · OPD 78.9% | 증류는 선생이 천장, RLVR 은 정답이 기준이에요 |
| 4. 두 과목 | 덧셈/뺄셈, 1,000스텝 | MOPD 98/97% · 섞은 RL 45/45% · 순서대로 RL 3/7% · 덧셈 선생 혼자 99/0% | 과목마다 그 과목 선생이 있어야 해요 |
| 5. 0% 학생에게 RLVR | 덧셈, 1,000~3,000스텝 | 정답 SFT 100% · 정답 끼워 넣기 70% · 칸마다 채점 24% · 부분 점수·쉬운 문제부터 0% | 0% 학생에게는 정답을 직접 보여 줘야 시작돼요 |
| 6. 백만 단위 + − × ÷ | dev GPU, 10,000스텝, 시드 3개 | 정답 SFT 82.5% · SFT 증류 82.1% · MOPD 81.2% · 선생 1명 OPD 81.1% · MOPD → RL 78.0% · Mix-RL 65.6% · Cascade-RL 50.8% | 선생이 다 완벽하면 증류 방법끼리 차이가 작고, RL 은 신호가 드물어 거의 못 배워요 |

<div class="rl-callout is-decision" markdown="1">
<span class="rl-callout__title">그래서 요즘 학습 순서가 이렇게 생겼어요</span>
과목마다 RL 로 전문가 선생을 만들고(RL 은 선생을 넘을 수 있으니까), 그 선생들의
빨간 펜으로 학생 한 명에게 모아요(MOPD). 섞어서 RL 하면 서로 방해하고, 순서대로
RL 하면 앞에서 배운 걸 잊어서요.
</div>

## 왜 이 실험을 했나 — 요즘 LLM 학습 순서 {#why-ko}

2025~2026년 공개 모델들의 학습 순서는 대체로 아래와 같아요. 이 글은 그중 파란 칸,
곧 **RL 과 증류(OPD·MOPD)** 를 아주 작게 만들어서 하나씩 돌려 본 거예요.

<div class="rl-flow">
  <div class="rl-flow__step"><b>사전학습</b><span>인터넷 글로 다음 단어 맞히기</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step"><b>SFT</b><span>좋은 답안 예시를 따라 쓰기. 이 실험의 "34% 출발점"</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step is-here"><b>과목별 RL</b><span>수학·코드 등 과목마다 채점기로 연습해서 전문가 선생을 만들기</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step is-here"><b>MOPD</b><span>선생들의 빨간 펜으로 학생 한 명에게 모으기</span></div>
</div>
<p class="rl-caption">그림 1. 요즘 학습 순서. 파란 칸이 이 실험에서 돌려 본 부분이에요.</p>

예를 들어 Xiaomi MiMo-V2-Flash 는 "SFT → 과목별 RL 로 선생 만들기 → 다중 선생
OPD" 세 단계라고 밝혔어요. DeepSeek-V4 는 V3.2 에서 쓰던 "여러 과목을 섞은 RL"
단계를 OPD 로 통째로 바꿨다고 해요(블로그 분석 기준이라 원문은 직접 확인하지
않았어요). 실험 4 가 바로 이 두 방식을 비교해요.

## 실험 무대 — 문제, 채점기, 모델 {#stage-ko}

### 문제: 세 자리 덧셈·뺄셈, 답은 1의 자리부터

모델은 `699+684=` 같은 문제를 받고 답을 한 글자씩 써요. 답은 **1의 자리부터**
네 자리를 쓰고, 끝에 `$`(끝 표시)를 붙여요. 1383 이면 `3831$` 이에요.

1의 자리부터 쓰는 건 손으로 세로셈할 때 순서예요. 1의 자리를 먼저 써야 받아올림을
알고, 그걸 알아야 10의 자리를 쓸 수 있어요. 그래서 이 다섯 글자가 아주 작은
**추론 과정**이에요. 앞 글자를 틀리면 뒤 글자도 흔들려요.

모델이 쓸 수 있는 글자는 `0`~`9`, `+`, `-`, `=`, `$` 이렇게 14개예요. 뺄셈은
항상 큰 수에서 작은 수를 빼서 답이 음수가 아니에요.

### 채점기(검증기): 다섯 글자가 정답과 전부 같으면 1점

RLVR 의 "V" 가 이 채점기예요(Verifiable reward, 기계로 확인되는 보상). 부분
점수는 없어요. 1383 을 1183 이라고 쓰면, 네 자리 중 세 자리가 맞아도 0점이에요.

```python
# rlsim.py — 채점기
def correct(seq, answers):
    return (seq[:, PROMPT_LEN:] == answers).all(1).float()   # 다 같으면 1, 아니면 0
```

### 모델: 작은 학생 한 명, 큰 선생 세 명

둘 다 GPT 와 같은 구조(트랜스포머)를 아주 작게 줄인 거예요. 선생이 학생보다 약
8배 커요.

| 모델 | 크기 | 파라미터 | 역할 |
| --- | --- | ---: | --- |
| 학생 | 2층, 폭 64 | 102,734 | 모든 실험에서 배우는 쪽 |
| 선생 | 4층, 폭 128 | 798,606 | SFT 증류·OPD·MOPD 에서 가르치는 쪽 |

선생 세 명은 정답으로 SFT 해서 만들었어요([자세히](#teachers-ko)). 각자 자기
과목만 배웠기 때문에 다른 과목은 거의 0% 예요. 이게 실험 4 에서 중요해져요.

| 선생 | 덧셈 | 뺄셈 | 만든 방법 |
| --- | ---: | ---: | --- |
| 덧셈 선생 | 99.8% | 0.2% | 덧셈만 SFT 500스텝 (16초) |
| 뺄셈 선생 | 0.0% | 100% | 뺄셈만 SFT 600스텝 (18초) |
| 약한 덧셈 선생 | 73.0% | 0.1% | 덧셈 SFT 를 70% 를 넘자마자 멈춤 (275스텝) |

### 선생은 이렇게 배웠어요 — 진짜 정답으로 SFT {#teachers-ko}

선생은 RL 도 증류도 쓰지 않았어요. 이 장난감 문제는 코드가 정답(a + b, a − b)을
공짜로 계산해 주니까, **정답 답안을 따라 쓰게만** 하면 됐어요. 코드로는
`load_or_train_teachers()` 가 `sft_until()` 을 부르는 게 전부예요
([코드 해설 12](#code-teachers-ko)).

```python
# 이름: (배우는 과목, 목표 정답률, 시드)
specs = {"add": (["+"], 0.995, 1), "sub": (["-"], 0.995, 2), "add-weak": (["+"], 0.70, 3)}

# 목표에 닿을 때까지 반복
prompts, answers, _ = mixed(BATCH, ops, rng)   # 자기 과목 무작위 256문제 + 진짜 정답
step_sft(model, opt, prompts, answers)         # 정답을 글자마다 따라 쓰기 (cross-entropy)
```

1. **모델:** 4층 · 폭 128 TinyGPT 를 무작위로 초기화해요. 798,606 파라미터로
   학생의 약 8배예요.
2. **학습:** 한 스텝에 자기 과목 문제 256개를 새로 뽑고, 정답 다섯 글자를
   teacher forcing 으로 따라 쓰게 해요. Adam, 학습률 0.001 이에요.
3. **멈추기:** 일정 스텝마다 검증용 1,000문제(시드 999)로 정답률을 재서 목표를
   넘으면 멈춰요. 그래프용 평가 문제(시드 12345)와는 따로예요.
4. **저장:** `results/teachers/*.pt` 로 저장하고, 실험에서는 불러다 쓰기만 해요.
   **실험 중에 선생은 전혀 학습하지 않아요.**

| 선생 | 목표 (확인 간격) | 멈춘 스텝 (시간) | 검증 정답률 |
| --- | --- | ---: | ---: |
| 덧셈 선생 | 99.5% (100스텝마다) | 500 (16초) | 99.9% |
| 뺄셈 선생 | 99.5% (100스텝마다) | 600 (18초) | 100% |
| 약한 덧셈 선생 | 70% (25스텝마다) | 275 (9초) | 75.2% |

- **약한 선생은 일부러 덜 가르쳤어요.** 방법은 덧셈 선생과 같고, 70% 를 넘자마자
  멈춘 것만 달라요. 그래서 실수가 문제마다 들쭉날쭉해요. 실험 3 에서 학생이
  처음엔 이 선생을 넘은 이유예요.
- **자기 과목 말고는 0% 예요.** 덧셈 선생은 학습 중에 `-` 글자를 한 번도 못
  봤어요. 그 글자의 임베딩은 기울기를 한 번도 받지 않아 무작위 초깃값 그대로예요.
  뺄셈 문제를 주면 548 − 286 을 834 로, 곧 더하기로 풀어요.
- **학생의 출발점(34%)도 같은 함수**로 만들었어요. 목표만 30% 로 낮췄어요.

실험의 "SFT 증류" 와 **같은 함수**(`step_sft`)예요. 넘기는 답만 달라요.

| | 따라 쓰는 답 | 선생이 틀리면 |
| --- | --- | --- |
| 선생 만들기 | `answers` — 진짜 정답 | 해당 없음. 정답이 기준이에요 |
| 실험의 SFT 증류 | `teacher_tokens(...)` — 선생이 쓴 답 | 틀린 답을 그대로 배워요 |

<div class="rl-callout is-warn" markdown="1">
<span class="rl-callout__title">선생은 맞는 앞부분만 보고 배웠어요</span>
OPD 에서 선생은 학생이 쓴, 틀릴 수도 있는 앞부분을 읽어요. 배울 때는 본 적 없는
상황이에요. 이 문제에서는 괜찮았어요. 덧셈은 각 자리 정답이 문제만 보면 정해져서,
학생이 100의 자리를 1 로 틀리게 쓴 뒤에도 선생은 1000의 자리에 정답 1 을 99.3% 로
줬어요. 실제 LLM 추론에서는 앞 단계가 틀리면 그 뒤에 맞는 말이 달라져요. 선생이
학생의 실수 상태를 얼마나 잘 받아 주는지가 OPD 품질을 좌우해요.
</div>

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">실제 LLM 에서 선생은</span>
정답 라벨을 공짜로 만들 수 없어서 선생은 자기 파이프라인으로 따로 만들어요.
사전학습 → SFT → 과목별 RLVR 순서예요. MiMo-V2-Flash 는 같은 출발 모델에서
과목마다 RL 로 전문가 선생을 만든 뒤 MOPD 로 모았다고 밝혔고, Qwen3 의 작은
모델들은 큰 Qwen3 모델을 선생으로 증류했다고 보고했어요. 이 장난감의 "정답으로
SFT" 가 실제로는 "RL 로 선생 키우기" 자리예요.
</div>

### 학생의 출발점: 정답률 약 34%

실제 LLM 은 RL 을 시작할 때 이미 사전학습과 SFT 를 마친 상태예요. 그래서 학생도
정답으로 조금 SFT 해서 정답률이 30% 를 넘으면 멈췄어요. 시드마다 380~1,115스텝이
걸렸고, 평가 문제 기준으로 평균 34% 에서 출발해요. 실험 2 만 예외로, 아무것도 안
배운 0% 학생에서 출발해요.

### 공정하게 비교하려고 맞춘 것

| 항목 | 값 | 뜻 |
| --- | --- | --- |
| 한 스텝에 쓰는 답안 | 256장 | 모든 방법이 같아요. RLVR 은 32문제 × 8번, 나머지는 256문제 × 1번 |
| 학습률 | 0.0005 (Adam) | 한 번에 파라미터를 얼마나 움직일지. 모든 방법이 같아요 |
| 답 뽑기 | temperature 1 | 학습 중에는 모델 확률 그대로 뽑아요(모험도 해요) |
| 평가 | 평가용으로 고정한 2,000문제 | 가장 확률 높은 글자만 골라 쓰게 하고(greedy) 채점해요 |
| 반복 | 시드 3개 | 그래프의 선은 평균, 옅은 띠는 최소~최대예요 |

## 학습법 네 가지를 한 문제로 하나씩 {#methods-ko}

같은 문제 하나로 네 방법이 무엇을 가르치는지 봐요. 문제는 **699 + 684**, 정답은
**1383** 이에요. 34% 출발점 학생(시드 0)은 이 문제를
<span class="ans">1<span class="bad">1</span>83</span> 이라고 써요. 100의 자리
하나만 틀렸어요.

### ① SFT 증류 — 모범답안 베껴 쓰기

선생이 먼저 문제를 풀어 모범답안을 써요. 학생은 그 답안을 한 글자씩 따라 써요.
이때 학생은 **선생이 쓴 앞부분**을 보면서 다음 글자를 맞혀요. 학생이 직접 푸는
단계가 없어요.

벌점은 "선생이 쓴 글자에 학생이 준 확률" 이 낮을수록 커요(−log 확률,
cross-entropy). 아래는 선생 답안 `1383` 을 따라 쓸 때의 벌점이에요.

| 자리 | 선생이 쓴 글자 | 학생이 그 글자에 건 확률 | 벌점 (−log 확률) |
| --- | --- | ---: | ---: |
| 1의 자리 | <span class="ans">3</span> | 99.7% | 0.00 |
| 10의 자리 | <span class="ans">8</span> | 74.4% | 0.30 |
| 100의 자리 | <span class="ans">3</span> | 10.6% | 2.25 |
| 1000의 자리 | <span class="ans">1</span> | 99.8% | 0.00 |
| 끝 표시 | <span class="ans">$</span> | 99.8% | 0.00 |

100의 자리에서 학생은 정답 3 에 10.6% 만 걸어서 벌점이 가장 커요. 학습은 이 벌점을
줄이는 쪽, 곧 "이 자리에선 3" 쪽으로 학생을 밀어요.

```python
# rlsim.py 의 step_sft 를 줄여 옮긴 것
targets = teacher_tokens(prompts, subjects, teachers)      # 선생이 쓴 모범답안
seq = torch.cat([prompts, targets], 1)
loss = F.cross_entropy(answer_logits(model, seq), targets)  # 글자마다 −log 확률
```

<p class="rl-caption">줄 단위 해설과 텐서 모양은 <a href="#code-sft-ko">코드 해설 7</a>에 있어요.</p>

### ② RLVR (GRPO) — 같은 문제를 8번 풀고 O/X 만 받기

학생이 같은 문제를 **8번** 직접 풀어요. 채점기는 답안마다 O/X 하나만 줘요. 그리고
8장의 평균과 비교해요. 평균보다 잘한 답안은 그 글자들의 확률을 올리고, 평균보다
못한 답안은 내려요. 이게 GRPO 예요(같은 문제의 답안끼리 비교하는 RL).

얼마나 올리고 내릴지는 **advantage**(이점)로 정해요. advantage = (그 답안 점수 −
8장 평균) ÷ 8장 표준편차예요.

**예 A — 699 + 684 를 8번 풀었더니 전부 틀렸어요.**

<div class="chips"><span class="chip"><span class="ans">1<span class="bad">0</span><span class="bad">7</span>3</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">5</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">2</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">0</span><span class="bad">7</span>3</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">0</span><span class="bad">7</span>3</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">2</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">1</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span><span class="chip"><span class="ans">1<span class="bad">1</span>83</span><span class="mark-no">X</span><span class="adv">0</span></span></div>

8장이 전부 0점이라 평균도 0, 차이도 0 이에요. advantage 가 전부 0 이라서 **이
문제에서는 아무것도 배우지 못해요.** 어느 자리가 틀렸는지도 몰라요.

**예 B — 227 + 189 (정답 416) 는 8번 중 2번 맞혔어요.**

<div class="chips"><span class="chip"><span class="ans"><span class="bad">5</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip is-ok"><span class="ans">416</span><span class="mark-ok">O</span><span class="adv">+1.62</span></span><span class="chip"><span class="ans"><span class="bad">6</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip"><span class="ans"><span class="bad">2</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip is-ok"><span class="ans">416</span><span class="mark-ok">O</span><span class="adv">+1.62</span></span><span class="chip"><span class="ans"><span class="bad">3</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip"><span class="ans"><span class="bad">8</span>16</span><span class="mark-no">X</span><span class="adv">−0.54</span></span><span class="chip"><span class="ans"><span class="bad">9</span><span class="bad">0</span>6</span><span class="mark-no">X</span><span class="adv">−0.54</span></span></div>

**+1.62 와 −0.54 는 이렇게 나와요.** 같은 문제 8장 안에서 그 답안이 평균보다
얼마나 잘했는지를 재요.

| 단계 | 하는 일 | 계산 | 결과 |
| --- | --- | --- | --- |
| ① 점수 | 맞으면 1, 틀리면 0 | — | <span class="ans">0 1 0 0 1 0 0 0</span> |
| ② 평균 | 8장 점수의 평균 | 2 ÷ 8 | **0.25** |
| ③ 평균과의 차이 | 그 답안 점수 − 평균 | 맞은 답 1 − 0.25 / 틀린 답 0 − 0.25 | **+0.75** / **−0.25** |
| ④ 표준편차 | ③ 을 제곱해 다 더하고 (8 − 1) 로 나눈 뒤 √ | 0.75² × 2 + 0.25² × 6 = 1.50; ÷ 7 = 0.214 → √ | **0.463** |
| ⑤ advantage | ③ ÷ ④ | +0.75 ÷ 0.463 / −0.25 ÷ 0.463 | **+1.62** / **−0.54** |

부호(+/−)는 ③ 에서, 크기는 ⑤ 에서 정해져요. ④ 에서 8 이 아니라 7 로 나누는 건
PyTorch `std()` 의 기본값이에요. 8 로 나누면 표준편차가 0.433 이라 +1.73 / −0.58
이 돼요.

그래서 맞은 두 장은 +1.62 만큼 밀어 올리고, 틀린 여섯 장은 −0.54 만큼 밀어
내려요. 그런데 틀린 <span class="ans">516</span> 도 뒤 두 자리 "16" 은 맞았어요.
그래도 답안 통째로 −0.54 라서 맞은 글자까지 같이 내려가요. O/X 하나로는 글자를
가려 주지 못해요. 다만 8장을 합치면 조금 달라져요([코드 해설 8](#code-rlvr-ko)).

아래에서 직접 O/X 를 바꿔 보세요. 맞은 장수에 따라 학습 신호가 어떻게 바뀌는지
보여요.

<div class="rl-grpo" data-rl-grpo></div>

**맞은 개수별로 정리하면.** 점수가 0 또는 1 뿐이라 8장 중 맞은 개수 하나로 숫자가
다 정해져요. 위 계산기와 같은 식이에요.

| 8장 중 맞은 수 | 평균 · 표준편차 | 맞은 답안 | 틀린 답안 |
| --- | ---: | ---: | ---: |
| 0장 | 0.000 · 0.000 | — | 0 |
| 1장 | 0.125 · 0.354 | +2.47 | −0.35 |
| 2장 (예 B) | 0.250 · 0.463 | +1.62 | −0.54 |
| 3장 | 0.375 · 0.518 | +1.21 | −0.72 |
| 4장 | 0.500 · 0.535 | +0.94 | −0.94 |
| 5장 | 0.625 · 0.518 | +0.72 | −1.21 |
| 6장 | 0.750 · 0.463 | +0.54 | −1.62 |
| 7장 | 0.875 · 0.354 | +0.35 | −2.47 |
| 8장 | 1.000 · 0.000 | 0 | — |

- **드문 쪽이 크게 움직여요.** 1장만 맞히면 그 한 장이 +2.47 로 크게 올라가요.
  7장 맞히고 1장 틀리면 그 한 장이 −2.47 로 크게 내려가요.
- **합은 항상 0 이에요.** 2장 맞힘이면 +1.62 × 2 − 0.54 × 6 = 0 이에요. 올리는
  양과 내리는 양이 같아서, 8장이 모두 똑같이 쓴 글자는 상쇄돼요
  ([코드 해설 8](#code-rlvr-ko)).
- **8장이 다 맞거나 다 틀리면 0 이에요.** ③ 의 차이가 전부 0 이라 비교할 대상이
  없어요. 그래서 RLVR 은 **너무 어려운 문제에서도, 너무 쉬운 문제에서도** 배우지
  못해요.
- **틀린 정도는 반영되지 않아요.** 516(한 자리 틀림)과 906(두 자리 틀림)이 똑같이
  −0.54 예요. 점수가 0/1 뿐이라서예요. 부분 점수를 주는 채점기였다면 답안마다 값이
  달라져요.

**이 숫자가 실제로 하는 일.** 손실은 −advantage × log 확률(그 답안) 이에요. 이걸
줄이는 방향으로 학습하니까 +1.62 는 그 답안이 나올 확률을 1.62 의 세기로 올리고,
−0.54 는 0.54 의 세기로 내려요. 여기에 학습률이 곱해지고 256장 × 5글자로 평균을
내기 때문에 확률이 1.62 만큼 바뀐다는 뜻은 아니에요. 답안끼리의 상대적인
세기예요.

⑤ 의 표준편차 나누기는 논란이 있어요. 1장만 맞힌 아주 어려운 문제나 7장 맞힌 아주
쉬운 문제는 표준편차가 작아서, 나누면 큰 값이 몰려요. Dr. GRPO 는 이 나누기를 빼고
③ 의 차이(+0.75 / −0.25)를 그대로 써요.

```python
# rlsim.py 의 step_rlvr 를 줄여 옮긴 것
seq = write(model, prompts.repeat_interleave(8, 0))          # 문제마다 8번 풀기
reward = correct(seq, answers).view(-1, 8)                   # O/X
advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
logp = log_prob_of_written_tokens(model, seq)                # 쓴 글자들의 log 확률
loss = -(advantage.view(-1, 1) * logp).mean()                # 잘한 답은 올리고, 못한 답은 내리고
```

<p class="rl-caption">줄 단위 해설, 이 8장의 실제 log 확률, 그리고 8장을 합치면 신호가 어느 자리에 남는지는 <a href="#code-rlvr-ko">코드 해설 8</a>에 있어요.</p>

### ③ OPD — 내가 푼 답안에 선생의 빨간 펜

OPD(On-Policy Distillation)는 학생이 **직접** 풀어요. 그다음 선생이 학생 답안을
처음부터 읽어 가면서, 글자마다 "이 앞부분 다음에 나라면 뭘 쓸지" 확률을 줘요.
학생은 자기 확률이 선생 확률과 비슷해지도록 배워요. 선생은 진짜 정답으로 미리 SFT
해 둔 더 큰 모델이에요([선생은 이렇게 배웠어요](#teachers-ko)).

아래는 학생이 쓴 <span class="ans">1<span class="bad">1</span>83</span> 에 선생이
매긴 빨간 펜이에요. 각 칸은 그 자리에서 학생과 선생이 어떤 글자에 확률을 얼마나
걸었는지예요(상위 3개).

| 자리: 학생이 쓴 글자 | 학생의 확률 (상위) | 선생의 확률 (상위) | 벌점 (reverse KL) |
| --- | --- | --- | ---: |
| 1의 자리: <span class="ans">3</span> | <span class="ans">3</span> 99.7% · <span class="ans">1</span> 0.1% · <span class="ans">2</span> 0.1% | <span class="ans">3</span> 99.9% | **0.01** |
| 10의 자리: <span class="ans">8</span> | <span class="ans">8</span> 74.4% · <span class="ans">7</span> 22.3% · <span class="ans">9</span> 2.3% | <span class="ans">8</span> 99.9% | **1.34** |
| 100의 자리: <span class="ans"><span class="bad">1</span></span> (정답 3) | <span class="ans">1</span> 35.8% · <span class="ans">0</span> 22.4% · <span class="ans">2</span> 20.3% | <span class="ans">3</span> 99.2% · <span class="ans">4</span> 0.7% | **9.48** |
| 1000의 자리: <span class="ans">1</span> | <span class="ans">1</span> 99.8% · <span class="ans">2</span> 0.1% · <span class="ans">3</span> 0.1% | <span class="ans">1</span> 99.3% · <span class="ans">2</span> 0.6% | **0.01** |
| 끝 표시: <span class="ans">$</span> | <span class="ans">$</span> 99.8% · <span class="ans">1</span> 0.1% | <span class="ans">$</span> 100% | **0.01** |

벌점은 **reverse KL** 이에요. 쉽게 말하면 "학생이 확률을 건 글자마다, 선생은
거기에 얼마나 걸었나" 를 봐요. 학생이 많이 걸었는데 선생은 거의 안 건 글자가
있을수록 벌점이 커요. 식으로는 Σ 학생확률(c) × log(학생확률(c) ÷ 선생확률(c))
예요.

- **100의 자리 (벌점 9.48):** 학생은 1 에 36% 를 걸었는데 선생은 1 에 거의 0%
  예요. 선생은 3 에 99% 를 걸어요. 틀린 자리가 정확히 짚여요.
- **10의 자리 (벌점 1.34):** 8 로 맞게 썼는데도 벌점이 있어요. 학생이 7 에도 22%
  를 걸었는데 선생은 7 에 0% 라서요. **맞았지만 자신 없는 것**까지 고쳐 줘요.
  RLVR 은 이걸 못 해요.
- **나머지 자리 (0.01):** 학생과 선생이 거의 같아서 벌점이 없어요.

"On-policy" 는 학생이 **자기가 실제로 쓴 답안**으로 배운다는 뜻이에요. SFT 증류는
선생이 쓴 앞부분만 보지만, OPD 는 학생이 실수한 앞부분 뒤에서도 "여기서부터는
이렇게" 를 배워요. 대신 답안마다 선생 모델을 한 번 더 돌려야 해요.

```python
# rlsim.py 의 step_opd 를 줄여 옮긴 것
seq = write(model, prompts)                                   # 학생이 직접 풀기 (한 번)
s = answer_logits(model, seq).log_softmax(-1)                 # 글자마다 학생 확률
t = answer_logits(teacher, seq).log_softmax(-1)               # 같은 답안을 선생이 읽은 확률
loss = (s.exp() * (s - t)).sum(-1).mean()                     # reverse KL, 글자 14개 전부
```

<p class="rl-caption">기울기 식과 100의 자리에서 확률이 어디로 옮겨 가는지는 <a href="#code-opd-ko">코드 해설 9</a>에 있어요.</p>

### ④ MOPD — 과목마다 그 과목 선생의 빨간 펜

MOPD(Multi-teacher OPD)는 OPD 와 똑같은데, 문제마다 **과목표를 보고 그 과목
선생**에게 보내요. 덧셈 문제는 덧셈 선생이, 뺄셈 문제는 뺄셈 선생이 빨간 펜을
잡아요.

뺄셈 문제 **548 − 286**(정답 262)를 두 과목 출발점 학생이
<span class="ans">2<span class="bad">5</span>2</span> 이라고 썼어요. 같은 답안에
두 선생이 각각 빨간 펜을 들면 이렇게 달라요.

| 자리 | 학생이 쓴 글자 | 뺄셈 선생이라면 | 덧셈 선생이라면 |
| --- | --- | --- | --- |
| 1의 자리 | <span class="ans">2</span> | <span class="ans">2</span> | <span class="ans">4</span> |
| 10의 자리 | <span class="ans"><span class="bad">5</span></span> (정답 6) | <span class="ans">6</span> | <span class="ans">3</span> |
| 100의 자리 | <span class="ans">2</span> | <span class="ans">2</span> | <span class="ans">8</span> |
| 1000의 자리 | <span class="ans">0</span> | <span class="ans">0</span> | <span class="ans">1</span> |

뺄셈 선생은 틀린 10의 자리만 6 으로 고쳐 줘요. 그런데 덧셈 선생에게 이 답안을
맡기면 모든 자리를 다르게 고쳐요. 덧셈 선생이 이 문제를 혼자 풀면
<span class="ans">834</span> 을 써요. 548 + 286 이에요. 빼기 기호를 본 적이 없어서
무시하고 더해요. 이런 선생에게 뺄셈을 맡기면 학생에게 "더하라" 고 가르치게
돼요(실험 4).

```python
# rlsim.py — MOPD 는 OPD 에 선생 목록만 과목별로 넘겨요
step_opd(model, opt, prompts, subjects, teachers={"+": add_teacher, "-": sub_teacher})
```

| 방법 | 학생이 직접 푸나 | 피드백 단위 | 필요한 것 |
| --- | --- | --- | --- |
| SFT 증류 | 아니요 | 글자마다 (선생 답안 기준) | 선생 |
| RLVR | 예, 문제당 8번 | 답안마다 O/X 하나 | 채점기 |
| OPD | 예, 문제당 1번 | 글자마다 확률 분포 | 선생 |
| MOPD | 예, 문제당 1번 | 글자마다, 과목별 선생 | 과목별 선생 |

## 세 방법의 학습 환경 — 무엇을 보고, 누가 쓰고, 무엇을 돌려받나 {#env-ko}

세 방법은 같은 "게임" 위에서 돌아요. 문제를 보고 글자를 하나씩 다섯 개 쓰면 한
판이 끝나요. 방법마다 다른 건 **답안을 누가 쓰는지**와 **무엇을 돌려받는지** 두
가지예요.

### 게임의 규칙 — RL 용어로

| RL 용어 | 이 실험에서 | 코드 |
| --- | --- | --- |
| 상태 (state) | 문제 + 지금까지 쓴 글자. 예: `699+684=` 다음에 `3`, `8` 까지 쓴 상태 | `seq` 의 앞부분 |
| 행동 (action) | 다음 글자 하나. 14개 중에서 골라요 | `write()` 의 `nxt` |
| 한 판 (episode) | 다섯 글자를 쓰면 끝. 1의 자리 → 10 → 100 → 1000 → `$` | `ANSWER_LEN = 5` |
| 정책 (policy) | 학생 모델. 상태를 받아 글자 14개의 확률을 내요 | `TinyGPT` |
| 보상 (reward) | 방법마다 달라요. 아래 표 | `correct()` 또는 선생 |

### 방법별 환경

| 항목 | SFT 증류 | RLVR (GRPO) | OPD · MOPD |
| --- | --- | --- | --- |
| 한 스텝에 받는 문제 | 무작위 256문제 | 무작위 32문제, 문제마다 8번 | 무작위 256문제 |
| 답안을 쓰는 쪽 | **선생**. 가장 확률 높은 글자만 (greedy) | **학생**. 확률대로 뽑아서 (temperature 1) | **학생**. 확률대로 뽑아서, 문제당 1번 |
| 다음 글자를 맞힐 때 보는 앞부분 | 선생이 쓴 글자 (teacher forcing) | 자기가 쓴 글자 | 자기가 쓴 글자 |
| 돌려받는 것 | 자리마다 "선생이 쓴 글자" 하나 | 답안마다 0 또는 1, 다 쓴 뒤 한 번 | 자리마다 선생의 글자 14개 확률 |
| 보상으로 치면 | 보상 없이 시연을 따라 하기 (행동 복제) | 드문 보상: 다섯 글자에 숫자 하나 | 촘촘한 보상: 글자마다 −reverse KL |
| 정답이 필요한가 | 아니요. 선생 답을 정답으로 취급 | **예**. 채점기 `correct()` | 아니요 |
| 선생이 필요한가 | 예. 매 스텝 답안 256장을 새로 씀 | 아니요 | 예. 학생 답안 256장을 한 번 읽음 (MOPD 는 과목별로 나눠서) |
| 손실 | cross-entropy | −advantage × log 확률 | reverse KL (글자 14개 전부 합) |
| 한 스텝에 모델을 돌리는 횟수 | 선생 5번 (답 쓰기) + 학생 1번 (+역전파) | 학생 5번 (답 쓰기) + 학생 1번 (+역전파) | 학생 5번 (답 쓰기) + 학생 1번 (+역전파) + 선생 1번 (읽기) |
| 1,000스텝 걸린 시간 (실험 1, 평가 포함) | 45초 | 22초 | 31초 |
| 코드 | [`step_sft`](#code-sft-ko) + `teacher_tokens` | [`step_rlvr`](#code-rlvr-ko) | [`step_opd`](#code-opd-ko) |

- **SFT 가 가장 느린 이유:** 학생보다 8배 큰 선생이 매 스텝 답을 *써요*. 쓰기는
  글자 수만큼(5번) 모델을 돌려야 해요. OPD 의 선생은 학생이 이미 쓴 답안을
  *읽기만* 해서 한 번에 다섯 칸이 다 나와요. 실제 LLM 에서도 OPD 의 선생 비용은
  생성이 아니라 읽기(prefill)라서 상대적으로 싸요.
- **SFT 와 OPD 는 정답을 쓰지 않아요.** `mixed()` 가 정답도 함께 돌려주지만 두
  방법은 버려요. 선생만 있으면 되고, 그래서 실험 3 처럼 선생이 틀리면 학생도 그대로
  배워요. RLVR 은 반대로 선생 없이 정답(채점기)만 있으면 돼요.
- **RLVR 만 같은 문제를 여러 번 풀어요.** 한 장의 O/X 로는 "잘했나" 를 알 수 없고,
  같은 문제의 다른 답안과 비교해야 해서예요. 그래서 같은 256장이라도 문제 수는
  32개로 줄어요.

### 모든 방법이 똑같이 쓰는 것

- **출발 가중치:** 시드마다 출발점 학생을 한 번 만들고, 방법마다
  `copy.deepcopy(start)` 로 복사해서 써요. 세 방법이 정확히 같은 가중치에서
  출발해요.
- **옵티마이저:** 방법마다 새 Adam, 학습률 0.0005. 기울기 전체 길이가 1 을 넘으면
  1 로 잘라요 (`clip_grad_norm_`).
- **문제 순서:** 방법마다 같은 시드로 난수 생성기를 새로 만들어요. SFT 와 OPD 는 한
  스텝에 256문제씩 뽑아서 **똑같은 문제를 같은 순서로** 봐요. RLVR 은 32문제씩
  뽑아서 문제 순서가 달라요.
- **평가:** 20스텝마다, 시드 12345 로 고정한 과목별 2,000문제를 greedy 로 풀게
  해서 정답률을 재요. 평가는 학습에 영향을 주지 않아요.

### 실험마다 바뀐 것

| 실험 | 출발 학생 | 과목 · 스텝 | 선생 |
| --- | --- | --- | --- |
| 1 | 덧셈 SFT 로 34% (시드별 410 · 1,115 · 380스텝) | 덧셈 · 1,000 | SFT·OPD: 덧셈 선생 (99.8%) |
| 2 | 무작위 초기화 (0%) | 덧셈 · 1,000 | OPD: 덧셈 선생 |
| 3 | 실험 1 과 같은 34% 학생 | 덧셈 · 3,000 | SFT·OPD: 약한 덧셈 선생 (73%) |
| 4 | 두 과목 SFT 로 덧셈 34% · 뺄셈 30% (1,585 · 1,220 · 400스텝) | 덧셈·뺄셈 반반 · 1,000. Cascade 는 덧셈 500 → 뺄셈 500 | MOPD: 과목별 선생. 대조군: 두 과목 다 덧셈 선생 |

### 실행 환경

Apple M4 Max 노트북의 CPU 만 썼어요(GPU 없음). Python 3.11, PyTorch 2.10, NumPy
2.3 이고, `torch.set_num_threads(4)` 로 스레드 4개만 써요. 선생 셋을 만드는 데 1분
안쪽, 실험 넷에 약 35분 걸렸어요. 외부 데이터나 사전학습 가중치는 하나도 안 써요.
문제는 코드가 그 자리에서 만들어요.

## 실험 1 — 34% 에서 출발한 학생 {#exp1-ko}

**알고 싶은 것:** 출발점이 같을 때 어느 방법이 빨리 배우나요?<br>
**설정:** 34% 학생, 덧셈만, 1,000스텝. SFT 증류와 OPD 는 덧셈 선생(99.8%)에게
배워요.

<figure class="rl-chart" data-rl-chart="single"></figure>
<p class="rl-caption">그림 2. 덧셈 정답률. 선 위에 마우스를 올리면 그 스텝의 값이 보여요.</p>

| 방법 | 1,000스텝 후 (시드 범위) | 90% 도달 | CPU 시간 |
| --- | ---: | ---: | ---: |
| SFT 증류 | 99.9% (99.8~99.9) | 180스텝 | 45초 |
| OPD | 98.6% (97.4~100) | 280스텝 | 31초 |
| RLVR | 81.0% (78.5~83.4) | 못 닿음 | 22초 |

**왜 이렇게 됐나:** 한 스텝에 받는 정보량이 달라요. RLVR 은 256장을 쓰고도 받는 건
"32문제 각각에서 몇 장이 맞았나" 뿐이에요. OPD 는 답안 한 장에서 다섯 글자 각각의
확률 분포(14가지)를 받아요.

게다가 RLVR 은 배울 거리가 있는 문제가 갈수록 줄었어요. 처음에는 61% 였는데
1,000스텝 무렵엔 24% 예요. 처음엔 8번 다 틀리는 어려운 문제가, 나중엔 8번 다
맞히는 쉬운 문제가 신호 0 이에요(그림 3 오른쪽).

이 실험에서 SFT 증류가 OPD 보다 조금 빨랐어요. 답이 다섯 글자뿐이라 "자기
실수에서 이어 쓰기" 를 배울 일이 거의 없어서예요. 이 점은
[같은 점, 다른 점](#real-ko)에서 다시 다뤄요.

## 실험 2 — 아무것도 모르는 학생 {#exp2-ko}

**알고 싶은 것:** 아무것도 모르는 학생도 각 방법으로 배울 수 있나요?<br>
**설정:** 무작위로 초기화한 학생(0%), 덧셈만, 1,000스텝.

<div class="rl-pair">
<div><div class="rl-pair__title">정답률 — 0% 에서 출발</div><figure class="rl-chart" data-rl-chart="cold"></figure></div>
<div><div class="rl-pair__title">RLVR 에서 배울 거리가 있던 문제</div><figure class="rl-chart" data-rl-chart="signal"></figure></div>
</div>
<p class="rl-caption">그림 3. 왼쪽: 0% 학생의 정답률. 오른쪽: RLVR 에서 8장이 다 같지 않아 신호가 있던 문제의 비율.</p>

| 방법 | 1,000스텝 후 (시드 범위) | 90% 도달 | 배울 거리 있던 문제 |
| --- | ---: | ---: | ---: |
| OPD | 96.4% (93.4~98.4) | 740스텝 | 항상 100% |
| RLVR | 0.0% (0.0~0.0) | 못 닿음 | 0% |

**왜 이렇게 됐나:** 아무것도 모르는 학생이 다섯 글자를 우연히 다 맞힐 확률은 대략
(1/14)⁵, 약 54만분의 1 이에요. 1,000스텝 동안 256,000장을 써도 맞을 거라
기대되는 답안이 0.5장이에요. 8장이 매번 다 틀리니 신호가 계속 0 이고, 학생은
그대로예요.

OPD 는 틀린 답안에도 글자마다 "여기엔 이 숫자" 를 알려 줘서 바닥부터 배워요.

### OPD 는 왜 400~500스텝에 갑자기 오르나 {#cold-jump-ko}

처음 400스텝은 거의 0% 였다가 500스텝 무렵 갑자기 올라요. 시드 0 을 똑같이 다시
돌리면서(정답률 곡선이 원래와 똑같이 재현돼요) 답의 자리별 정답률을 따로 재
봤어요.

| 스텝 | 자리별 정답률 % (1 · 10 · 100 · 1000 · $) | 전체 정답 | KL |
| ---: | --- | ---: | ---: |
| 0 | <span class="ans">11 · 10 · 0 · 0 · 0</span> | 0.0% | — |
| 100 | <span class="ans">10 · 11 · 10 · 71 · 100</span> | 0.2% | 4.57 |
| 200 | <span class="ans">12 · 10 · 11 · 75 · 100</span> | 0.3% | 4.45 |
| 300 | <span class="ans">30 · 10 · 10 · 73 · 100</span> | 0.5% | 4.30 |
| 400 | <span class="ans">90 · 26 · 10 · 78 · 100</span> | 2.3% | 3.19 |
| 500 | <span class="ans">99 · 90 · 81 · 98 · 100</span> | 73.4% | 0.92 |
| 600 | <span class="ans">99 · 98 · 98 · 100 · 100</span> | 95.3% | 0.12 |
| 1,000 | <span class="ans">99 · 99 · 99 · 100 · 100</span> | 97.4% | 0.04 |

<p class="rl-caption">평가 문제 2,000개, greedy. KL 은 그 직전 20스텝의 평균 손실이에요.</p>

- **쉬운 자리부터 배워요.** 끝 표시 `$` 는 100스텝 만에 100% 예요. 1000의 자리(0
  아니면 1)도 금방 70% 대에 와요. 계산이 거의 필요 없는 자리예요.
- **계산이 필요한 자리는 한동안 10% 예요.** 1 · 10 · 100의 자리는 300스텝까지 10%
  안팎이에요. 숫자 10개 중 아무거나 찍는 수준이에요. KL 도 4.3~4.6 에서 거의 안
  내려가요. 문제의 어느 칸끼리 더해야 하는지를 아직 못 찾은 상태로 보여요.
- **덧셈은 세로셈 순서대로 풀려요.** 1의 자리가 300 → 400스텝에 30% → 90%, 이어서
  400 → 500스텝에 10의 자리가 26% → 90%, 100의 자리가 10% → 81% 로 올라요.
  받아올림이 없는 1의 자리가 먼저고, 앞 자리 받아올림이 필요한 자리가 뒤따라요.
  뺄셈은 순서가 달라요([아래](#cold-sub-ko)).
- **전체 정답은 다섯 자리가 다 맞아야 해요.** 400스텝에 1의 자리는 90% 인데 전체
  정답은 2.3% 예요. 나머지 자리가 아직 찍기 수준이라서요. 전체 정답률은 자리별
  정답률을 곱한 값과 거의 같아요(500스텝: 곱 71%, 실제 73%).

정리하면 두 가지가 겹쳐요. 하나는 자리 규칙 자체를 한동안 못 찾다가 찾으면 빠르게
배우는 거예요. 자리별 정답률도 실제로 뛰어요. 다른 하나는 다섯 글자가 다 맞아야
1점인 채점이에요. 자리마다 차례로 배우는 과정이 전체 정답률로는 "가만히 있다가 한
번에 뛰는" 모양으로 보이게 해요. 앞 칸에 정답을 넣어 줘도(teacher forcing) 자리별
정답률이 거의 같아서, 앞 칸 실수가 뒤로 번진 탓은 아니에요. 34% 에서 출발한 실험
1 은 이미 규칙을 어느 정도 알고 시작해서 이런 정체 구간이 없어요.

### 뺄셈은? — 순서를 정하는 건 "계산이 얼마나 복잡한가" {#cold-sub-ko}

뺄셈은 1의 자리부터가 아니에요. 그렇다고 가로(읽는) 순서도 아니에요. OPD 와 정답
SFT 의 시드 6개 모두 이렇게 됐어요.

- **100의 자리가 가장 먼저 반을 넘어요.** 200~220스텝에 50% 를 넘고, 그 뒤 60~70%
  대에서 한동안 멈춰요.
- **완성(90%)은 1의 자리가 먼저예요.** 1의 자리가 420~940스텝에 90% 를 넘고,
  100의 자리가 그 뒤예요. 10의 자리가 가장 늦어요. OPD 는 1,000스텝 안에 10의
  자리가 90% 를 못 넘었어요.
- **뺄셈이 덧셈보다 어려워요.** OPD 1,000스텝 전체 정답이 뺄셈 72.3~83.1%, 덧셈
  93.5~98.4% 예요.

| 실험 | 50% 를 넘은 스텝 (1 · 10 · 100의 자리) | 90% 를 넘은 스텝 (1 · 10 · 100의 자리) | 마지막 전체 정답 (시드 범위) |
| --- | --- | --- | ---: |
| 덧셈 OPD (1,000스텝) | 433 · 540 · 560 | 507 · 647 · 653 | 96.4% (93.5~98.4) |
| 뺄셈 OPD (1,000스텝) | 340 · 620 · 213 | 827 · 못 닿음 · 970 (2/3) | 79.3% (72.3~83.1) |
| 덧셈 정답 SFT (2,000스텝) | 433 · 580 · 780 | 473 · 667 · 873 | 100.0% (100.0~100.0) |
| 뺄셈 정답 SFT (2,000스텝) | 493 · 880 · 213 | 573 · 1,100 · 887 | 99.8% (99.5~100.0) |

<p class="rl-caption">시드 3개 평균이에요. (2/3) 은 시드 셋 중 둘만 닿았다는 뜻이고, 평균은 닿은 시드로만 냈어요.</p>

자리마다 필요한 계산을 나눠 보면 이유가 보여요. 아래에서 "한 바퀴" 는 10 을
넘거나 0 아래로 내려가서 한 바퀴 돌아오는 계산이에요(7 + 5 → 2, 3 − 8 → 5).

| 자리 | 덧셈 | 뺄셈 (큰 수 − 작은 수) |
| --- | --- | --- |
| 1의 자리 | 한 바퀴 | 한 바퀴 |
| 10의 자리 | 한 바퀴 + 받아올림 (1의 자리를 봐야 해요) | 한 바퀴 + 받아내림 (1의 자리를 봐야 해요) |
| 100의 자리 | 한 바퀴 + 받아올림 (10·1의 자리를 다 봐야 해요) | 한 바퀴 없음 + 받아내림 (10·1의 자리를 다 봐야 해요) |

이 실험의 뺄셈은 늘 큰 수에서 작은 수를 빼서, 100의 자리끼리 빼면 0~9 가 바로
나와요. 한 바퀴 돌 일이 없어요. 그래서 받아내림을 무시하고 "a 의 100의 자리 − b 의
100의 자리" 만 해도 절반 넘게 맞아요. 60~70% 에 멈춘 게 이 단계로 보여요.
받아내림까지 맞히려면 아래 자리를 봐야 해서, 1의 자리를 배운 뒤에야 90% 를 넘어요.
10의 자리는 한 바퀴와 받아내림이 둘 다 필요해서 가장 늦어요.

덧셈의 1 → 10 → 100 순서도 세로셈 순서라서라기보다, 그 순서가 계산이 쉬운 순서와
같아서로 보여요. 모델은 쓰는 순서가 아니라 쉬운 계산부터 배워요.

### 답을 1000의 자리부터 쓰면? — 훨씬 못 배워요 {#cold-order-ko}

그럼 쓰는 순서는 상관없을까요? 답을 읽는 순서(1000의 자리부터)로 쓰게 바꿔서, 1의
자리부터 쓰는 지금 형식과 비교했어요. 이 형식에 맞는 선생이 없어서 정답 SFT 로, 0%
학생에서 2,000스텝 돌렸어요(시드 3개, `digit_order.py`).

<figure class="rl-chart" data-rl-bars="order"></figure>
<p class="rl-caption">그림 3-3. 정답 SFT 2,000스텝 뒤 자리별·전체 정답률(시드 3개 평균). 1000의 자리는 모든 형식에서 100% 라 뺐어요.</p>

| 형식 | 2,000스텝 전체 정답 (시드 범위) | 자리별 정답률 % (1 · 10 · 100 · 1000) | 90% 에 닿은 시드 (1 · 10 · 100 · 1000) |
| --- | ---: | --- | --- |
| 덧셈 · 1의 자리부터 | 100.0% (100.0~100.0) | 100 · 100 · 100 · 100 | 3/3 · 3/3 · 3/3 · 3/3 |
| 덧셈 · 1000의 자리부터 | 14.1% (7.8~22.7) | 31 · 71 · 95 · 100 | 0/3 · 1/3 · 3/3 · 3/3 |
| 뺄셈 · 1의 자리부터 | 99.8% (99.5~100.0) | 100 · 100 · 100 · 100 | 3/3 · 3/3 · 3/3 · 3/3 |
| 뺄셈 · 1000의 자리부터 | 13.7% (2.5~33.1) | 40 · 30 · 96 · 100 | 1/3 · 0/3 · 3/3 · 3/3 |

- **1의 자리부터 쓰면** 2,000스텝에 덧셈 100%, 뺄셈 99.5~100% 예요.
- **1000의 자리부터 쓰면** 덧셈 8~23%, 뺄셈 3~33% 예요. 1000·100의 자리는 금방
  배워요(90% 도달: 1000의 자리 0~100스텝, 100의 자리 420~920스텝). 10의 자리와
  1의 자리는 시드 6개 중 하나씩만 2,000스텝 안에 90% 에 닿았어요.

1의 자리부터 쓰면 모델이 앞에서 직접 쓴 낮은 자리가 다음 자리를 계산하는 메모가
돼요. 예를 들어 a 의 1의 자리가 7 인데 답의 1의 자리에 2 를 썼다면 받아올림이
있었다는 걸 바로 알 수 있어요(7 + 5 = 12). 1000의 자리부터 쓰면 이 메모 없이
받아올림을 머릿속에서 한 번에 계산해야 해요. 이 형식에서 100의 자리가 93~97% 에서
더 오르지 않은 것도 받아올림을 10의 자리만 보고 어림한 결과로 보여요. 덧셈이라면
10의 자리끼리 합이 딱 9 일 때만 1의 자리가 받아올림을 결정하는데, 그런 문제가 약
4.5% 라 어림만으로 95% 안팎까지 가요. 작은 트랜스포머에게 덧셈 답을 거꾸로(1의
자리부터) 쓰게 하면 훨씬 잘 배운다는 연구 결과(Lee 등, 2023)와 같은 방향이에요.

다만 1000의 자리부터 쓸 때 계산이 가장 쉬운 1의 자리가 가장 늦게 배워지는 이유는
확인하지 못했어요.

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">실제 LLM 에서는</span>
LLM 에게 추론 과정(chain of thought)을 쓰게 하는 이유와 같아요. 중간 결과를 먼저
써 두면 다음 단계가 쉬워져요. 1의 자리부터 쓰는 다섯 글자가 이 실험에서 가장 작은
"추론 과정" 이에요([실험 무대](#stage-ko)).
</div>

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">실제 LLM 에서는</span>
그래서 RL 전에 SFT 로 "가끔은 맞히는" 상태를 먼저 만들어요(cold-start SFT). 이
실험의 출발점 학생(무작위 → 정답 SFT 380~1,115스텝 만에 30%)과 선생(무작위 →
500스텝 만에 99.9%)이 바로 이렇게 만들어졌어요. 작은 모델은 아예 RL 대신 큰
모델에서 증류해요. Qwen3 의 작은 모델들도 이렇게 만들었다고 보고했어요. SFT 말고
다른 방법으로도 시작되는지는 [실험 5](#exp5-ko)에서 비교했어요.
</div>

## 실험 3 — 선생이 약하면 {#exp3-ko}

**알고 싶은 것:** 선생이 틀리는 선생이면 학생은 어디까지 가나요?<br>
**설정:** 34% 학생, 덧셈만, 3,000스텝. SFT 증류와 OPD 는 약한 덧셈 선생(73%)에게
배워요. RLVR 은 선생 없이 채점기만 써요.

<figure class="rl-chart" data-rl-chart="weak"></figure>
<p class="rl-caption">그림 4. 덧셈 정답률. 점선은 약한 선생의 실력(73%)이에요.</p>

| 방법 | 최고점 (그때 스텝) | 3,000스텝 후 (시드 범위) | 90% 도달 |
| --- | ---: | ---: | ---: |
| OPD (약한 선생) | 84.5% (840) | 78.9% (78.8~79.1) | 못 닿음 |
| SFT 증류 (약한 선생) | 82.5% (680) | 79.4% (79.1~79.9) | 못 닿음 |
| RLVR | 92.4% (2,920) | 91.8% (88.3~98.5) | 1,660스텝 |

**처음엔 학생이 선생을 넘었어요.** 73% 선생에게 배운 학생이 84% 까지 갔어요.
선생의 실수가 들쭉날쭉해서예요. 문제마다 다른 자리에서 틀리니까, 학생은 실수보다
공통된 규칙을 먼저 배워요.

**오래 배우면 다시 내려왔어요.** 학생이 선생의 확률을 점점 정확히 닮으면서 선생의
실수까지 닮아 79% 로 내려와요. 증류는 결국 "선생을 닮아라" 라서 선생이
천장이에요.

**RLVR 은 느리지만 천장이 없어요.** 기준이 선생이 아니라 진짜 정답이라서요.
1,000스텝 무렵 증류를 따라잡고, 1,660스텝에 90% 를 넘었어요.

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">실제 LLM 에서는</span>
그래서 "RL 로 선생을 키우고 → 증류로 옮기는" 순서예요. 증류만으로는 가장 좋은
선생보다 나아질 수 없고, RL 만으로는 너무 느리고 비싸서요.
</div>

## 실험 4 — 과목이 둘이면 {#exp4-ko}

**알고 싶은 것:** 학생 한 명이 두 과목을 다 잘하게 하려면 어떻게 하나요?<br>
**설정:** 두 과목을 조금씩 배운 학생(덧셈 34%, 뺄셈 30%), 1,000스텝, 네 가지 방법.

- **Mix-RL** — 한 스텝에 덧셈 16문제와 뺄셈 16문제를 섞어 각 8번씩 풀고 O/X.
- **Cascade-RL** — 덧셈만 500스텝 RL 한 다음, 뺄셈만 500스텝 RL.
- **MOPD** — 한 스텝에 덧셈 128문제는 덧셈 선생이, 뺄셈 128문제는 뺄셈 선생이 빨간
  펜.
- **덧셈 선생 혼자** — 같은 OPD 인데 두 과목 다 덧셈 선생이 빨간 펜.

<div class="rl-pair">
<div><div class="rl-pair__title">덧셈 정답률</div><figure class="rl-chart" data-rl-chart="multi-add"></figure></div>
<div><div class="rl-pair__title">뺄셈 정답률</div><figure class="rl-chart" data-rl-chart="multi-sub"></figure></div>
</div>
<p class="rl-caption">그림 5. 두 과목 정답률. 세로 점선(500스텝)에서 Cascade-RL 이 덧셈에서 뺄셈으로 넘어가요.</p>

| 방법 | 덧셈 (시드 범위) | 뺄셈 (시드 범위) | 한 줄 요약 |
| --- | ---: | ---: | --- |
| MOPD | 98.3% (97.2~99.2) | 97.5% (95.6~98.7) | 둘 다 지켰어요 |
| 덧셈 선생 혼자 | 99.1% (98.8~99.4) | 0.2% (0.1~0.2) | 뺄셈을 더하기로 배웠어요 |
| Mix-RL | 44.6% (42.5~48.4) | 44.5% (40.0~50.2) | 떨어졌다가 천천히 올라요 |
| Cascade-RL | 2.6% (0.5~5.8) | 6.7% (0.7~17.6) | 둘 다 무너졌어요 |

### MOPD — 과목마다 맞는 선생

두 과목 모두 1,000스텝에 97% 를 넘었어요. 90% 에 닿은 건 덧셈 480스텝, 뺄셈
540스텝이에요. 과목마다 그 과목을 아는 선생이 글자마다 고쳐 주니까 서로 방해하지
않아요.

### 덧셈 선생 혼자 — 모르는 과목은 가르칠 수 없어요

덧셈은 99% 인데 뺄셈은 처음 20스텝 만에 30% 에서 3% 로 떨어지고 0% 가 됐어요.
앞에서 본 것처럼 덧셈 선생은 548 − 286 을 834 라고 써요. 학생은 그 빨간 펜을
충실히 따라서 뺄셈 문제도 더했어요. MOPD 가 과목마다 선생을 따로 두는 이유예요.

### Mix-RL — 섞으면 서로 흔들어요

처음 80스텝 동안 덧셈은 34% 에서 18% 로, 뺄셈은 30% 에서 21% 로 오히려
떨어졌어요. 한 과목의 맞은 답을 강화한 게 같은 파라미터를 쓰는 다른 과목을 흔든
거예요(시소 효과). 과목당 문제도 한 스텝에 16개로 절반이라 신호도 적어요. 그 뒤
천천히 올라 1,000스텝에 45% 였어요.

### Cascade-RL — 하나를 배우면 다른 하나를 잊어요

- **0~500스텝 (덧셈만):** 덧셈은 34% → 67% 로 올랐지만, 뺄셈은 30% → 3.5% 로
  떨어졌어요. 뺄셈은 한 번도 안 했는데 잊은 거예요(파괴적 망각).
- **500스텝 (뺄셈으로 전환):** 이미 뺄셈을 거의 못 해서 8번 다 틀리는 문제가
  대부분이에요. 배울 거리가 있는 문제가 덧셈 단계의 58.5% 에서 4.9% 로 줄었어요.
- **500~600스텝:** 덧셈도 67% → 6.5% 로 무너졌어요. 드물게 오는 뺄셈 신호로
  파라미터가 계속 움직이면서 덧셈까지 흔든 것으로 보여요. 정확한 원인은 따로
  확인하지 않았어요.

<div class="rl-callout is-warn" markdown="1">
<span class="rl-callout__title">Cascade 붕괴의 크기는 과장됐을 수 있어요</span>
이 실험에는 "원래 모델에서 너무 멀어지지 마" 라는 KL 제약도, 앞 과목 데이터를 조금
섞는 장치도 없어요. 실제 파이프라인은 이런 장치로 잊어버림을 줄여요. 잊어버리는
방향 자체는 실제로도 보고되는 문제지만, 3% 까지 무너지는 크기는 이 설정 탓이
커요.
</div>

## 실험 5 — 0% 학생에게 RLVR 을 시작시키려면 {#exp5-ko}

**알고 싶은 것:** 실험 2 에서 RLVR 은 0% 학생에게서 신호를 하나도 못 받았어요.
무엇을 바꾸면 배우기 시작하나요?<br>
**설정:** 실험 2 와 같은 무작위 학생(0%), 덧셈만, 스텝마다 답안 256장, 시드
3개예요. 다섯 방법을 1,000스텝으로 비교하고, 배우기 시작한 셋은 3,000스텝까지 더
돌렸어요.

| 방법 | 바꾼 것 | 코드에서 |
| --- | --- | --- |
| 쉬운 문제부터 | 0~9 끼리 300스텝 → 0~99 끼리 300스텝 → 0~999 끼리 400스텝. 채점은 그대로 O/X | 문제만 바꿔요 |
| 부분 점수 (답안 하나에) | 점수 = 다섯 칸 중 맞은 칸 비율(0, 0.2, …, 1). 답안마다 숫자 하나 | `hit.mean(1)` |
| 칸마다 채점 | 칸마다 맞았는지 따로 채점하고, 같은 문제 8장의 같은 칸끼리 advantage 를 계산 | `hit` 를 칸마다 |
| 정답 끼워 넣기 | 문제마다 8장 중 1장을 정답으로 바꿔 넣어요. 나머지 7장은 학생이 써요 | `seq[::8] = answers` |
| 정답 SFT (기준선) | RL 없이 정답 256장을 그대로 따라 쓰기 | `step_sft(answers)` |

```python
def step_rlvr_variant(model, opt, prompts, answers, method: str) -> dict:
    """rlsim.step_rlvr 와 같은 GRPO 한 스텝 + 변형 스위치."""
    prompts, answers = prompts.repeat_interleave(s.GROUP, 0), answers.repeat_interleave(s.GROUP, 0)
    seq = s.write(model, prompts)
    if method == "inject":  # 문제마다 첫 장을 정답으로 바꿔 넣는다
        seq[:: s.GROUP, s.PROMPT_LEN :] = answers[:: s.GROUP]
    hit = (seq[:, s.PROMPT_LEN :] == answers).float()  # (256, 5) 칸마다 맞았나
    exact = hit.all(1).float()
    logp = s.answer_logits(model, seq).log_softmax(-1).gather(-1, seq[:, s.PROMPT_LEN :, None]).squeeze(-1)
    if method == "token":  # 칸마다 따로: 같은 문제 8장의 같은 칸끼리 평균·표준편차
        reward = hit.view(-1, s.GROUP, s.ANSWER_LEN)
        advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
        loss = -(advantage.reshape(-1, s.ANSWER_LEN) * logp).mean()
        signal = (reward.std(1) > 0).float().mean().item()
    else:
        reward = (hit.mean(1) if method == "partial" else exact).view(-1, s.GROUP)
        advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
        loss = -(advantage.view(-1, 1) * logp).mean()
        signal = (reward.std(1) > 0).float().mean().item()
    s.update(opt, model, loss)
    own = exact.view(-1, s.GROUP)[:, 1:] if method == "inject" else exact.view(-1, s.GROUP)  # 학생이 직접 쓴 답안만
    return {"signal": signal, "own_exact": own.mean().item()}
```

<p class="rl-caption">exp5.py (50~71줄) — rlsim.py 의 <code>step_rlvr</code> 에 스위치만 붙인 거예요. rlsim.py 는 그대로 두고 함수만 가져다 써요.</p>

### 1,000스텝 결과

| 방법 | 1,000스텝 정답률 (시드 범위) | 자리별 정답률 % (1 · 10 · 100 · 1000 · $) | 배울 거리가 있던 비율 |
| --- | ---: | --- | ---: |
| RLVR 그대로 (실험 2) | 0.0% (0.0~0.0) | 처음 그대로 | 0% |
| 쉬운 문제부터 | 0.0% (0.0~0.0) | 처음 그대로 | 0% |
| 부분 점수 (답안 하나에) | 0.0% (0.0~0.1) | <span class="ans">14 · 11 · 10 · 50 · 100</span> | 86% |
| 칸마다 채점 | 0.2% (0.1~0.2) | <span class="ans">44 · 15 · 18 · 66 · 100</span> | 32% (칸 기준) |
| 정답 끼워 넣기 | 0.6% (0.2~1.4) | <span class="ans">37 · 14 · 10 · 76 · 100</span> | 100% |
| 정답 SFT | 72.3% (16.9~100.0) | <span class="ans">100 · 100 · 72 · 94 · 100</span> | — |
| 참고: OPD (실험 2) | 96.4% (93.4~98.4) | — | 100% |

<p class="rl-caption">자리별 정답률은 세 시드 평균이에요. "처음 그대로" 는 기울기가 계속 0 이라 가중치가 출발점에서 한 번도 안 바뀐 경우예요.</p>

### 3,000스텝까지 — 배우기 시작한 셋

<figure class="rl-chart" data-rl-chart="exp5"></figure>
<p class="rl-caption">그림 6. 0% 학생의 덧셈 정답률. 선 위에 마우스를 올리면 그 스텝의 값이 보여요.</p>

| 방법 | 3,000스텝 정답률 (시드 범위) | 90% 도달 (시드 0 · 1 · 2) | 자리별 정답률 % (1 · 10 · 100 · 1000 · $) |
| --- | ---: | ---: | --- |
| 정답 SFT | 100.0% (100.0~100.0) | 820 · 1,140 · 680 | <span class="ans">100 · 100 · 100 · 100 · 100</span> |
| 정답 끼워 넣기 | 69.7% (9.7~100.0) | 1,800 · 못 닿음 · 1,480 | <span class="ans">100 · 100 · 70 · 94 · 100</span> |
| 칸마다 채점 | 23.9% (5.5~36.3) | 못 닿음 · 못 닿음 · 못 닿음 | <span class="ans">82 · 69 · 49 · 80 · 100</span> |

### 왜 이렇게 됐나

- **쉬운 문제부터 — 가중치가 한 번도 안 움직였어요.** 0~9 끼리의 쉬운 문제도 답은
  다섯 글자예요(7 + 5 → `2100$`). 형식부터 모르는 0% 학생은 한 장도 다 맞히지
  못하고, 8장이 다 0점이면 advantage 가 0 이라 기울기도 0 이에요. 1,000스텝 뒤
  자리별 정답률이 처음과 똑같아요. 쉬운 문제부터 가르치는 건 "가끔은 맞히는"
  학생에게만 통해요.
- **부분 점수(답안 하나에) — 쉬운 칸만 배웠어요.** 끝 표시 `$` 는 100% 가 됐고,
  1000의 자리는 세 시드 모두 정확히 50.2% 에 멈췄어요. 평가 문제 중 합이 1,000
  이상인 비율이 50.2% 라서, 항상 1 을 쓰는 것으로 보여요. 1·10·100의 자리는 10%,
  곧 찍기 수준이에요. 점수가 답안 전체에 숫자 하나라서, 1의 자리를 우연히 맞힌
  +0.2 가 다섯 글자 모두에 똑같이 나눠져요. 다른 칸이 우연히 맞고 틀린 잡음에
  묻혀서 어느 칸 덕분인지 가려지지 않아요.
- **칸마다 채점 — 자리별로는 배우지만 느려요.** 1의 자리를 맞힌 답안은 1의 자리
  글자만 올라가요. 그래서 부분 점수와 달리 자리별 정답률이 실제로
  올라요(3,000스텝에 1·10·100의 자리 82 · 69 · 49%). 하지만 맞는 숫자를
  우연히(10%) 써 봐야 배우는 시행착오라, 전체 정답은 3,000스텝에도 24% 예요. 실제
  LLM 의 과정 보상(process reward)을 아주 작게 만든 셈이에요.
- **정답 끼워 넣기 — 되지만 SFT 보다 2배쯤 느려요.** 3,000스텝이면 두 시드는
  99~100% 예요(90% 도달 1,480 · 1,800스텝). 한 시드는 9.7% 인데, 이 시드는 정답
  SFT 에서도 가장 느렸어요. 처음엔 학생이 쓴 7장이 전부 틀려서 똑같이 −0.35 씩
  내려가요. 학생이 확률대로 쓴 답들을 고르게 내리는 건 평균적으로 아무 방향도
  아니에요(확률대로 뽑은 답의 log 확률 기울기는 평균이 0). 그래서 실제로 배우는 건
  +2.47 을 받은 정답 32장뿐이고, 256장이 모두 정답인 SFT 에 비하면 데이터가 1/8
  이고 잡음이 섞여 있어요. 학생이 스스로 맞히기 시작하면 그때부터 진짜 RLVR 이
  돼요. 마지막엔 학생이 직접 쓴 답의 99% 가 정답이었어요.
- **정답 SFT — 가장 빨라요.** 1,000스텝에서는 시드에 따라 17~100% 였고,
  3,000스텝이면 세 시드 모두 100% 예요. 90% 에 닿은 건 680~1,140스텝으로, 실험 2
  의 OPD(520~840스텝)와 비슷해요.

<div class="rl-callout is-decision" markdown="1">
<span class="rl-callout__title">0% 학생에게는 정답을 직접 보여 줘야 해요</span>
정답 SFT, OPD, 정답 끼워 넣기처럼 "무엇이 정답인지" 를 보여 주는 방법만 제대로
배웠어요. 채점 방식만 바꾸는 방법은 우연히라도 맞힐 수 있는 단위(칸마다)로 쪼갤
때만 조금 배우고, 다섯 글자를 다 맞혀야 하는 방식은 문제를 쉽게 해도 아무것도 못
배워요. 그래서 실제로는 cold-start SFT 로 "가끔은 맞히는" 상태를 먼저 만들고 RL 을
해요. 정답 끼워 넣기는 RL 답안 묶음에 더 강한 모델의 답을 섞는 연구(예: LUFFY)와
같은 발상이고, 칸마다 채점은 과정 보상 모델(PRM)과 같은 발상이에요.
</div>

<div class="rl-callout is-warn" markdown="1">
<span class="rl-callout__title">이 실험의 한계</span>
- 시드 3개, 3,000스텝까지만 봤어요. 정답 끼워 넣기의 느린 시드와 칸마다 채점이 더
  오래 돌리면 어디까지 가는지는 확인하지 않았어요.
- 정답 끼워 넣기는 끼워 넣은 답을 학생 답처럼 그대로 썼어요. 실제 연구는 학생이 쓴
  답이 아니라는 점을 보정(importance weighting 등)해요.
- 칸마다 채점은 이 장난감이 자리마다 정답을 알아서 가능했어요. 실제 추론에는
  단계별 정답이 없어서 과정 보상 모델을 따로 학습해야 해요.
</div>

## 실험 6 — 백만 단위 사칙연산, 선생 4명 {#exp6-ko}

**알고 싶은 것:** 실험 1~5 는 세 자리 덧셈·뺄셈이라 너무 단순하고 정답이 공짜라는
한계가 있었어요. 문제를 백만 단위 사칙연산으로 키우고 연산별 선생 4명을 두면 MOPD
가 다른 방법보다 확실히 앞설까요?<br>
**설정:** dev GPU(H100 NVL 2장)에서 작업 61개를 약 3시간 40분 돌렸어요. 학생
실행마다 10,000스텝, 시드 3개예요. 코드는 `big/bigsim.py` 하나예요(여기에는 싣지
않았어요).

| | 실험 1~5 | 실험 6 |
| --- | --- | --- |
| 수 | 0~999 | 0~999,999. 두 수의 자리 수를 1~6 에서 고르게 뽑아요 |
| 연산 | + − | + − × ÷ (나눗셈은 몫과 나머지) |
| 답 | 네 자리 + `$` | 12자리 + `$`. 덧셈·뺄셈·곱셈은 1의 자리부터, 나눗셈은 몫 → 나머지를 큰 자리부터(긴 나눗셈 순서) |
| 학생 · 선생 | 10만 · 80만 파라미터 | 320만 · 1,070만 파라미터 |
| 평가 | 문제 2,000개 | 연산 × 자리 수(1~6) 24칸 × 250문제 = 6,000문제 |

### "정답이 공짜" 를 줄이려고 넣은 것

- **정답 SFT 기준선:** 정답 풀이를 그대로 줄 수 있을 때의 결과예요. 다른 방법이
  이걸 넘어야 의미가 있어요.
- **RL 로 다듬은 선생:** 정답 SFT 를 10,000스텝만 하고 RLVR 을 20,000스텝 더 한
  선생이에요. 실제 선생이 만들어지는 방식에 가까워요.
- **체계적으로 틀리는 선생:** 1000의 자리로 가는 받아올림을 늘 빼먹는 덧셈
  선생이에요. 무작위가 아니라 같은 실수를 반복해요.
- **가끔 틀리는 채점기:** RL 에서 O/X 를 10% 확률로 뒤집어요.

### 선생은 거의 완벽해요

| 선생 | 정답률 % (자기 연산, 네 연산 선생은 + · − · × · ÷) | 자리 수별 % (1~6) | 학습 시간 |
| --- | --- | --- | ---: |
| 덧셈 (정답 SFT) | 100.0 | 100 · 100 · 100 · 100 · 100 · 100 | 35분 |
| 뺄셈 (정답 SFT) | 100.0 | 100 · 100 · 100 · 100 · 100 · 100 | 35분 |
| 곱셈 (정답 SFT) | 99.8 | 100 · 100 · 100 · 100 · 100 · 99 | 35분 |
| 나눗셈 (정답 SFT) | 97.1 | 100 · 100 · 100 · 100 · 98 · 86 | 35분 |
| 네 연산 선생 (학습량 4배) | 100.0 · 100.0 · 100.0 · 97.3 | — | 138분 |
| 곱셈 (RL 로 다듬음) | 99.6 | 100 · 100 · 100 · 100 · 100 · 98 | 142분 |
| 나눗셈 (RL 로 다듬음) | 90.5 | 100 · 100 · 99 · 95 · 84 · 65 | 69분 |
| 덧셈 (받아올림 버그) | 80.4 | 100 · 100 · 80 · 70 · 71 · 61 | 36분 |

1,070만 파라미터 선생이 6자리 × 6자리 곱셈을 풀이 없이 99.8% 맞혀요. 1의 자리부터
쓰는 형식과 사실상 끝없는 문제 수 덕분으로 보여요. 네 연산을 다 배운 선생(학습량
4배)도 덧셈·뺄셈·곱셈 100%, 나눗셈 97.3% 로 전문 선생만큼 잘해요. RL 로 다듬은
선생은 RL 20,000스텝이 정답률을 올리지 못했어요(곱셈 99.9 → 99.6%, 나눗셈 91.8 →
90.5%).

출발 학생은 정답 SFT 1,000스텝이에요. 덧셈·뺄셈은 이미 99.9~100% 이고, 곱셈은
28~32%, 나눗셈은 23~32% 에서 출발해요.

### 결과 — 증류 방법끼리는 거의 같고, RL 은 거의 못 배워요

| 방법 (10,000스텝) | 전체 정답률 (시드 범위) | 덧셈 · 뺄셈 % | 곱셈 · 나눗셈 % |
| --- | ---: | --- | --- |
| 출발 학생 | 64.3% (63.4~65.5) | 100.0 · 100.0 | 30.2 · 27.0 |
| 정답 SFT (기준선) | 82.5% (80.6~83.6) | 99.9 · 99.9 | 57.7 · 72.3 |
| SFT 증류 | 82.1% (80.6~83.6) | 100.0 · 100.0 | 58.2 · 70.3 |
| MOPD | 81.2% (80.4~81.9) | 100.0 · 100.0 | 55.1 · 69.6 |
| 선생 1명 OPD | 81.1% (79.8~81.9) | 100.0 · 99.9 | 58.4 · 66.1 |
| MOPD → RL | 78.0% (77.4~78.9) | 100.0 · 100.0 | 48.1 · 63.8 |
| Mix-RL | 65.6% (64.5~66.8) | 100.0 · 100.0 | 31.8 · 30.4 |
| Cascade-RL | 50.8% (43.5~55.0) | 69.8 · 76.1 | 26.9 · 30.6 |

<figure class="rl-chart" data-rl-bars="big"></figure>
<p class="rl-caption">그림 6-1. 10,000스텝 뒤 학생의 곱셈·나눗셈 정답률(시드 3개 평균).</p>

- **증류 네 가지가 정답 SFT 와 같아요.** 정답 SFT 82.5%, SFT 증류 82.1%, MOPD
  81.2%, 선생 1명 OPD 81.1% 예요. 선생이 거의 완벽하니 선생 답을 따라 쓰나 정답을
  따라 쓰나 같아요. MOPD 가 앞서지 않았어요.
- **MOPD 가 앞서지 않은 이유:** 이 실험에선 네 연산을 다 배운 선생도 전문 선생만큼
  잘해서, 과목마다 선생을 두는 이점이 없어요. 실험 4 에서 MOPD 가 이긴 건 덧셈
  선생이 뺄셈을 전혀 몰랐기 때문이에요. 답도 13칸으로 짧아서, 학생이 자기 실수
  뒤에서 이어 쓰는 법을 배우는 OPD 의 장점이 드러날 일이 적어요.
- **천장은 학생이에요.** 선생은 곱셈 99.8% 인데 학생은 어떤 방법으로든 58%
  근처예요. 큰 자리 곱셈에서 막혀요(아래 표). 증류 곡선은 마지막 2,000스텝에도
  0.9~2.3%p 씩 오르고 있어서, 10,000스텝은 아직 수렴 전이에요.
- **Mix-RL 은 거의 제자리예요.** 곱셈 30 → 32%, 나눗셈 27 → 30% 예요. 8장 중
  일부만 맞아서 배울 거리가 있는 문제가 3% 안팎이에요. 13칸을 다 맞혀야 1점인데
  학생이 큰 곱셈·나눗셈을 거의 못 맞혀서예요.
- **Cascade-RL 은 앞 과목을 잊어요.** 덧셈 → 뺄셈 → 곱셈 → 나눗셈 순서로
  2,500스텝씩 했더니, 덧셈이 100 → 69.8%, 뺄셈이 100 → 76.1% 로 내려갔어요.
  곱셈·나눗셈은 출발점 근처예요.
- **MOPD → RL 은 MOPD 만 하는 것보다 낮아요**(78.0 vs 81.2%). MOPD 5,000스텝 뒤에
  RL 5,000스텝이 +1.0~1.4%p 를 더했는데, 같은 5,000스텝을 MOPD 로 계속하면
  +4.0~4.9%p 였어요.

| 자리 수별 정답률 % (두 수 중 긴 쪽 1~6자리) | 곱셈 | 나눗셈 |
| --- | --- | --- |
| 선생 (정답 SFT) | 100 · 100 · 100 · 100 · 100 · 99 | 100 · 100 · 100 · 100 · 98 · 86 |
| 출발 학생 | 100 · 39 · 18 · 11 · 9 · 4 | 91 · 39 · 14 · 9 · 5 · 3 |
| 정답 SFT (기준선) | 100 · 99 · 65 · 35 · 27 · 20 | 100 · 98 · 85 · 73 · 42 · 35 |
| SFT 증류 | 100 · 99 · 66 · 38 · 26 · 20 | 100 · 98 · 85 · 67 · 40 · 32 |
| MOPD | 100 · 93 · 60 · 36 · 25 · 17 | 100 · 98 · 85 · 65 · 40 · 29 |
| 선생 1명 OPD | 100 · 94 · 69 · 41 · 27 · 20 | 100 · 97 · 80 · 58 · 35 · 26 |
| MOPD → RL | 100 · 84 · 46 · 25 · 19 · 14 | 100 · 97 · 75 · 54 · 32 · 25 |
| Mix-RL | 100 · 41 · 22 · 13 · 10 · 5 | 96 · 45 · 19 · 12 · 7 · 4 |
| Cascade-RL | 95 · 33 · 15 · 9 · 6 · 3 | 97 · 45 · 19 · 12 · 6 · 4 |

### 현실성 조건

| 조건 | 방법 | 이 조건 (시드 범위) | 기준 조건 |
| --- | --- | ---: | ---: |
| RL 로 다듬은 선생 | MOPD | 82.2% (81.1~83.1) | 81.2% (80.4~81.9) |
| | SFT 증류 | 82.0% (80.9~83.2) | 82.1% (80.6~83.6) |
| | MOPD → RL | 78.8% (77.5~79.5) | 78.0% (77.4~78.9) |
| 버그 선생 (덧셈만 봄) | MOPD | 덧셈 80.4% (80.4~80.4) | 덧셈 100.0% (99.9~100.0) |
| | SFT 증류 | 덧셈 80.3% (80.1~80.4) | 덧셈 100.0% (99.9~100.0) |
| | MOPD → RL | 덧셈 86.9% (80.4~100.0), 시드별 80.4 · 100.0 · 80.4 | 덧셈 100.0% (100.0~100.0) |
| 10% 틀리는 채점기 | Mix-RL | 63.1% (62.0~64.1) | 65.6% (64.5~66.8) |
| | Cascade-RL | 18.4% (10.8~29.3) | 50.8% (43.5~55.0) |
| | MOPD → RL | 77.1% (76.5~77.8) | 78.0% (77.4~78.9) |

<figure class="rl-chart" data-rl-bars="real"></figure>
<p class="rl-caption">그림 6-2. 기준 조건과 버그 선생(덧셈 정답률), 10% 틀리는 채점기(전체 정답률) 비교. 시드 3개 평균이에요.</p>

- **RL 로 다듬은 선생 — 학생 결과는 같아요.** 선생 나눗셈이 90.5% 로 정답 SFT
  선생(97.1%)보다 낮지만, 학생 나눗셈은 70% 근처라 선생 차이가 학생 천장
  위에서만 생겼어요.
- **버그 선생 — 증류는 실수를 그대로 옮겨요.** 출발 학생은 덧셈 99.9~100% 였는데,
  MOPD 와 SFT 증류 모두 덧셈이 80.4% 로 내려갔어요. 버그 선생의 점수 그대로예요.
  학생이 원래 알던 걸 선생의 실수로 덮어썼어요.
- **버그 선생 + RL — 세 시드 중 하나만 고쳐졌어요.** 시드 1 은 RL 을 시작하고
  1,000스텝 만에 덧셈이 99.8% 로 돌아왔고 끝에는 100% 였지만, 시드 0·2 는 끝까지
  80.4% 였어요. 학생이 버그 답을 자신 있게 써서 8장이 다 똑같이 틀리면 신호가
  없어요. 우연히 받아올림을 맞게 쓴 답안이 나와야 고쳐져요. 실험 2·5 와 같은
  이유예요.
- **가끔 틀리는 채점기 — Cascade-RL 이 무너졌어요.** 덧셈·뺄셈 단계(이미 ~100%)에서
  "배울 거리가 있는 문제" 가 57% 로 보였어요. 깨끗한 채점기에선 0.7~2.1% 였으니
  거의 전부 가짜 신호예요. 곱셈 단계로 넘어가면 대부분 틀린 답안 중 하나가 O 로
  뒤집혀 +2.47 로 크게 밀리면서, 5,250~6,000스텝에 덧셈까지 50% 아래로
  무너졌어요. Mix-RL 은 무너지지 않았지만 곱셈·나눗셈이 출발점보다 낮아졌고, MOPD →
  RL 은 OPD 로 배운 게 있어서 거의 버텼어요.

### RL 학습률 — 3e-4 면 모델이 망가져요

처음엔 OPD 와 같은 학습률 3e-4 로 RL 을 돌렸어요. 그랬더니 SFT 로 99.9% 였던 곱셈
선생이 RL 1,000스텝 만에 80% 대로 떨어졌어요. 배울 거리가 있는 문제가 몇 % 뿐이라
RL 기울기는 작고 잡음이 큰데, Adam 은 파라미터마다 걸음 크기를 맞춰서 그 잡음도
3e-4 크기의 걸음으로 만들어요. OPD 는 글자마다 신호가 촘촘해서 같은 학습률로도
괜찮았어요.

<figure class="rl-chart" data-rl-bars="lr"></figure>
<p class="rl-caption">그림 6-3. RL 학습률 3e-4 와 3e-5 의 결과. 선생: SFT 10,000스텝 뒤 RL 20,000스텝을 한 정답률(SFT 끝은 곱셈 99.9%, 나눗셈 91.8%). 학생: 시드 0.</p>

| 정답률 % | RL 학습률 3e-4 | RL 학습률 3e-5 |
| --- | --- | --- |
| 곱셈 선생: SFT 끝 → RL 20,000스텝 뒤 | 99.9 → 77.5 | 99.9 → 99.6 |
| 나눗셈 선생: SFT 끝 → RL 20,000스텝 뒤 | 91.8 → 58.3 | 91.8 → 90.5 |
| Mix-RL 학생 (시드 0): 곱셈 · 나눗셈 | 14.9 · 18.3 | 32.3 · 35.0 |
| Cascade-RL 학생 (시드 0): 덧셈 · 뺄셈 | 0.1 · 2.3 | 75.3 · 82.8 |
| 짧은 탐색: Mix-RL 1,200스텝 뒤 곱셈 (출발 30.3) | 16.3 (1e-4: 26.9) | 29.5 (1e-5: 30.5) |

<p class="rl-caption">마지막 줄은 <code>big/rl_lr_test.py</code> 로그 값이에요(시드 0 출발 학생, 1,200스텝).</p>

학습률만 3e-5 로 낮추니 무너지지 않았어요. 위 결과(실험 6 의 RL 이 들어간 실행
전부)는 모두 3e-5 로 다시 돌린 거예요. 실제 LLM 의 RL 학습률이 1e-6 안팎으로 SFT
보다 훨씬 작은 이유와 같아요.

<div class="rl-callout is-decision" markdown="1">
<span class="rl-callout__title">MOPD 가 훨씬 높아질까? — 이 설정에선 아니었어요</span>
MOPD 의 이점은 세 조건에서 커져요. ① 선생 한 명이 모든 과목을 잘하지 못할 때(실험
4, 실제로는 수학·코드·에이전트처럼 과목별로 RL 한 선생들의 강점이 다를 때), ② 답이
길어서 학생의 실수가 뒤로 번질 때, ③ 학생이 커서 선생에 가까이 갈 수 있을 때예요.
실험 6 은 세 조건이 모두 약했어요. 다음으로 키울 곳은 풀이 과정(CoT)을 쓰게 해서
답을 길게 만들거나, 학생을 키우거나, 네 연산 선생의 크기를 줄여 과목끼리 부딪히게
만드는 거예요.
</div>

<div class="rl-callout is-warn" markdown="1">
<span class="rl-callout__title">이 실험의 한계</span>
- 학생이 작고(320만) 10,000스텝이라 수렴 전이에요. 더 오래 돌리면 방법끼리 차이가
  커지거나 줄 수 있어요.
- 답만 바로 쓰는 형식이에요. 풀이 과정이 없어서 실제 LLM 의 긴 추론과는 달라요.
- 선생은 시드 1개로 한 번씩만 만들었어요. 학생 쪽만 시드 3개예요.
- RL 이 들어간 실행은 처음에 3e-4 로 돌다가 중간에 3e-5 로 바꿔 다시 돌렸어요.
  3e-4 결과는 위 학습률 비교에만 썼어요.
</div>

## 실제 LLM 학습과 같은 점, 다른 점 {#real-ko}

이 실험은 개념을 눈으로 보려고 만든 장난감이에요. 방향은 참고하되 숫자를 실제
모델에 그대로 옮기면 안 돼요.

| 항목 | 이 실험 | 실제 LLM |
| --- | --- | --- |
| 답 길이 | 다섯 글자 | 수백~수만 토큰의 추론 |
| SFT 증류 vs OPD | 거의 같아요(SFT 가 조금 빨라요). 자기 실수에서 이어 쓸 일이 없어서요 | 긴 추론에서는 OPD 가 낫다는 보고가 많아요. 학생이 자기가 실제로 만드는 실수 상태에서 배워서요 |
| 안전장치 | KL 제약 없음, 학습률 하나로 고정 | KL 제약, 낮은 학습률, 데이터 섞기 |
| 가장 비싼 방법 | SFT(매 스텝 선생이 답을 씀). RLVR 은 채점기만 있어서 가장 싸요 | RL. 문제마다 긴 답을 여러 번 생성해야 해요. SFT 데이터는 미리 한 번 만들어 둬요 |
| OPD 조건 | 선생과 학생이 같은 글자표(14개)를 써요 | 선생과 학생의 토크나이저가 같아야 하고, 선생의 글자별 확률(logprob)을 받을 수 있어야 해요 |
| 채점기 | 정답과 글자까지 같으면 1 | 수학은 답 동치 비교, 코드는 테스트 통과, 글쓰기는 루브릭 + LLM 채점 |

## 코드 해설 — rlsim.py 를 한 줄씩 {#code-ko}

[`rlsim.py`]({{ '/files/rl-toy/rlsim.py' | relative_url }}) 는 PyTorch 와 NumPy 만
쓰는 364줄짜리 파일 하나예요. 아래 코드는 파일에서 그대로 잘라 왔고, 캡션의 숫자가
파일의 줄 번호예요. 텐서 모양과 값은 같은 코드를 실제로 돌려서 뽑았어요. 공개한
파일은 주석을 영어로 옮겼고, 줄 번호와 동작은 그대로예요.

### 0. 파일 지도 {#code-map-ko}

<div class="rl-flow">
  <div class="rl-flow__step"><b>main()</b><span>명령줄 인자 읽기</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step"><b>load_or_train_teachers()</b><span>선생 셋 만들기 (또는 불러오기)</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step"><b>experiment()</b><span>실험 하나 = 출발 학생 × 학습법들 × 시드 3개</span></div>
  <div class="rl-flow__arrow">→</div>
  <div class="rl-flow__step is-here"><b>post_train()</b><span>스텝마다 step_sft / step_rlvr / step_opd</span></div>
</div>
<p class="rl-caption">그림 7. 호출 순서. 파란 칸 안에서 학습법이 갈려요.</p>

| 줄 | 함수 | 하는 일 |
| --- | --- | --- |
| 30–38 | 상수 | 글자표, 길이, 배치, 학습률 |
| 45–66 | `problems` · `mixed` · `correct` | 문제 만들기, 과목 섞기, 채점기 |
| 72–118 | `TinyGPT` · `answer_logits` · `write` · `accuracy` | 모델, 답 자리 예측 꺼내기, 답 쓰기, 정답률 |
| 124–174 | `update` · `step_sft` · `teacher_tokens` · `step_rlvr` · `step_opd` | 학습법 한 스텝씩 |
| 180–220 | `sft_until` · `post_train` | 목표까지 SFT, 학습 루프 |
| 226–347 | `test_sets` · `load_or_train_teachers` · `warm_student` · `experiment` | 평가 문제, 선생, 출발 학생, 실험 정의 |

### 1. 글자와 문제 — `VOCAB`, `problems()`, `mixed()` {#code-data-ko}

```python
VOCAB = list("0123456789+-=$")  # "$" = 답 끝
STOI = {c: i for i, c in enumerate(VOCAB)}
PROMPT_LEN, ANSWER_LEN = 8, 5  # "123+045=" 다음에 네 자리를 1의 자리부터, 그리고 "$"
SEQ_LEN = PROMPT_LEN + ANSWER_LEN
BATCH = 256  # 한 스텝에 학생이 쓰는 답안 수 — 모든 방법이 같다
GROUP = 8  # RLVR: 한 문제에 쓰는 답안 수 (32문제 × 8번)
LR = 5e-4
LOG_EVERY = 20
WARM_TARGET = 0.30  # 학생의 출발점: 정답률 30% (사전학습 + 약간의 SFT 를 흉내)
```

<p class="rl-caption">rlsim.py, 30~38줄</p>

```python
def problems(n: int, op: str, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """n개의 문제(프롬프트)와 정답을 토큰 id 로. 뺄셈은 큰 수에서 작은 수를 뺀다."""
    a, b = rng.integers(0, 1000, n), rng.integers(0, 1000, n)
    if op == "-":
        a, b = np.maximum(a, b), np.minimum(a, b)
    c = a + b if op == "+" else a - b
    prompts = [[STOI[ch] for ch in f"{x:03d}{op}{y:03d}="] for x, y in zip(a, b)]
    answers = [[STOI[ch] for ch in f"{z:04d}"[::-1] + "$"] for z in c]
    return torch.tensor(prompts), torch.tensor(answers)


def mixed(n: int, ops: list[str], rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """n문제를 과목별로 똑같이 나눠서. 셋째 값은 행마다의 과목."""
    per = n // len(ops)
    parts = [problems(per, op, rng) for op in ops]
    subjects = [op for op in ops for _ in range(per)]
    return torch.cat([p for p, _ in parts]), torch.cat([a for _, a in parts]), subjects
```

<p class="rl-caption">rlsim.py, 45~61줄</p>

- `VOCAB` 은 글자 14개, `STOI` 는 글자 → 번호예요. `0`~`9` 는 0~9, `+` 10, `-`
  11, `=` 12, `$` 13.
- `f"{x:03d}{op}{y:03d}="` 는 세 자리로 맞춰요. 45 + 7 은 `045+007=` 이 돼요.
  그래서 문제는 항상 8글자예요.
- `f"{z:04d}"[::-1] + "$"` 는 네 자리로 채운 뒤 뒤집어요. 답이 항상 5글자예요.
- 뺄셈은 `np.maximum` / `np.minimum` 으로 큰 수를 앞에 둬요.
- `mixed()` 는 n 을 과목 수로 나눠 앞 절반은 덧셈, 뒤 절반은 뺄셈으로 붙여요. 셋째
  값 `subjects` 는 행마다의 과목표예요. MOPD 가 이걸 보고 선생을 골라요.

```text
699+684=  →  [6, 9, 9, 10, 6, 8, 4, 12]   prompts 한 줄 (8칸)
1383      →  "1383" 뒤집기 → "3831" + "$"
          →  [3, 8, 3, 1, 13]              answers 한 줄 (5칸)
```

### 2. 채점기 — `correct()` {#code-verifier-ko}

```python
def correct(seq: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    """검증기: 쓴 답이 정답과 글자까지 같으면 1, 아니면 0."""
    return (seq[:, PROMPT_LEN:] == answers).all(1).float()
```

<p class="rl-caption">rlsim.py, 64~66줄</p>

```text
seq                   (256, 13)  문제 8칸 + 답 5칸
seq[:, PROMPT_LEN:]   (256, 5)   답 5칸만
== answers            (256, 5)   칸마다 True / False
.all(1)               (256,)     다섯 칸이 전부 True 인가
.float()              (256,)     1.0 또는 0.0
```

끝 표시 `$` 까지 비교해요. 숫자 넷을 맞혀도 `$` 대신 다른 글자를 쓰면 0점이에요.

### 3. 모델 — `TinyGPT` {#code-model-ko}

```python
class TinyGPT(nn.Module):
    def __init__(self, layers: int, width: int, heads: int = 4):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), width)
        self.pos = nn.Embedding(SEQ_LEN, width)
        block = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, len(VOCAB))
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(SEQ_LEN), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.tok(x) + self.pos.weight[:t]
        h = self.blocks(h, mask=self.mask[:t, :t], is_causal=True)
        return self.head(self.norm(h))


def student() -> TinyGPT:
    return TinyGPT(layers=2, width=64)


def teacher() -> TinyGPT:
    return TinyGPT(layers=4, width=128)
```

<p class="rl-caption">rlsim.py, 72~97줄</p>

- `self.tok` 는 글자 번호를 64차원 벡터로 바꿔요(글자 임베딩). `self.pos` 는
  0~12번째 칸마다 학습되는 벡터예요(위치 임베딩). 둘을 더한 게 입력이에요.
- `nn.TransformerEncoderLayer` 에 causal mask(`generate_square_subsequent_mask`)를
  주면 각 칸이 자기와 앞 칸만 봐요. 이름은 Encoder 지만 이렇게 쓰면 GPT 와 같은
  decoder-only 구조예요.
- `norm_first=True` 는 LayerNorm 을 attention·MLP 앞에 두는 pre-LN 방식이에요
  (GPT-2 이후 표준). dropout 은 0 이에요.
- `self.head` 는 칸마다 글자 14개의 점수(로짓)를 내요. 출력 모양은 (배치, 칸 수,
  14) 예요.
- 학생과 선생은 같은 클래스이고 크기만 달라요. 같은 글자표를 쓰니까 선생의 확률을
  학생 확률과 칸마다·글자마다 바로 비교할 수 있어요. OPD 가 되는 조건이에요.

| 부분 | 학생 (2층, 폭 64) | 선생 (4층, 폭 128) | 학생 계산식 |
| --- | ---: | ---: | --- |
| 글자 임베딩 | 896 | 1,792 | 14 × 64 |
| 위치 임베딩 | 832 | 1,664 | 13 × 64 |
| attention | 33,280 | 264,192 | 2층 × (64×192+192 + 64×64+64) |
| MLP | 66,176 | 526,848 | 2층 × (64×256+256 + 256×64+64) |
| LayerNorm (층 안) | 512 | 2,048 | 2층 × 2개 × (64+64) |
| 마지막 LayerNorm | 128 | 256 | 64 + 64 |
| 출력층 | 910 | 1,806 | 64 × 14 + 14 |
| **합계** | **102,734** | **798,606** | |

### 4. 답 자리의 예측만 꺼내기 — `answer_logits()` {#code-slice-ko}

```python
def answer_logits(model: TinyGPT, seq: torch.Tensor) -> torch.Tensor:
    """답 다섯 글자 각각을 쓰기 직전의 예측 (batch, 5, vocab)."""
    return model(seq)[:, PROMPT_LEN - 1 : -1]
```

<p class="rl-caption">rlsim.py, 100~102줄</p>

GPT 는 칸 i 의 출력으로 칸 i+1 의 글자를 맞혀요. 그래서 답 첫 글자(칸 8)의 예측은
칸 7(`=`)의 출력에 있어요. 한 칸씩 밀려 있는 게 핵심이에요.

```text
  0  1  2  3  4  5  6  7  8  9 10 11 12   ← 칸
  6  9  9  +  6  8  4  =  3  8  3  1  $   ← 들어가는 글자
  9  9  +  6  8  4  =  3  8  3  1  $  -   ← 이 칸의 출력이 맞히는 글자
                       ^  ^  ^  ^  ^      ← [:, PROMPT_LEN-1 : -1] = 칸 7~11
```

`model(seq)` 는 (배치, 13, 14) 이고, `[:, 7:12]` 로 자르면 (배치, 5, 14) 예요. 칸
12(`$` 다음)의 출력은 맞힐 글자가 없어서 버려요. SFT · RLVR · OPD 가 모두 이 함수
하나로 "답 다섯 칸에서 학생이 건 확률" 을 얻어요.

### 5. 답 쓰기 — `write()` {#code-write-ko}

```python
@torch.no_grad()
def write(model: TinyGPT, prompts: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    """모델이 답을 한 글자씩 쓴다. greedy=False 면 확률대로 뽑는다 (temperature 1)."""
    seq = prompts
    for _ in range(ANSWER_LEN):
        logits = model(seq)[:, -1]
        nxt = logits.argmax(-1) if greedy else torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return seq


def accuracy(model: TinyGPT, test: tuple[torch.Tensor, torch.Tensor]) -> float:
    prompts, answers = test
    return correct(write(model, prompts, greedy=True), answers).mean().item()
```

<p class="rl-caption">rlsim.py, 105~118줄</p>

- **`@torch.no_grad()`:** 쓰는 동안은 기울기를 기록하지 않아요. 학습에 필요한
  확률은 다 쓴 답안을 `answer_logits()` 로 **한 번 더 읽어서** 계산해요. 실제 RL
  프레임워크도 같아요. 답안은 추론 엔진(vLLM 등)이 만들고, 학습 쪽이 그 답안을 다시
  읽어 확률을 계산해요.
- **다섯 번 반복:** 매번 지금까지의 `seq` 전체를 모델에 넣고 마지막 칸의
  로짓(`[:, -1]`)만 써요. 이전 계산을 저장해 두는 KV 캐시가 없어서 앞부분을 매번
  다시 계산하지만, 13칸이라 괜찮아요.
- **`greedy=True` → `argmax`:** 가장 높은 글자만 써요. 평가와 선생 답안에 써요.
- **`greedy=False` → `torch.multinomial(softmax)`:** 확률대로 뽑아요(temperature
  1). RLVR · OPD 의 학생 답안에 써요. 학생이 30% 로 믿는 글자는 30% 확률로 나와요.
  같은 문제를 8번 풀면 답이 갈리는 이유예요.

### 6. 업데이트 — `update()` {#code-update-ko}

```python
def update(opt: torch.optim.Optimizer, model: TinyGPT, loss: torch.Tensor) -> None:
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

<p class="rl-caption">rlsim.py, 124~128줄</p>

세 방법 모두 이 네 줄로 파라미터를 움직여요. 다른 건 손실뿐이에요. 기울기 전체
길이가 1 을 넘으면 1 로 줄여서(clip) 한 스텝이 너무 크게 튀지 않게 해요.

### 7. SFT 증류 한 스텝 — `step_sft()`, `teacher_tokens()` {#code-sft-ko}

```python
def step_sft(model: TinyGPT, opt, prompts: torch.Tensor, targets: torch.Tensor) -> dict:
    """주어진 답을 글자마다 따라 쓰게 한다 (cross-entropy)."""
    seq = torch.cat([prompts, targets], 1)
    loss = F.cross_entropy(answer_logits(model, seq).reshape(-1, len(VOCAB)), targets.reshape(-1))
    update(opt, model, loss)
    return {}


def teacher_tokens(prompts: torch.Tensor, subjects: list[str], teachers: dict[str, TinyGPT]) -> torch.Tensor:
    """과목마다 그 과목 선생이 쓴 모범답안 (greedy)."""
    targets = torch.empty(len(prompts), ANSWER_LEN, dtype=torch.long)
    for op, model in teachers.items():
        rows = torch.tensor([s == op for s in subjects])
        if rows.any():
            targets[rows] = write(model, prompts[rows], greedy=True)[:, PROMPT_LEN:]
    return targets
```

<p class="rl-caption">rlsim.py, 131~146줄</p>

| 줄 | 코드 | 모양 | 뜻 |
| --- | --- | ---: | --- |
| 213 | `mixed(BATCH, ops, rng)` | <span class="ans">(256, 8)</span> | 문제 256개 × 글자 8개 |
| 145 | `teacher_tokens(prompts, subjects, teachers)` | <span class="ans">(256, 5)</span> | 선생이 쓴 답 256장 × 5글자 |
| 133 | `torch.cat([prompts, targets], 1)` | <span class="ans">(256, 13)</span> | 문제 뒤에 선생 답을 붙인 것 |
| 134 | `answer_logits(model, seq)` | <span class="ans">(256, 5, 14)</span> | 답 5칸 × 글자 14개의 점수 |
| 134 | `.reshape(-1, len(VOCAB))` | <span class="ans">(1,280, 14)</span> | 1,280글자를 한 줄로 펴서 cross_entropy 에 |

- **143줄 `rows`:** 과목표가 이 선생 과목인 행만 True 인 마스크예요. 그 행만
  선생에게 greedy 로 쓰게 해요. 선생이 둘이면 행을 나눠 두 번 써요.
- **133줄 `torch.cat`:** 문제 뒤에 선생 답을 붙여요. 학생은 이걸 읽으면서 칸마다
  다음 글자를 맞혀요. 앞부분이 항상 선생 글자라서(teacher forcing) 학생은 실제로
  쓰지 않아요. 학생이 앞 칸에서 실수했을 때 어떻게 이어 쓸지는 배울 기회가 없어요.
- **134줄 `F.cross_entropy`:** 256 × 5 = 1,280글자 각각의 −log(선생 글자에 건
  확률) 평균이에요.
- **정답은 안 써요:** `post_train()` 215줄에서 `answers` 대신 `teacher_tokens(...)`
  를 넘겨요.

### 8. RLVR 한 스텝 — `step_rlvr()` {#code-rlvr-ko}

```python
def step_rlvr(model: TinyGPT, opt, prompts: torch.Tensor, answers: torch.Tensor) -> dict:
    """GRPO: 한 문제에 8번 쓰고, 맞은 답은 그 글자들의 확률을 올리고 틀린 답은 내린다.
    8번이 전부 맞거나 전부 틀리면 비교할 게 없어서 그 문제에서는 배울 게 0이다."""
    prompts, answers = prompts.repeat_interleave(GROUP, 0), answers.repeat_interleave(GROUP, 0)
    seq = write(model, prompts)
    reward = correct(seq, answers).view(-1, GROUP)
    advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
    logp = answer_logits(model, seq).log_softmax(-1).gather(-1, seq[:, PROMPT_LEN:, None]).squeeze(-1)
    update(opt, model, -(advantage.view(-1, 1) * logp).mean())
    return {"signal": (reward.std(1) > 0).float().mean().item(), "reward": reward.mean().item()}
```

<p class="rl-caption">rlsim.py, 149~158줄</p>

| 줄 | 코드 | 모양 | 뜻 |
| --- | --- | ---: | --- |
| 210 | `mixed(BATCH // GROUP, ops, rng)` | <span class="ans">(32, 8)</span> | 문제 32개 |
| 152 | `prompts.repeat_interleave(GROUP, 0)` | <span class="ans">(256, 8)</span> | 문제마다 8장씩 붙여 256행 |
| 153 | `write(model, prompts)` | <span class="ans">(256, 13)</span> | 학생이 확률대로 쓴 답안 256장 |
| 154 | `correct(seq, answers).view(-1, GROUP)` | <span class="ans">(32, 8)</span> | 행 = 문제, 열 = 8번의 시도, 값은 0/1 |
| 155 | `advantage` | <span class="ans">(32, 8)</span> | 문제마다 (점수 − 평균) ÷ 표준편차 |
| 156 | `….gather(-1, 쓴 글자).squeeze(-1)` | <span class="ans">(256, 5)</span> | 답안마다 쓴 글자 5개의 log 확률 |
| 157 | `advantage.view(-1, 1)` | <span class="ans">(256, 1)</span> | logp 와 곱하면 다섯 글자에 똑같이 퍼져요 |

**152줄 `repeat_interleave` — 같은 문제 8장을 붙여 놓기.** `repeat` 와 헷갈리기
쉬워요.

```text
문제               [A, B, C]
repeat_interleave  [A, A, B, B, C, C]   ← 이걸 써요
repeat             [A, B, C, A, B, C]
```

154줄의 `.view(-1, 8)` 은 연속한 8행을 한 문제로 묶어요. 그래서 위 방식이어야 "한
행 = 한 문제의 8번 시도" 가 돼요. `repeat` 를 쓰면 서로 다른 문제의 답안끼리 평균을
내요. 에러 없이 조용히 틀리는 종류의 버그예요.

**155줄 advantage — 문제마다 평균과 표준편차.** `mean(1)`, `std(1)` 은 8번의 시도
방향(열)으로 계산해요. PyTorch 의 `std` 는 기본으로 n−1 로 나눠요. 227 + 189 처럼
8장 중 2장이 맞으면 평균 0.25, 표준편차 0.463 라서 맞은 답 +1.62, 틀린 답 −0.54
예요. n 으로 나눴다면 표준편차 0.433 라서 +1.73 / −0.58 가 됐을 거예요. 8장이 모두
같으면 표준편차가 0 이라 `1e-6` 을 더해 0 으로 나누기를 피하고, 분자도 0 이라
advantage 는 정확히 0 이에요.

**156줄 logp — 쓴 글자의 log 확률만 골라내기.** `log_softmax(-1)` 로 칸마다 글자
14개의 log 확률 (256, 5, 14) 을 만들고, `gather` 로 실제로 쓴 글자 하나씩만 뽑아
(256, 5) 가 돼요.

**157줄 손실.** advantage (256, 1) 를 logp (256, 5) 에 곱하면 답안 하나의
advantage 가 다섯 글자에 똑같이 곱해져요. 앞의 − 는 옵티마이저가 손실을 *줄이는*
쪽으로 움직이기 때문이에요. 손실을 줄이는 것 = advantage × log 확률을 키우는 것이라,
advantage 가 + 인 답안의 글자는 확률이 오르고 − 인 답안의 글자는 내려가요.

이게 policy gradient 예요. 답안 전체가 나올 확률의 log 는 글자 다섯 개 log 확률의
합이에요. 기대 점수의 기울기는 "점수 × log 확률의 기울기" 의 평균이고, 여기서 같은
문제 답안들의 평균 점수(baseline)를 빼도 기댓값은 그대로라 잡음만 줄어요.

**실제 값 — 227 + 189 의 8장**

| 답안 | 쓴 순서 (1의 자리부터) | 채점 · advantage | 칸별 log 확률 (1 · 10 · 100 · 1000 · $) |
| --- | --- | --- | --- |
| <span class="ans"><span class="bad">5</span>16</span> | <span class="ans">6 1 5 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−2.09</b> −0.00 −0.00</span> |
| <span class="ans">416</span> | <span class="ans">6 1 4 0 $</span> | <span class="mark-ok">O</span> +1.62 | <span class="lp">−0.01 −0.13 <b>−1.56</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">6</span>16</span> | <span class="ans">6 1 6 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−2.16</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">2</span>16</span> | <span class="ans">6 1 2 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−3.34</b> −0.00 −0.00</span> |
| <span class="ans">416</span> | <span class="ans">6 1 4 0 $</span> | <span class="mark-ok">O</span> +1.62 | <span class="lp">−0.01 −0.13 <b>−1.56</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">3</span>16</span> | <span class="ans">6 1 3 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−2.65</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">8</span>16</span> | <span class="ans">6 1 8 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 −0.13 <b>−1.65</b> −0.00 −0.00</span> |
| <span class="ans"><span class="bad">9</span><span class="bad">0</span>6</span> | <span class="ans">6 0 9 0 $</span> | <span class="mark-no">X</span> −0.54 | <span class="lp">−0.01 <b>−2.31</b> <b>−1.77</b> −0.00 −0.00</span> |

log 확률이 −0.00 인 칸은 학생이 이미 거의 100% 확신하는 글자예요. log 확률의
기울기는 (1 − 확률) 에 비례해서, 이런 칸은 advantage 가 커도 거의 안 움직여요.
실제로 움직이는 건 학생이 망설인 100의 자리예요.

**8장을 합치면 신호가 어디에 남나.** 앞에서 "틀린 516 은 맞은 글자까지 같이
내려간다" 고 했어요. 답안 한 장만 보면 맞는 말이에요. 그런데 손실은 8장의 합이라,
**앞부분이 같은 칸에서 같은 글자**를 쓴 답안들의 advantage 는 서로 더해져요.
앞부분이 같으면 학생 확률도 같은 값이라서요.

| 자리 | 앞부분 (쓴 순서) | 쓴 글자 × 장수 | advantage 합 |
| --- | --- | --- | ---: |
| 1의 자리 | — | <span class="ans">6</span> × 8장 | 0.00 |
| 10의 자리 | <span class="ans">6</span> | <span class="ans">1</span> × 7장 | **+0.54** |
| | <span class="ans">6</span> | <span class="ans">0</span> × 1장 | −0.54 |
| 100의 자리 | <span class="ans">6 1</span> | <span class="ans">4</span> × 2장 | **+3.24** |
| | <span class="ans">6 1</span> | <span class="ans">2</span> × 1장 | −0.54 |
| | <span class="ans">6 1</span> | <span class="ans">3</span> × 1장 | −0.54 |
| | <span class="ans">6 1</span> | <span class="ans">5</span> × 1장 | −0.54 |
| | <span class="ans">6 1</span> | <span class="ans">6</span> × 1장 | −0.54 |
| | <span class="ans">6 1</span> | <span class="ans">8</span> × 1장 | −0.54 |
| | <span class="ans">6 0</span> | <span class="ans">9</span> × 1장 | −0.54 |

- **1의 자리:** 8장 모두 6 을 썼어요. +1.62 × 2 + (−0.54) × 6 = 0 이라 정확히
  상쇄돼요. advantage 는 합이 항상 0 이라서, 모두가 똑같이 쓴 글자는 신호를 받지
  않아요.
- **10의 자리:** 7장이 1, 906 한 장만 0 을 썼어요. 1 은 +0.54, 0 은 −0.54 를
  받아요.
- **100의 자리 (앞이 "61" 로 같은 7장):** 4 는 +3.24, 나머지 다섯 글자는 −0.54
  씩이에요. 답이 갈린 칸에 신호가 모여요.
- **1000의 자리와 끝 표시:** 앞부분이 답안마다 달라서 상쇄되지 않지만, 학생이 이미
  거의 100% 확신하는 글자라 거의 움직이지 않아요.

그래서 O/X 하나만 받아도 여러 장을 비교하면 어느 칸이 문제였는지가 어느 정도
드러나요. 다만 699 + 684 처럼 8장이 다 틀리면 비교할 게 없어서 이것도 0 이에요.

**158줄 signal:** 8장의 표준편차가 0 보다 큰(= 다 같지는 않은) 문제의 비율이에요.
그림 3 의 "배울 거리가 있던 문제" 가 이 값이에요. 학습에는 쓰지 않고 기록만 해요.

### 9. OPD · MOPD 한 스텝 — `step_opd()` {#code-opd-ko}

```python
def step_opd(model: TinyGPT, opt, prompts: torch.Tensor, subjects: list[str], teachers: dict[str, TinyGPT]) -> dict:
    """OPD: 학생이 직접 쓰고, 학생이 쓴 앞부분을 그대로 둔 채 글자마다 선생의 확률 분포와
    학생의 분포 차이(reverse KL, 어휘 전체)를 줄인다. 과목마다 다른 선생이면 MOPD."""
    seq = write(model, prompts)
    s = answer_logits(model, seq).log_softmax(-1)
    t = torch.empty_like(s)
    with torch.no_grad():
        for op, model_t in teachers.items():
            rows = torch.tensor([x == op for x in subjects])
            if rows.any():
                t[rows] = answer_logits(model_t, seq[rows]).log_softmax(-1)
    kl = (s.exp() * (s - t)).sum(-1)
    update(opt, model, kl.mean())
    return {"signal": 1.0, "kl": kl.mean().item()}
```

<p class="rl-caption">rlsim.py, 161~174줄</p>

| 줄 | 코드 | 모양 | 뜻 |
| --- | --- | ---: | --- |
| 213 | `mixed(BATCH, ops, rng)` | <span class="ans">(256, 8)</span> | 문제 256개 |
| 164 | `write(model, prompts)` | <span class="ans">(256, 13)</span> | 학생이 확률대로 쓴 답안 256장 |
| 165 | `answer_logits(model, seq).log_softmax(-1)` | <span class="ans">(256, 5, 14)</span> | 칸마다 학생의 글자 14개 log 확률 (기울기 있음) |
| 171 | `answer_logits(model_t, seq[rows]).log_softmax(-1)` | <span class="ans">(256, 5, 14)</span> | 같은 답안을 선생이 읽은 log 확률 (기울기 없음) |
| 172 | `(s.exp() * (s - t)).sum(-1)` | <span class="ans">(256, 5)</span> | 14글자를 합쳐 칸마다 reverse KL 하나 |
| 173 | `kl.mean()` | <span class="ans">스칼라</span> | 1,280칸 평균 = 손실 |

- **164줄:** 학생이 문제마다 확률대로 한 번 써요. 기울기는 기록하지 않아요.
- **165줄:** 학생이 자기 답안을 다시 읽은 log 확률이에요. 여기로 기울기가 흘러요.
- **166–171줄:** 선생이 **학생이 쓴 답안**을 읽은 log 확률이에요. 선생은 쓰지 않고
  읽기만 해서 한 번에 다섯 칸이 다 나와요. `rows` 마스크로 과목마다 다른 선생에게
  보내면 MOPD 예요.
- **172줄:** reverse KL = Σ<sub>글자 14개</sub> 학생 확률 × (log 학생 확률 − log
  선생 확률). `s.exp()` 가 학생 확률, `s - t` 가 log 비율이에요. 표본 하나로 어림한
  값이 아니라 14글자를 전부 더한 정확한 값이에요.
- **173–174줄:** 256 × 5 칸의 평균이 손실이에요. 모든 칸이 신호를 받으니 signal 은
  항상 1 이에요.

OPD 와 MOPD, 실험 4 의 대조군은 같은 함수예요. `teachers` 에 무엇을 넣는지만
달라요.

```text
OPD   (실험 1·2)        {"+": add}
OPD   (실험 3)          {"+": add_weak}
MOPD  (실험 4)          {"+": add, "-": sub}
대조군 (실험 4)          {"+": add, "-": add}   # 뺄셈 문제도 덧셈 선생이 읽어요
```

**기울기가 어떻게 생겼나.** 손으로 미분하면, 칸 하나의 로짓 z<sub>c</sub> 에 대한
reverse KL 의 기울기는 아래와 같아요. autograd 결과와 비교해서 차이가 1.2×10⁻⁷
였어요.

```text
∂KL/∂z_c = p_학생(c) × ( log p_학생(c) − log p_선생(c) − KL )
```

경사하강은 기울기 반대로 가니까, 로짓 기준으로 **log 비율이 그 칸의 KL 보다 큰
글자는 내려가고, 작은 글자는 올라가요.** SFT 의 기울기는 p<sub>학생</sub>(c) − [c 가
선생 글자] 라서, 선생 글자 하나만 올리고 나머지는 학생 확률에 비례해 내려요.

**실제 값 — 699 + 684 의 100의 자리** (앞부분은 3, 8). 로짓을 기울기 방향으로 아주
조금 움직였을 때, 확률이 어느 글자에서 빠져서(−) 어느 글자로 가는지(+)를 비율로
나타냈어요. 숫자가 아닌 글자 넷은 0% 라 뺐어요.

| 글자 | 학생 → 선생 확률 | SFT 가 옮기는 몫 | OPD 가 옮기는 몫 |
| --- | --- | ---: | ---: |
| <span class="ans">0</span> | 22.4% → < 0.001% | <span class="down">−18%</span> | <span class="up">+2.5%</span> |
| <span class="ans">1</span> | 35.8% → < 0.001% | <span class="down">**−69%**</span> | <span class="down">**−100%**</span> |
| <span class="ans">2</span> | 20.3% → 0.004% | <span class="down">−13%</span> | <span class="up">**+34%**</span> |
| <span class="ans">3</span> | 10.6% → 99.2% | <span class="up">**+91%**</span> | <span class="up">**+47%**</span> |
| <span class="ans">4</span> | 2.3% → 0.7% | <span class="up">+2.0%</span> | <span class="up">+3.9%</span> |
| <span class="ans">5</span> | 1.7% → 0.002% | <span class="up">+1.5%</span> | <span class="up">+2.2%</span> |
| <span class="ans">6</span> | 0.7% → 0.003% | <span class="up">+0.7%</span> | <span class="up">+0.8%</span> |
| <span class="ans">7</span> | 0.9% → < 0.001% | <span class="up">+0.9%</span> | <span class="up">+1.2%</span> |
| <span class="ans">8</span> | 1.1% → 0.006% | <span class="up">+1.0%</span> | <span class="up">+1.4%</span> |
| <span class="ans">9</span> | 4.2% → 0.016% | <span class="up">+3.0%</span> | <span class="up">+6.9%</span> |

- **SFT:** 들어오는 확률의 91% 가 정답 3 으로 가요. 빠지는 쪽은 학생 확률에
  비례해서 1(69%), 0(18%), 2(13%) 예요. 나머지 9% 는 확률이 작던 4~9 에 조금씩
  가요. 이 글자들도 로짓으로는 내려가지만 1·0·2 가 훨씬 크게 내려가서 확률로는 조금
  올라가요. 확률은 글자끼리 나눠 갖는 거라 늘 상대적이에요.
- **OPD:** 빠지는 확률은 **전부 1** 에서 나와요. 1 은 학생이 실제로 쓴 글자이고,
  선생은 1 에 거의 0 을 줘서 log 비율이 13.75 로 가장 커요. 들어오는 쪽은 3(47%)이
  가장 많지만 **2(34%)도 올라가요**. 0 은 로짓으로는 내려가지만 1 이 훨씬 크게
  내려가는 바람에 확률은 거의 그대로예요(+2.5%).

선생이 0.004% 만 주는 2 가 왜 올라갈까요? 2 의 log 비율(8.53)이 이 칸의 KL(9.48)
보다 작아서예요. 지금 가장 큰 문제인 1 의 확률을 다른 글자로 넘기는데, 이미 확률이
큰 2 가 많이 받아요. 1 이 줄어 KL 이 내려가면 2 도 내려가는 쪽으로 바뀌어요. 아래는
이 칸의 로짓 14개만 보폭 0.5 로 직접 움직여 본 사고 실험이에요.

| 걸음 | SFT: 글자 1 · 2 · 3 의 확률 (%) | OPD: 글자 1 · 2 · 3 의 확률 (%) |
| --- | --- | --- |
| 0 | <span class="ans">36 · 20 · 11</span> | <span class="ans">36 · 20 · 11</span> |
| 1 | <span class="ans">31 · 19 · 17</span> | <span class="ans">19 · 25 · 22</span> |
| 2 | <span class="ans">27 · 17 · 26</span> | <span class="ans">9 · 18 · 50</span> |
| 3 | <span class="ans">22 · 15 · 36</span> | <span class="ans">3 · 6 · 81</span> |
| 4 | <span class="ans">18 · 13 · 46</span> | <span class="ans">2 · 3 · 90</span> |
| 5 | <span class="ans">15 · 11 · 54</span> | <span class="ans">1 · 2 · 93</span> |

OPD 는 첫 걸음에 1 을 먼저 걷어내고(36% → 19%) 2 는 잠깐 늘었다가(20% → 25%), 두
번째 걸음부터 2 도 줄고 3 이 올라가요. SFT 는 처음부터 3 만 꾸준히 올려요. 두
손실은 기울기 크기가 달라서 걸음 수로 빠르기를 비교하면 안 되고, 방향만 보세요.
실제 학습은 파라미터를 Adam 으로 움직이기 때문에 로짓이 이렇게 따로따로 움직이지도
않아요.

<div class="rl-callout" markdown="1">
<span class="rl-callout__title">실제 LLM 의 OPD 는 172줄이 달라요</span>
어휘가 15만 개 안팎이라 칸마다 모든 토큰을 더하기는 비싸요. 그래서 학생이 실제로 쓴
토큰 하나의 log 확률 차이(log p<sub>학생</sub> − log p<sub>선생</sub>)를 그 토큰의
벌점으로 삼고, RLVR 처럼 policy gradient 로 학습하는 구현이 있어요(Thinking
Machines 가 2025년에 공개한 OPD 글이 이 방식이에요). 그러면 선생에게서는 쓴 토큰의
log 확률만 받으면 돼요. 어느 쪽이든 선생과 학생의 토크나이저는 같아야 해요.
</div>

### 10. 학습 루프 — `post_train()` {#code-loop-ko}

```python
def post_train(method: str, model: TinyGPT, plan: list[tuple[int, list[str]]], teachers: dict, tests: dict, seed: int) -> dict:
    """plan = [(스텝 수, 과목들), ...] — 단계가 둘이면 cascade (앞 과목을 다 한 뒤 다음 과목)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    curve, signals, step, started = [], [], 0, time.time()
    schedule = [ops for steps, ops in plan for _ in range(steps)]
    for step in range(len(schedule) + 1):
        if step % LOG_EVERY == 0:
            point = {"step": step, **{op: accuracy(model, tests[op]) for op in tests}}
            point["signal"] = float(np.mean(signals)) if signals else None
            curve.append(point)
            signals = []
        if step == len(schedule):
            break
        ops = schedule[step]
        if method == "rlvr":
            prompts, answers, _ = mixed(BATCH // GROUP, ops, rng)
            stats = step_rlvr(model, opt, prompts, answers)
        else:
            prompts, answers, subjects = mixed(BATCH, ops, rng)
            if method == "sft":
                stats = step_sft(model, opt, prompts, teacher_tokens(prompts, subjects, teachers))
            else:
                stats = step_opd(model, opt, prompts, subjects, teachers)
        if "signal" in stats:
            signals.append(stats["signal"])
    return {"curve": curve, "seconds": round(time.time() - started, 1)}
```

<p class="rl-caption">rlsim.py, 193~220줄</p>

- **199줄:** `plan` 을 스텝마다의 과목 목록으로 펼쳐요. `[(1000, ["+", "-"])]` 는
  1,000칸 모두 `["+", "-"]` 예요. Cascade 의 `[(500, ["+"]), (500, ["-"])]` 는 앞
  500칸이 `["+"]`, 뒤 500칸이 `["-"]` 예요. 이 한 줄이 Mix-RL 과 Cascade-RL 의 차이
  전부예요.
- **201–205줄:** 20스텝마다(0스텝과 마지막 스텝 포함) 과목별 평가 문제의 greedy
  정답률을 재고, 그동안 모은 signal 의 평균을 같이 적어요. 그래프의 점 하나가
  이거예요. 1,000스텝이면 51개예요.
- **209–211줄:** RLVR 은 `BATCH // GROUP` = 32문제를 뽑아요. 과목이 둘이면
  16문제씩이에요.
- **213–217줄:** 나머지는 256문제를 뽑아요. SFT 면 선생 답을 만들어 `step_sft` 에,
  아니면 `step_opd` 에 넘겨요. OPD 와 MOPD 는 같은 줄이에요.

### 11. 실험 정의 — `experiment()` {#code-exp-ko}

```python
    if name == "multi":
        runs = {
            "mix-rlvr": ("rlvr", [(steps, ["+", "-"])], {}),
            "cascade-rlvr": ("rlvr", [(steps // 2, ["+"]), (steps // 2, ["-"])], {}),
            "mopd": ("opd", [(steps, ["+", "-"])], {"+": add, "-": sub}),
            "opd-add-teacher-only": ("opd", [(steps, ["+", "-"])], {"+": add, "-": add}),
        }
    else:
        runs = {
            "single": {
                "sft": ("sft", [(steps, ["+"])], {"+": add}),
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd": ("opd", [(steps, ["+"])], {"+": add}),
            },
            "cold": {
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd": ("opd", [(steps, ["+"])], {"+": add}),
            },
            "weak": {
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd-weak": ("opd", [(steps, ["+"])], {"+": weak}),
                "sft-weak": ("sft", [(steps, ["+"])], {"+": weak}),
            },
        }[name]
```

<p class="rl-caption">rlsim.py, 303~326줄</p>

run 하나는 (학습법, 계획, 선생 사전) 세 개로 정해져요. 이 표가 실험 설계의
전부예요.

```python
    for seed in range(seeds):
        if name == "cold":
            torch.manual_seed(seed)
            start, info = student(), {"steps": 0, "+": accuracy(student(), tests["+"])}
        else:
            start, info = warm_student(ops, seed)
        result["warm"].append(info)
        if name == "single" and seed == 0:
            result["example"] = example_card(start, add)
        for run, (method, plan, run_teachers) in runs.items():
            out = post_train(method, copy.deepcopy(start), plan, run_teachers, tests, seed)
            result["runs"].setdefault(run, []).append(out)
```

<p class="rl-caption">rlsim.py, 331~342줄</p>

시드마다 출발 학생을 한 번 만들고(실험 2 는 무작위 모델), 학습법마다
`copy.deepcopy(start)` 로 복사해서 `post_train()` 에 넘겨요. 학습법끼리 가중치를
공유하지 않아요.

### 12. 선생과 출발 학생 — `sft_until()`, `load_or_train_teachers()`, `warm_student()` {#code-teachers-ko}

```python
def sft_until(model: TinyGPT, ops: list[str], target: float, rng, val: dict, check_every: int, max_steps: int) -> dict:
    """정답으로 SFT 를 하다가 정답률이 target 에 닿으면 멈춘다 (선생 만들기, 학생 출발점 만들기)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(max_steps + 1):
        if step % check_every == 0:
            accs = {op: accuracy(model, val[op]) for op in ops}
            if np.mean(list(accs.values())) >= target or step == max_steps:
                return {"steps": step, **accs}
        prompts, answers, _ = mixed(BATCH, ops, rng)
        step_sft(model, opt, prompts, answers)
    raise AssertionError
```

<p class="rl-caption">rlsim.py, 180~190줄</p>

```python
def load_or_train_teachers() -> dict[str, TinyGPT]:
    """덧셈 전문 선생, 뺄셈 전문 선생 (각자 자기 과목만 배움), 그리고 덜 배운 덧셈 선생 (약 70%)."""
    folder = OUT / "teachers"
    folder.mkdir(parents=True, exist_ok=True)
    specs = {"add": (["+"], 0.995, 1), "sub": (["-"], 0.995, 2), "add-weak": (["+"], 0.70, 3)}
    teachers, report = {}, {}
    for name, (ops, target, seed) in specs.items():
        torch.manual_seed(seed)
        model = teacher()
        path = folder / f"{name}.pt"
        if path.exists():
            model.load_state_dict(torch.load(path))
        else:
            started = time.time()
            info = sft_until(model, ops, target, np.random.default_rng(seed), val_sets(), 25 if target < 0.9 else 100, 30000)
            torch.save(model.state_dict(), path)
            print(f"teacher {name}: {info} in {time.time() - started:.0f}s", flush=True)
        teachers[name] = model
        report[name] = {op: accuracy(model, t) for op, t in test_sets().items()}
    (folder / "accuracy.json").write_text(json.dumps(report, indent=1))
    print("teacher test accuracy:", report, flush=True)
    return teachers


def warm_student(ops: list[str], seed: int) -> tuple[TinyGPT, dict]:
    torch.manual_seed(seed)
    model = student()
    info = sft_until(model, ops, WARM_TARGET, np.random.default_rng(1000 + seed), val_sets(), 5, 5000)
    return model, info
```

<p class="rl-caption">rlsim.py, 236~264줄</p>

- **`sft_until()`:** 진짜 정답으로 SFT 해요(학습률 0.001). `check_every` 스텝마다
  검증 문제 1,000개(시드 999)로 재서 목표에 닿으면 멈춰요. 멈출 때를 정하는 문제와
  그래프용 평가 문제(시드 12345)가 따로예요.
- **선생 셋:** 덧셈 선생은 덧셈만 99.5% 까지(시드 1), 뺄셈 선생은 뺄셈만 99.5%
  까지(시드 2) 배워요. 약한 선생은 덧셈만 70% 까지(시드 3)인데, 25스텝마다 재서
  70% 를 처음 넘은 275스텝에서 멈췄어요. 평가 문제에선 73% 가 나왔어요.
- **저장:** 선생은 `results/teachers/*.pt` 로 저장해 두고, 다음 실행부터는 다시
  학습하지 않고 불러와요.
- **`warm_student()`:** 목표 30%, 5스텝마다 확인해요. 가중치 초기화는 실험 시드로,
  문제는 1000 + 시드로 뽑아요. 실험 1 과 3 은 같은 함수·같은 시드라 출발 학생이
  똑같아요.

### 13. 실제 LLM 학습 코드에서는 이 줄들이 이렇게 바뀌어요 {#code-real-ko}

| 이 코드 | 실제 LLM 학습 (TRL · verl 같은 프레임워크) | 왜 |
| --- | --- | --- |
| `write()` 의 반복문 | vLLM · SGLang 같은 추론 엔진이 KV 캐시로 생성하고, 스텝마다 학습 중인 가중치를 엔진으로 옮겨요 | 답이 수천 토큰이라 생성이 전체 시간의 대부분이에요 |
| `answer_logits()` 한 번 | 현재 모델, 답안을 만들 당시 모델(old), 기준 모델(ref) 세 개의 log 확률을 계산해요 | 아래의 확률 비율과 KL 에 필요해요 |
| `correct()` | 수식 동치 비교(Math-Verify), 코드 테스트 샌드박스, 루브릭 + LLM 채점 | 정답의 형태가 다양해요 |
| 155줄 advantage | 같은 GRPO 식을 써요. Dr. GRPO 는 표준편차로 나누기를 빼고, DAPO 는 signal 이 0 인 그룹을 버리고 다시 뽑아요(dynamic sampling) | 이 실험에서 본 signal 감소를 직접 겨냥한 수정이에요 |
| 157줄 `-(A × logp).mean()` | 한 묶음의 답안으로 여러 번 업데이트하므로 확률 비율 π / π<sub>old</sub> 를 곱하고 0.8~1.2 로 잘라요(PPO clip). KL(π ‖ π<sub>ref</sub>) 벌점을 더하기도 해요 | 여기선 묶음마다 한 번만 업데이트해서 비율이 늘 1 이라 필요 없었어요 |
| 256 × 5 평균 | 답 길이가 제각각이라 토큰 단위로 평균할지 답안 단위로 평균할지가 결과를 바꿔요(DAPO 의 token-level loss) | 여기선 모든 답이 5글자라 차이가 없어요 |
| 172줄의 14글자 합 | 쓴 토큰 하나의 log 확률 차이만 쓰거나, 상위 몇 개 토큰만 써요 | 어휘가 15만 개 안팎이에요 |
| 학습률 5e-4 | 1e-6 안팎 | 큰 모델은 조금씩 움직여야 안정적이에요 |

### 14. 알고 있는 허술한 점 {#code-gaps-ko}

- **평가 문제가 학습 중에 나왔을 수 있어요.** 덧셈 문제는 1,000 × 1,000 = 100만
  가지예요. SFT · OPD 는 1,000스텝 동안 256,000문제를 무작위로 뽑으니까, 평가 문제
  하나가 그사이 한 번이라도 나왔을 확률은 22.6% 예요(RLVR 은 32,000문제라 3.1%).
  10만 파라미터 모델이 100만 가지를 외우기는 어려워서 정답률은 규칙을 배웠는지를
  잰다고 봐도 되지만, 엄밀히 "처음 보는 문제" 는 아니에요. 막으려면 `problems()`
  에서 평가 문제를 걸러야 해요.
- **실험 2 의 출발 정답률 기록:** 334줄이 `student()` 를 두 번 불러서, 기록한 0%
  는 실제 출발 모델이 아니라 새로 만든 다른 무작위 모델의 값이에요. 둘 다 무작위
  모델이라 0% 인 건 같아서 결과는 그대로예요.
- **선생이 없는 과목:** `step_opd()` 는 `teachers` 에 없는 과목의 행을 초기화하지
  않은 채(`torch.empty_like`) 계산해요. 이번 실험에선 모든 과목에 선생이 있어서
  생기지 않았어요. 고치려면 빠진 과목이 있을 때 멈추게 해야 해요.
- **안전장치가 없어요:** KL(π ‖ π<sub>ref</sub>) 제약, 앞 과목 데이터 섞기, 학습률
  조절이 없어요. Cascade 붕괴가 과장됐을 수 있는 이유예요.

## 용어 풀이 {#terms-ko}

| 용어 | 쉬운 말 | 이 실험에서 |
| --- | --- | --- |
| 정책 (policy) | 학습하는 모델 자체. "이 상황에서 어떤 글자를 쓸지" 를 정하니까 정책이라 불러요 | 학생 모델 |
| 롤아웃 (rollout) | 모델이 직접 써 본 답안 | 스텝마다 256장 |
| 보상 (reward) | 답안 하나의 점수 | 맞으면 1, 틀리면 0 |
| 검증기 (verifier) | 답을 기계로 채점하는 프로그램 | `correct()` |
| RLVR | 검증기 점수로 하는 강화학습 (RL with Verifiable Rewards) | O/X 채점 |
| GRPO | 같은 문제를 여러 번 풀게 해서 답안끼리 비교하는 RLVR 방식 | 문제당 8번 |
| advantage | 그 답안이 같은 문제 답안들의 평균보다 얼마나 나은가 | (점수 − 평균) ÷ 표준편차 |
| on-policy | 학생이 직접 쓴 답안으로 배우기 | RLVR, OPD, MOPD |
| off-policy | 남(선생)이 쓴 답안으로 배우기 | SFT 증류 |
| 증류 (distillation) | 큰 모델(선생)의 실력을 작은 모델(학생)에게 옮기기 | SFT 증류, OPD, MOPD |
| KL 발산 | 두 확률 분포가 얼마나 다른가. 같으면 0 | OPD 벌점 |
| reverse KL | 학생이 확률을 건 곳을 기준으로 잰 KL. 학생이 선생이 안 쓰는 글자에 걸면 크게 벌점 | OPD 벌점 |
| temperature | 확률대로 뽑을 때 얼마나 모험할지. 1 이면 확률 그대로 | 학습 중 1 |
| greedy | 매번 가장 확률 높은 글자만 고르기 | 평가할 때 |
| 스텝 | 파라미터를 한 번 업데이트하는 단위 | 답안 256장마다 한 번 |
| 시드 (seed) | 난수의 출발점. 바꾸면 같은 실험도 결과가 조금 달라져요 | 3개로 반복 |
| 로짓 (logit) | softmax 를 거치기 전의 글자별 점수. softmax 를 거치면 확률이 돼요 | `model(seq)` 의 출력 |
| teacher forcing | 다음 글자를 맞힐 때 앞부분으로 자기 글자 대신 정해진 답(선생 답)을 넣어 주기 | SFT 증류 |
| 행동 복제 (behavior cloning) | 보상 없이 시연을 그대로 따라 하는 학습. SFT 를 RL 쪽 말로 부른 이름 | SFT 증류 |
| policy gradient | 점수가 높았던 답안이 더 잘 나오도록 그 답안의 log 확률을 키우는 기울기 | `step_rlvr` 의 손실 |
| baseline | 점수에서 빼 주는 기준값. 기울기의 평균은 그대로 두고 잡음만 줄여요 | 같은 문제 8장의 평균 점수 |
| KV 캐시 | 글자를 쓸 때 앞부분 계산을 저장해 두고 다시 쓰는 것 | 안 씀 (13칸이라) |
| 과정 보상 (process reward) | 답 전체가 아니라 중간 단계마다 맞았는지 주는 점수 | 실험 5 의 칸마다 채점 |
| off-policy guidance | RL 답안 묶음에 남(선생·정답)이 쓴 답을 섞어 넣기 | 실험 5 의 정답 끼워 넣기 |
| 커리큘럼 (curriculum) | 쉬운 문제부터 차례로 어렵게 가르치기 | 실험 5 의 쉬운 문제부터 |
| cold start | 아무것도 모르는 상태에서 시작 | 실험 2 |
| 파괴적 망각 | 새 과목을 배우다 옛 과목을 잊는 것 | Cascade-RL |
| 시소 효과 (see-saw) | 여러 과목을 섞어 배우면 하나가 오를 때 다른 하나가 내려가는 것 | Mix-RL |

## 다시 돌려 보기 {#rerun-ko}

파이썬 3, PyTorch, NumPy 만 있으면 돼요. GPU 없이 CPU 로 약 35분 걸려요.
[`rlsim.py`]({{ '/files/rl-toy/rlsim.py' | relative_url }}) 를 내려받으세요.

```bash
# 선생 셋 학습 → 실험 1~4 (1,000스텝, 시드 3개) → 실험 3 만 3,000스텝으로 다시
python3 rlsim.py all --steps 1000
python3 rlsim.py weak --steps 3000

# 빠른 확인 (시드 1개, 100스텝, 1분 안쪽)
python3 rlsim.py single --seeds 1 --steps 100
```

`rlsim.py` 로 실험 1~4 가 재현되고, 결과는 `results/*.json` 에 쌓여요. 실험 5,
자리별 정답률, 차트는 `rlsim.py` 위에 얹은 작은 보조 스크립트(`exp5.py`,
`cold_digits.py`, `digit_order.py`, `trace.py`, `summary.py`, `charts.py`)로
만들었고 여기에는 싣지 않았어요. 실험 5 의 핵심은 위에 옮긴 `step_rlvr_variant`
함수예요. 실험 6 은 따로 만든 더 큰 스크립트(`big/bigsim.py`, `big/orchestrate.py`
로 실행, H100 2장에 약 3시간 40분)라 역시 싣지 않았어요. 명령줄의 기본값은
`--steps 500` 이라, 이 글의 실험처럼 1,000스텝으로 돌리려면 값을 넘겨야 해요.

---

*세 자리 덧셈·뺄셈 장난감 실험이에요. 실제 LLM 학습의 방향을 눈으로 보려고
만들었고, 숫자는 이 설정에서만 유효해요.*

</div>
