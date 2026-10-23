---
title: "Questions decide the score: two FDE-Bench cases, drawn out"
title_ko: "점수는 질문이 정한다: FDE-Bench 케이스 두 개를 그림으로 뜯어보기"
permalink: /posts/fde-bench-two-cases-drawn-out/
date: 2026-10-22
last_modified_at: 2026-10-22
eyebrow: "FIELD NOTE / AGENT EVALUATION"
dek: "I downloaded FDE-Bench, opened a routing case and a credit-scoring case, and ran my own submissions through their real evaluators. Most zero scores come from a rule nobody wrote down, and the routing case's two geometry penalties are easiest to understand by dragging a point around."
dek_ko: "FDE-Bench 를 내려받아 경로 계획 케이스와 신용평가 케이스를 열고, 직접 만든 제출물을 실제 채점기로 채점해 봤어요. 0점은 대부분 아무도 적어 두지 않은 규칙에서 나오고, 경로 케이스의 두 가지 기하 벌점은 점 하나를 끌어 보면 가장 빨리 이해돼요."
read_time: true
comments: false
share: false
related: false
---

<link rel="stylesheet" href="{{ '/assets/css/posts/fde-bench.css' | relative_url }}?v={{ site.jarvis_asset_version }}">
<script src="{{ '/assets/js/posts/fde-bench.js' | relative_url }}?v={{ site.jarvis_asset_version }}" defer></script>

<div class="lang-block" lang="en" markdown="1">

[FDE-Bench](https://arxiv.org/abs/2610.14539) (Baidu AI Cloud, 2026) tests whether
an agent can do a forward-deployed engineer's job: take a vague customer request,
ask the right questions, and deliver something the customer would accept. Its 49
cases come from real customer projects. The best of 18 agent setups reached the
FDE's own solution in only 6.12% of runs.

Numbers like that are easier to believe once you've seen a case up close. So I
downloaded the benchmark, opened two cases, built a few submissions of the kind an
agent would plausibly write, and scored them with each case's own evaluator. None
of my submissions are real agent runs; they're stand-ins for common mistakes.

## How a case is packaged {#package}

| File | Agent sees it? | What it holds |
| --- | --- | --- |
| `instruction.md` | yes | one paragraph from the customer |
| `data/` | yes | business data |
| `information.md` | no | full business context; the simulated customer answers from it |
| `gt.json` | no | the requirements the agent is supposed to draw out |
| `tests/evaluator.py` | no | the grader |
| `tests/best_solution/` | no | the FDE's accepted solution |

The agent may ask the simulated customer up to 30 rounds of questions, then has
to deliver. The evaluator gives no feedback along the way.

## Case 002: city delivery routing {#case-002}

The request, translated from Chinese:

> We run a delivery warehouse that serves all downstream retail stores. We have
> 12 trucks with drivers, 7,125 kg capacity each. The trunk leg out of and back to
> the warehouse runs at about 70 km/h, roads between stores at 35 km/h, and each
> stop takes 120 seconds. Assign every store and plan a route per truck. Keep
> drivers' workloads as even as possible; keep each truck's stores geographically
> together, with different trucks' areas not overlapping; given all that, the
> shorter the total delivery time the better.

The data is one CSV: 1,335 rows of store code, weight, longitude, latitude and an
urban/rural flag. The warehouse sits just outside the north-west corner of the
service area, so every truck starts with a long trunk leg. The total load is
60,903 kg, which is 8.55 trucks' worth.

### Seven rules the request leaves out

| # | Hidden rule (`gt.json`) | What happens if you don't ask |
| --- | --- | --- |
| 1 | The row with no weight is the warehouse (code `0`); never write it in a route | writing `0` at both ends of each route scores **0** |
| 2 | Exactly 12 routes; empty routes are allowed as filler | using 9 trucks scores **0** |
| 3 | "Even workload" means longest route minus shortest route, in time | balancing by weight optimizes the wrong thing |
| 4 | Total time is the sum of the 12 routes; 70 km/h only on the first and last leg | the objective is computed differently from the grader |
| 5 | Area overlap is counted with convex hulls | you can't measure "together" |
| 6 | Crossing route segments of different trucks are counted | same |
| 7 | Priority: overlap and crossings > balance > total time | you don't know what to trade away |

### The score {#score}

```text
cost  = total time (s) × 1
      + (longest − shortest route) (s) × 90
      + hull intrusions × 600
      + segment crossings × 1,800
score = 281,497 (the FDE's cost) ÷ your cost
```

The weights live only in the evaluator. Asking the customer gets you rule 7's
order, never the numbers.

## What counts as a hull intrusion {#hull}

For each truck, take the smallest convex polygon around its stores (the depot is
not included). Every store of **another** truck that lies inside that polygon is one
intrusion. Three details matter, and the demo below shows all of them:

- A store **on the edge** counts. The test is `cross < 0 → outside`, so zero is inside.
- A truck with fewer than 3 stores has no polygon, but **its stores can still
  intrude** into other trucks' polygons.
- It works both ways. One stray store stretches its own truck's polygon over the
  neighbour's stores, so a single misplaced stop can cost several intrusions.

<figure class="fde-fig" data-fde="hull"></figure>
<p class="fde-caption">Figure 1. Drag vehicle B's highlighted store. The count uses the evaluator's own hull and inside tests, ported to the page.</p>

## What counts as a crossing {#cross}

Each route is a closed path: depot → stores in order → depot. Every pair of
segments from **two different trucks** that properly cross is one crossing,
including the depot legs.

- Segments that only **touch at an endpoint** don't count. All routes share the
  depot, and that never counts.
- A truck crossing **its own** path doesn't count.
- A depot leg cutting through another truck's area does count, which is easy to
  miss.

<figure class="fde-fig" data-fde="cross"></figure>
<p class="fde-caption">Figure 2. Drag vehicle B's highlighted stop. Red × marks counted crossings; the dashed grey ring is A's self-crossing, which is free.</p>

## Four submissions on the real map {#map}

| Submission | Idea | Score |
| --- | --- | ---: |
| FDE reference | the accepted solution | 1.000 |
| E · weight-balanced sweep | cut the map into 12 wedges around the depot with equal weight, then 2-opt | 0.137 |
| G · nearest-neighbour fill | each truck takes the nearest unvisited store until it holds 1/12 of the load | 0.079 |
| B · 9 trucks + 3 empty | knows rule 2, not rule 3 | 0.063 |

<figure class="fde-fig" data-fde="map" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">Figure 3. All 1,334 stores. Switch submissions and layers; hover a store or a × for details. Red rings are intruding stores, red × are crossings.</p>

- **The reference and E** both have zero intrusions and zero crossings. Their
  territories are clean.
- **G** is what "always go to the nearest store" produces: early trucks grab the
  area around the depot, and later trucks have to reach through it. That gives
  1,259 intrusions and 191 crossings.
- **B** puts its stores into 9 routes, so the 9 polygons are larger and overlap
  more: 718 intrusions.

### Zooming in on the worst pairs {#pairs}

<figure class="fde-fig" data-fde="pairs" data-variant="greedy" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">Figure 4. Submission G: the two trucks with the most intrusions and crossings between them, others greyed out. Step through the top five pairs.</p>

The typical pattern is a long, thin route whose polygon covers most of a
neighbour's area. Almost every store inside it counts, and the depot legs cross
the neighbour's internal segments.

## The penalty that actually decides it: balance {#balance}

Clean territories alone don't get you far. E has none of G's geometry problems
and still scores only 0.137.

<figure class="fde-fig" data-fde="hours" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">Figure 5. Hours per truck. The FDE reference puts all 12 trucks at 6.52 h; E ranges from 3.90 to 9.41 h.</p>

E balanced **weight** (4.7–5.4 t per truck), so trucks serving the far south-east
spend hours more on the road. The reference does the opposite: loads range from
2.7 t to 7.1 t so that every route takes the same time. Distant areas get lighter
loads.

<figure class="fde-fig" data-fde="cost" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">Figure 6. Each submission's cost split into the evaluator's four terms. Hover a segment for its value.</p>

- In E, 87% of the cost is the balance term: a 5.5-hour spread × 90.
- E's total time is actually 6% **shorter** than the reference's. The reference
  gives up total time to equalize the routes, which is exactly the trade rule 7
  asks for.
- G pays for everything at once. B pays mostly for balance, because its empty
  routes count as 0 hours and become the "shortest route".

### Which rules each submission broke

<table class="fde-trap">
<thead><tr><th>Submission</th><th>1 depot</th><th>2 twelve</th><th>3 balance metric</th><th>5–6 geometry</th><th>Result</th></tr></thead>
<tbody>
<tr><td>A · 9 trucks</td><td class="fde-pass">ok</td><td class="fde-fail">✗</td><td>–</td><td>–</td><td class="fde-fail">0 (rejected)</td></tr>
<tr><td>C · depot written</td><td class="fde-fail">✗</td><td class="fde-pass">ok</td><td>–</td><td>–</td><td class="fde-fail">0 (rejected)</td></tr>
<tr><td>B · 9 + 3 empty</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-fail">✗</td><td class="fde-fail">718 · 30</td><td>0.063</td></tr>
<tr><td>G · nearest fill</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-fail">✗</td><td class="fde-fail">1,259 · 191</td><td>0.079</td></tr>
<tr><td>E · weight sweep</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-fail">✗ (weight)</td><td class="fde-pass">0 · 0</td><td>0.137</td></tr>
<tr><td>FDE reference</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-pass">0 · 0</td><td>1.000</td></tr>
</tbody>
</table>

Each rule a submission misses costs a step on the score ladder: a missed hard
rule means 0, the wrong balance metric caps the score near 0.14, and messy
geometry pushes it lower. Even E is a decent piece of engineering, but it answers
a slightly different question from the one the grader asks.

## Ask why, not only what {#why}

All seven hidden rules say *what* to count. None of them says *why*. Two "why"
questions would take an FDE further than any one of the seven.

**Why are crossings penalized at all?** The request only says areas shouldn't
overlap. The evaluator prices one crossing at 1,800 s, the same as 30 minutes of
extra driving, and one intrusion at 10 minutes. The case files never give the
reason. Common reasons in delivery are drivers who know their own area, two trucks
not working the same streets, and simpler dispatch. That's my guess, though, not
the customer's answer. The answer matters because it tells you how much driving you
may trade for clean territories, and whether one crossing is a disaster or a
nuisance.

**Why does the accepted plan look like a fan?** All 12 routes leave from and
return to one depot in the corner of the map. If no truck may reach into another
truck's polygon and no two routes may cross, the clean way to split the map is
into wedges around the depot. So the fan comes from the geometry rules, not from
taste. The balance rule then sets each wedge's width: wedges that reach far get
fewer stores (2.7 t versus 7.1 t). Ask this first and you know the shape of the
answer before you optimize anything. E already had the shape and only got the
widths wrong.

### Change the price list, change the winner {#weights}

The cost formula is a price list. One second of driving costs 1, and one second
between the busiest and the idlest truck costs 90. An intrusion costs 600 and a
crossing 1,800. Another company would write a different list, and a different plan
would win. Pick a company or move the sliders to see the ranking change.

<figure class="fde-fig" data-fde="weights" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">Figure 7. The same four plans priced with different weights. Each bar is a plan's cost split into the four terms. The score is the FDE plan's cost divided by that plan's cost under the same weights.</p>

- **Driving time only:** B wins because it drives the least. The FDE plan drives
  8.7% more and comes last.
- **Territories only:** E wins with 1.059. Its fan is clean, and it drives 6% less
  than the FDE plan.
- **This customer:** the FDE plan wins. A balance weight of 1 would already be
  enough, since it passes E above 0.8. At 90, balance decides everything.

These are four fixed plans. Under another price list the best plan itself would
look different. A time-only company has no reason to draw a fan, and a
territory-only company has no reason to equalize hours. That's why the price list
is the first thing to learn, and "why" is the question that reveals it.

<div class="fde-callout" markdown="1">
<span class="fde-callout__title">Same lesson, another case</span>
Case 044 predicts a siting score for bank branches and grades by mean absolute
error. Under that metric, a model trained on absolute error wins (9.34 versus
9.68). Graded by root-mean-square error, the squared-error model would win
instead (14.09 versus 14.15). Even the best single guess flips: the median wins on
absolute error (11.01 versus 11.53), the mean on squared error (16.44 versus 17.01).
</div>

## Case 031: a credit default scorecard {#case-031}

A consumer lender wants a default-probability model for new applicants, "with
defaulters ranked high and good payers low, as separable as possible". There are
280 training rows and 120 test rows (33 defaults).

Only two rules are hidden here: only ranking counts (AUC and KS weighted 50/50,
no calibration), and **tied scores are treated as one band**. A third rule is in
the business context but not among the questions: the output must be a
probability in [0, 1].

| Submission | AUC | KS | Score |
| --- | ---: | ---: | ---: |
| A · logistic regression | 0.876 | 0.591 | 0.848 |
| B · gradient boosting, defaults | 0.830 | 0.551 | 0.798 |
| C · 300–850 scorecard points | – | – | **0** (out of range) |
| D · raw logit | – | – | **0** (out of range) |
| E · A rounded to 0.1 (11 bands) | 0.858 | 0.522 | 0.797 |
| FDE reference | 0.927 | 0.805 | 1.000 |

The classic banking deliverable, a 300–850 points scorecard, is rejected
outright. The case is literally called "scorecard", which nudges you toward it.

Banding (E) hurts in a way you can see:

<figure class="fde-fig" data-fde="ks" data-src="{{ '/assets/data/posts/fde-bench-031.json' | relative_url }}"></figure>
<p class="fde-caption">Figure 8. Customers sorted riskiest first. Each curve shows how many defaulters are caught as non-defaulters are let through. KS is the widest vertical gap above the diagonal. Dots mark where E's 11 bands end.</p>

With ties treated as one band, the evaluator can only cut between bands. E's
curve becomes 11 straight chords, and the best cut it can find is lower (0.52
versus 0.59). Bands are normal in credit practice; here they cost points.

This case has the opposite shape from 002. It has fewer hidden rules, and a plain
model gets 0.85. The last 15% comes from modelling skill and, with 33 defaulters,
from luck. The paper reports that no ML run reached the reference level.

## How close is this to real work? {#limits}

It's still a simplified world.

- **The data has no noise.** Weights, coordinates and labels are taken as correct.
  Nothing is mislabeled, and the messy parts are spelled out in the requirements.
- **The customer is a simulation.** It answers from one fixed document. A real
  customer forgets, contradicts themselves and changes their mind.
- **The price list is fixed and hidden.** In a real project the weights come out of
  meetings, and they move.

Even so, it's a step closer to real agentic ability than tests like MMLU. MMLU asks
multiple-choice questions with one right answer each. Here the agent gets an
underspecified request, business data and a customer to question. It has to decide
what to ask, build something, run it, check it, and deliver a file that a strict
grader accepts, all within 12 hours. That's much closer to the job than picking A,
B, C or D.

## Takeaways {#takeaways}

- **Most zeros are rules, not maths.** All four of my submissions that were
  rejected outright (A and C in case 002, C and D in case 031) broke a format or
  business rule; none failed on the optimization or the model. The paper finds the
  same: 78.5% of zero-score runs violated a constraint.
- **The right first questions are visible in the data and the wording.** A row
  with no weight, "12 trucks are available", "even" with no unit, "scorecard" with
  "probability" in the same sentence. Each one is a question an FDE would ask in
  the first meeting.
- **"Why" shows the shape of the answer.** Asking why crossings are penalized, and
  why the accepted plan is a fan, tells you to cut wedges around the depot before
  you write any code.
- **Even with every rule, the weights stay hidden.** Rule 7 gives an order, not
  ×90. The weights are the customer's own design, and a different design crowns a
  different plan ([figure 7](#weights)). Getting to 1.0 means reading what this
  customer actually expects: here, routes balanced to the second.

---

*Case data: FDE-Bench (Li, Huang et al., 2026), CC BY-NC-ND 4.0. Figures are
drawn from case 002's store coordinates and case 031's test labels for
commentary; scores come from each case's own `tests/evaluator.py`. Submissions A–G
are my own simple heuristics, not agent runs.*

</div>

<div class="lang-block" lang="ko" markdown="1">

[FDE-Bench](https://arxiv.org/abs/2610.14539)(Baidu AI Cloud, 2026)는 에이전트가
FDE(forward-deployed engineer)의 일을 할 수 있는지를 봐요. 모호한 고객 요청을 받아
맞는 질문을 하고, 고객이 수락할 결과물을 내는 일이에요. 49개 케이스는 실제 고객
프로젝트에서 왔고, 18개 에이전트 구성 중 가장 잘한 것도 FDE 정답 수준에 닿은 실행이
6.12% 뿐이었어요.

이런 숫자는 케이스 하나를 가까이서 봐야 실감이 나요. 그래서 벤치마크를 받아 케이스
두 개를 열고, 에이전트가 낼 법한 제출물을 몇 개 만들어 각 케이스의 채점기로
채점했어요. 제 제출물은 실제 에이전트 실행이 아니라, 흔한 실수를 흉내 낸 거예요.

## 케이스는 이렇게 생겼어요 {#package-ko}

| 파일 | 에이전트가 보나 | 내용 |
| --- | --- | --- |
| `instruction.md` | 봐요 | 고객이 보낸 요청 한 문단 |
| `data/` | 봐요 | 업무 데이터 |
| `information.md` | 안 봐요 | 전체 업무 정보. 시뮬레이션 고객이 이걸 보고 답해요 |
| `gt.json` | 안 봐요 | 에이전트가 캐내야 할 요구사항 |
| `tests/evaluator.py` | 안 봐요 | 채점기 |
| `tests/best_solution/` | 안 봐요 | 고객이 수락한 FDE 정답 |

에이전트는 시뮬레이션 고객에게 최대 30번 질문하고 결과물을 내요. 그동안 채점기는
아무 피드백도 주지 않아요.

## 케이스 002: 도시 배송 경로 {#case-002-ko}

요청서(중국어 원문 번역):

> 저희는 배송 창고를 운영하면서 하위 소매점 전체에 물건을 보냅니다. 트럭과 기사
> 12팀이 있고 트럭당 7,125kg 실을 수 있습니다. 창고에서 나가고 돌아오는 간선 구간은
> 약 70km/h, 매장 사이 길은 35km/h, 매장마다 120초 걸립니다. 모든 매장을 배정하고
> 트럭별 경로를 짜 주세요. 기사 업무량은 최대한 균등하게, 트럭마다 맡은 매장은 한
> 덩어리로 모이게, 다른 트럭과 구역이 겹치지 않게 해 주세요. 그다음엔 총 배송 시간이
> 짧을수록 좋습니다.

데이터는 CSV 하나예요. 1,335행이고 매장 코드, 무게, 경도, 위도, 도시/농촌 열이
있어요. 창고는 배송 구역의 서북쪽 귀퉁이 바로 바깥에 있어서, 모든 트럭이 긴 간선
구간으로 출발해요. 총 화물은 60,903kg 으로 트럭 8.55대 분량이에요.

### 요청서에 없는 규칙 7개

| # | 숨은 규칙 (`gt.json`) | 묻지 않으면 |
| --- | --- | --- |
| 1 | 무게가 빈 행은 창고(코드 `0`)예요. 경로에 적으면 안 돼요 | 경로 양끝에 `0` 을 적으면 **0점** |
| 2 | 경로는 정확히 12개. 빈 경로로 채워도 돼요 | 9대만 쓰면 **0점** |
| 3 | "균등"은 가장 긴 경로와 가장 짧은 경로의 **시간** 차이예요 | 무게로 맞추면 엉뚱한 걸 최적화해요 |
| 4 | 총 시간은 12개 경로의 합. 70km/h는 첫 구간과 마지막 구간에만 | 목적 함수를 채점기와 다르게 계산해요 |
| 5 | 구역 겹침은 볼록 다각형으로 세요 | "한 덩어리"를 잴 방법이 없어요 |
| 6 | 다른 트럭끼리 경로 선분이 교차하면 세요 | 위와 같아요 |
| 7 | 우선순위: 겹침·교차 > 균등 > 총 시간 | 무엇을 희생할지 몰라요 |

### 점수 {#score-ko}

```text
비용 = 총 시간(초) × 1
     + (최장 − 최단 경로)(초) × 90
     + 다각형 침범 횟수 × 600
     + 선분 교차 횟수 × 1,800
점수 = 281,497 (FDE 정답의 비용) ÷ 내 비용
```

가중치는 채점기 안에만 있어요. 고객에게 물어도 7번의 순서까지만 알 수 있고, 숫자는
안 나와요.

## 다각형 침범은 어떻게 세나 {#hull-ko}

트럭마다 그 트럭 매장들을 감싸는 가장 작은 볼록 다각형을 그려요(창고는 빼요). 그
안에 **다른** 트럭의 매장이 하나 있으면 침범 1회예요. 중요한 세부가 세 가지 있고,
아래 그림에서 다 확인할 수 있어요.

- **경계 위**도 세요. 판정이 `cross < 0 → 바깥` 이라 0이면 안쪽이에요.
- 매장이 3곳 미만인 트럭은 자기 다각형이 없어요. 하지만 그 트럭의 매장도 **남의
  다각형 안에 들어가면** 세요.
- 양방향이에요. 동떨어진 매장 하나가 자기 트럭의 다각형을 옆 트럭 매장 위로 늘려서,
  잘못 넣은 정차 하나가 침범 여러 회가 될 수 있어요.

<figure class="fde-fig" data-fde="hull"></figure>
<p class="fde-caption">그림 1. 차량 B의 강조된 매장을 끌어 보세요. 채점기의 다각형·안쪽 판정을 그대로 옮겨서 세요.</p>

## 교차는 어떻게 세나 {#cross-ko}

경로는 닫힌 길이에요: 창고 → 매장들 순서대로 → 창고. **서로 다른 두 트럭**의 선분
쌍이 제대로 교차하면 1회예요. 창고 구간도 포함해요.

- **끝점에서 맞닿기만** 하면 안 세요. 모든 경로가 창고를 공유하지만 그건 세지
  않아요.
- **자기 경로끼리** 교차하는 건 안 세요.
- 창고 구간이 다른 트럭 구역을 가로지르면 세요. 놓치기 쉬운 부분이에요.

<figure class="fde-fig" data-fde="cross"></figure>
<p class="fde-caption">그림 2. 차량 B의 강조된 정차 지점을 끌어 보세요. 빨간 ×가 세는 교차이고, 회색 점선 원은 A의 자기 교차라 벌점이 없어요.</p>

## 실제 지도 위의 제출물 네 개 {#map-ko}

| 제출물 | 방법 | 점수 |
| --- | --- | ---: |
| FDE 정답 | 고객이 수락한 솔루션 | 1.000 |
| E · 무게로 맞춘 부채꼴 | 창고 주위를 무게가 같은 부채꼴 12개로 자르고 2-opt | 0.137 |
| G · 가까운 곳부터 채우기 | 트럭마다 가장 가까운 매장을 화물 1/12 이 될 때까지 담기 | 0.079 |
| B · 9대 + 빈 경로 3개 | 2번 규칙은 알고 3번은 모름 | 0.063 |

<figure class="fde-fig" data-fde="map" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">그림 3. 매장 1,334곳 전체. 제출물과 레이어를 바꿔 보고, 매장이나 × 위에 마우스를 올려 보세요. 빨간 원은 침범한 매장, 빨간 ×는 교차예요.</p>

- **정답과 E** 는 둘 다 침범 0회, 교차 0회예요. 구역이 깔끔해요.
- **G** 는 "항상 가장 가까운 매장으로"가 만드는 모양이에요. 먼저 나간 트럭이 창고
  주변을 차지하고, 나중 트럭은 그 사이를 뚫고 나가야 해요. 그래서 침범 1,259회,
  교차 191회예요.
- **B** 는 매장을 경로 9개에 나눠서 다각형이 크고 서로 많이 겹쳐요. 침범 718회예요.

### 가장 심한 쌍을 확대해 보면 {#pairs-ko}

<figure class="fde-fig" data-fde="pairs" data-variant="greedy" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">그림 4. 제출물 G에서 서로 침범·교차가 가장 많은 두 트럭만 보여 줘요. 상위 다섯 쌍을 넘겨 볼 수 있어요.</p>

전형적인 모양은 길고 가는 경로예요. 그 다각형이 옆 트럭 구역을 거의 덮어서 안에
든 매장이 대부분 침범으로 세지고, 창고 구간은 옆 트럭의 중간 선분들을 가로질러요.

## 실제로 승부를 가르는 벌점: 균등 {#balance-ko}

구역만 깔끔하다고 되는 게 아니에요. E는 G 같은 기하 문제가 하나도 없는데도
0.137이에요.

<figure class="fde-fig" data-fde="hours" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">그림 5. 트럭별 경로 시간. FDE 정답은 12대 모두 6.52시간이고, E는 3.90~9.41시간이에요.</p>

E는 **무게**를 맞췄어요(트럭당 4.7~5.4t). 그래서 남동쪽 끝을 맡은 트럭은 길에서
몇 시간을 더 보내요. 정답은 반대로 해요. 모든 경로의 시간이 같아지도록 적재량을
2.7t 에서 7.1t 까지 다르게 줘서, 먼 구역 트럭은 적게 실어요.

<figure class="fde-fig" data-fde="cost" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">그림 6. 제출물마다 비용을 채점기의 네 항으로 나눴어요. 막대 위에 마우스를 올리면 값이 보여요.</p>

- E 비용의 87% 가 균등 항이에요. 5.5시간 차이 × 90이에요.
- E의 총 시간은 정답보다 오히려 6% **짧아요**. 정답은 총 시간을 내주고 경로 시간을
  똑같이 맞췄어요. 7번 규칙이 요구하는 맞교환이 바로 이거예요.
- G는 모든 항에서 다 벌점을 받아요. B는 대부분 균등 항이에요. 빈 경로가 0시간이라
  "가장 짧은 경로"가 되기 때문이에요.

### 제출물마다 어긴 규칙

<table class="fde-trap">
<thead><tr><th>제출물</th><th>1 창고</th><th>2 12개</th><th>3 균등 기준</th><th>5–6 기하</th><th>결과</th></tr></thead>
<tbody>
<tr><td>A · 9대</td><td class="fde-pass">ok</td><td class="fde-fail">✗</td><td>–</td><td>–</td><td class="fde-fail">0 (무효)</td></tr>
<tr><td>C · 창고를 적음</td><td class="fde-fail">✗</td><td class="fde-pass">ok</td><td>–</td><td>–</td><td class="fde-fail">0 (무효)</td></tr>
<tr><td>B · 9대 + 빈 경로 3개</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-fail">✗</td><td class="fde-fail">718 · 30</td><td>0.063</td></tr>
<tr><td>G · 가까운 곳부터</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-fail">✗</td><td class="fde-fail">1,259 · 191</td><td>0.079</td></tr>
<tr><td>E · 무게 부채꼴</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-fail">✗ (무게)</td><td class="fde-pass">0 · 0</td><td>0.137</td></tr>
<tr><td>FDE 정답</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-pass">ok</td><td class="fde-pass">0 · 0</td><td>1.000</td></tr>
</tbody>
</table>

놓친 규칙 하나가 점수 사다리의 한 칸이에요. 하드 규칙을 놓치면 0점, 균등 기준을
틀리면 0.14 근처가 천장이고, 기하가 지저분하면 거기서 더 떨어져요. E도 엔지니어링으로는
괜찮은 솔루션이지만, 채점기가 묻는 것과 살짝 다른 질문에 답했어요.

## '무엇을'보다 '왜'를 물어요 {#why-ko}

숨은 규칙 7개는 전부 **무엇을** 셀지 말해요. **왜** 세는지는 하나도 말하지 않아요.
FDE라면 7개 중 어느 것보다 "왜" 질문 두 개가 더 멀리 데려가 줘요.

**교차는 왜 벌점일까요?** 요청서에는 구역이 겹치지 않게 해 달라고만 적혀 있어요.
채점기는 교차 한 번을 1,800초로 매겨요. 30분을 더 달린 것과 같아요. 침범 한 번은
10분이에요. 그런데 이유는 케이스 파일 어디에도 없어요. 배송에서 흔한 이유는 기사가
자기 구역을 잘 안다는 점, 두 트럭이 같은 길을 돌지 않는다는 점, 배차가 단순해진다는
점이에요. 다만 이건 제 추측이고, 고객의 답은 아니에요. 이 답이 중요한 건, 깔끔한
구역을 위해 주행을 얼마나 더 해도 되는지 알려 주기 때문이에요. 교차 한 번이 큰일인지
사소한 일인지도 알려 줘요.

**정답은 왜 부채꼴로 그려질까요?** 12개 경로는 모두 지도 귀퉁이의 창고 하나에서
나가고 돌아와요. 남의 다각형에 들어가면 안 되고 경로끼리 교차해도 안 되면, 지도를
깔끔하게 나누는 방법은 창고를 중심으로 부채꼴을 자르는 거예요. 그래서 부채꼴은
취향이 아니라 기하 규칙에서 나와요. 그다음 균등 규칙이 부채꼴마다 폭을 정해요. 멀리
뻗는 부채꼴은 매장을 적게 맡아요(2.7t 대 7.1t). 이 질문을 먼저 하면, 최적화를
시작하기 전에 답의 모양을 알 수 있어요. E는 모양은 맞혔고 폭만 틀렸어요.

### 가격표가 바뀌면 이기는 답이 바뀌어요 {#weights-ko}

비용 공식은 일종의 가격표예요. 주행 1초는 1, 가장 바쁜 트럭과 가장 한가한 트럭의
시간 차이 1초는 90이에요. 침범 한 번은 600, 교차 한 번은 1,800이에요. 다른 회사라면
다른 가격표를 쓸 거고, 그러면 다른 경로가 이겨요. 회사를 고르거나 슬라이더를 움직여서
순위가 바뀌는 걸 확인해 보세요.

<figure class="fde-fig" data-fde="weights" data-src="{{ '/assets/data/posts/fde-bench-002.json' | relative_url }}"></figure>
<p class="fde-caption">그림 7. 같은 경로 네 개를 다른 가중치로 다시 매긴 비용이에요. 막대는 비용을 네 항으로 나눈 거예요. 점수는 같은 가중치에서 FDE 정답의 비용을 그 경로의 비용으로 나눈 값이에요.</p>

- **주행 시간만 보는 회사:** B가 이겨요. 가장 적게 달려서예요. FDE 정답은 8.7% 더
  달려서 꼴찌예요.
- **구역만 보는 회사:** E가 1.059로 이겨요. 부채꼴이 깔끔하고, FDE 정답보다 6% 덜
  달려요.
- **이 고객:** FDE 정답이 이겨요. 균등 가중치는 1이어도 충분해요. 0.8을 넘으면 E를
  앞서거든요. 90이면 균등 항이 모든 걸 정해요.

이건 고정된 경로 네 개를 비교한 거예요. 가격표가 다르면 가장 좋은 경로의 모양부터
달라져요. 주행 시간만 보는 회사는 부채꼴을 그릴 이유가 없고, 구역만 보는 회사는
시간을 맞출 이유가 없어요. 그래서 가격표를 가장 먼저 알아내야 하고, 그걸 끌어내는
질문이 "왜"예요.

<div class="fde-callout" markdown="1">
<span class="fde-callout__title">다른 케이스도 같아요</span>
케이스 044는 은행 지점의 입지 점수를 예측하고, 평균 절대 오차(MAE)로 채점해요. 이
기준에서는 절대 오차로 학습한 모델이 이겨요(9.34 대 9.68). 제곱 오차(RMSE)로
채점했다면 제곱 오차로 학습한 모델이 이겼을 거예요(14.09 대 14.15). 숫자 하나로
찍는 가장 좋은 답도 바뀌어요. 절대 오차에서는 중앙값(11.01 대 11.53), 제곱 오차에서는
평균(16.44 대 17.01)이 이겨요.
</div>

## 케이스 031: 신용 부도 스코어카드 {#case-031-ko}

소비자 금융 회사가 새 신청자의 부도 확률 모델을 원해요. "부도 날 사람은 위로,
정상 상환할 사람은 아래로, 최대한 잘 갈리게"요. 학습 데이터 280행, 평가 120행(부도
33명)이에요.

여기서 숨은 규칙은 두 개뿐이에요. 순서만 본다(AUC와 KS를 반씩, 보정은 안 봄),
그리고 **같은 점수는 한 등급으로 친다**. 세 번째 규칙은 업무 정보에는 있지만 질문
대상에는 없어요. 결과는 0~1 사이 확률이어야 해요.

| 제출물 | AUC | KS | 점수 |
| --- | ---: | ---: | ---: |
| A · 로지스틱 회귀 | 0.876 | 0.591 | 0.848 |
| B · Gradient Boosting 기본값 | 0.830 | 0.551 | 0.798 |
| C · 300~850 스코어카드 점수 | – | – | **0** (범위 위반) |
| D · 로지스틱 원시 점수 | – | – | **0** (범위 위반) |
| E · A를 0.1 단위로 반올림 (11개 등급) | 0.858 | 0.522 | 0.797 |
| FDE 정답 | 0.927 | 0.805 | 1.000 |

은행에서 가장 익숙한 결과물인 300~850 스코어카드는 바로 무효예요. 케이스 이름이
"스코어카드"라서 오히려 그쪽으로 유도해요.

등급으로 뭉치면(E) 손해가 눈에 보여요.

<figure class="fde-fig" data-fde="ks" data-src="{{ '/assets/data/posts/fde-bench-031.json' | relative_url }}"></figure>
<p class="fde-caption">그림 8. 위험한 사람부터 줄 세웠을 때, 정상 고객을 통과시키는 만큼 부도 고객을 얼마나 잡는지예요. KS는 대각선 위로 가장 크게 벌어진 세로 간격이에요. 점은 E의 11개 등급이 끝나는 곳이에요.</p>

같은 점수를 한 등급으로 치면 채점기는 등급 사이에서만 자를 수 있어요. E의 곡선은
직선 11개로 바뀌고, 찾을 수 있는 가장 좋은 분할도 낮아져요(0.52 대 0.59). 실무에서
등급은 흔하지만, 여기서는 점수를 깎아요.

002와 반대 모양이에요. 숨은 규칙이 적고, 평범한 모델로도 0.85가 나와요. 남은
15% 는 모델링 실력과, 부도 33명이라는 작은 표본에서 오는 운으로 채워야 해요. 논문은
ML 케이스에서 정답 수준에 닿은 실행이 하나도 없었다고 보고했어요.

## 실제 일과 얼마나 가까울까요? {#limits-ko}

아직은 많이 단순한 세상이에요.

- **데이터에 노이즈가 없어요.** 무게, 좌표, 정답 라벨은 맞다고 보고 시작해요. 잘못
  붙은 라벨이 없고, 지저분한 부분은 요구사항에 다 적혀 있어요.
- **고객은 시뮬레이션이에요.** 정해진 문서 하나를 보고 답해요. 실제 고객은 잊어버리고,
  앞뒤가 다른 말을 하고, 마음도 바꿔요.
- **가격표는 고정돼 있고 숨어 있어요.** 실제 프로젝트에서 가중치는 회의에서 정해지고,
  계속 움직여요.

그래도 MMLU 같은 시험보다는 실제 에이전트 능력에 한 걸음 더 다가갔어요. MMLU는 정답이
하나인 객관식 문제를 내요. 여기서는 빠진 게 많은 요청서, 업무 데이터, 질문할 수 있는
고객을 받아요. 무엇을 물을지 정하고, 만들고, 돌려 보고, 확인해야 해요. 그리고 까다로운
채점기가 받아 줄 파일을 12시간 안에 내야 해요. 보기 넷 중 하나를 고르는 것보다 실제
일에 훨씬 가까워요.

## 정리 {#takeaways-ko}

- **0점은 대부분 계산이 아니라 규칙이에요.** 바로 무효가 된 제 제출물 네 개(002의
  A·C, 031의 C·D)는 전부 형식이나 업무 규칙을 어겼어요. 최적화나 모델이 모자라서
  0점이 된 건 하나도 없어요. 논문도 0점 실행의 78.5% 가 제약 위반이라고 했어요.
- **첫 질문거리는 데이터와 문구에 보여요.** 무게가 빈 행, "12대가 나갈 수 있다",
  단위 없는 "균등", 한 문장에 같이 나오는 "스코어카드"와 "확률". 전부 FDE가 첫
  미팅에서 던질 질문이에요.
- **"왜"를 물으면 답의 모양이 보여요.** 교차를 왜 벌점으로 매기는지, 정답이 왜
  부채꼴인지 물어보세요. 그러면 코드를 쓰기 전에 창고 주위로 부채꼴을 자르면 된다는
  걸 알 수 있어요.
- **규칙을 다 알아도 가중치는 숨어 있어요.** 7번 규칙은 순서만 주지 ×90은 안
  알려 줘요. 가중치는 고객이 직접 설계한 거라, 다르게 설계하면 다른 경로가
  이겨요([그림 7](#weights-ko)). 1.0에 닿으려면 이 고객이 진짜로 기대하는 것, 여기서는
  초 단위까지 맞춘 균등을 읽어 내야 해요.

---

*케이스 데이터: FDE-Bench (Li, Huang 외, 2026), CC BY-NC-ND 4.0. 그림은 해설을 위해
케이스 002의 매장 좌표와 케이스 031의 평가 정답으로 그렸고, 점수는 각 케이스의
`tests/evaluator.py` 로 냈어요. 제출물 A~G는 제가 만든 단순한 휴리스틱이고, 에이전트
실행이 아니에요.*

</div>
