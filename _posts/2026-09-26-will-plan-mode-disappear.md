---
title: "Will Plan Mode Disappear?"
permalink: /posts/will-plan-mode-disappear/
date: 2026-09-26
last_modified_at: 2026-09-26
eyebrow: "ESSAY / CODING AGENTS"
dek: "Codex와 Claude Code를 쓰면서 든 생각. Planning은 사라지지 않는다. 다만 Plan Mode라는 명시적인 기능은 agent 내부로 흡수될 가능성이 높다."
read_time: true
comments: false
share: false
related: false
---

요즘 Codex나 Claude Code를 쓰면서 한 가지 생각이 든다.

**앞으로도 우리는 계속 Plan Mode를 사용할까?**

개인적으로는 그렇지 않을 것 같다.

정확히 말하면 planning 자체가 사라진다기보다, **Plan Mode라는 명시적인 기능**이
사라질 가능성이 높다고 생각한다.

## Plan Mode는 왜 필요했을까?

Coding agent에서 Plan Mode가 하는 일은 꽤 명확하다.

예를 들어 내가 이렇게 요청했다고 해보자.

> "Authentication을 추가해줘."

사람에게는 간단한 한 문장이지만 agent 입장에서는 그렇지 않다.

- 기존 authentication이 있는가?
- OAuth인가, email/password인가?
- 어떤 library를 사용할 것인가?
- DB schema를 바꿔야 하는가?
- 기존 API와 compatibility는 어떻게 유지할 것인가?
- test는 어디까지 추가할 것인가?

User query는 우리가 생각하는 것보다 훨씬 incomplete하다.

그래서 agent가 바로 코드를 작성하기보다 repository를 읽고, 요구사항을 해석하고,
task를 decomposition하고, implementation strategy를 만든 다음 사람에게 보여준다.
대략 이런 구조다.

```text
User Query → Interpret → Plan → Human Review → Execute
```

Anthropic도 Plan Mode를 비슷한 맥락에서 설명한다. 각 action마다 하나씩 승인을
요청하는 대신 Claude가 실행할 계획을 먼저 보여주고, 사용자는 이를 review, edit,
approve한다. Anthropic의 표현을 빌리면 이는 사용자의 oversight 수준을
*"individual step에서 overall strategy로"* 옮기는 것이다
([Anthropic, Trustworthy agents in practice](https://www.anthropic.com/research/trustworthy-agents)).

Claude Code 문서는 agent loop를 **gather context → take action → verify results**의
세 단계로 설명하고, 이 단계들이 서로 섞이며 반복된다고 말한다
([Claude Code Docs](https://code.claude.com/docs/en/how-claude-code-works)).
Plan Mode는 이 loop에서 context를 모으고 전략을 세우는 부분을 사용자에게 명시적으로
노출해서, execution 전에 검토할 수 있도록 만든 장치라고 볼 수 있다.

Codex에도 비슷한 Plan Mode가 있다. 공개된 Codex의
[Plan Mode template](https://github.com/openai/codex/blob/main/codex-rs/collaboration-mode-templates/templates/plan.md)을
보면, 이 mode에서 agent는 파일 읽기, 검색, static analysis처럼 repository를
바꾸지 않는 action으로 사실을 모으고 모호함을 줄인 뒤, 구현자가 추가 결정을 할
필요가 없는 *"decision complete"* plan을 만든다. 파일 수정이나 migration 같은
mutating action은 금지된다.

그런데 여기서 궁금한 점이 생긴다.

**왜 이것이 별도의 Mode여야 할까?**

## Planning은 사실 Critic에 더 가까울지도 모른다

나는 Plan Mode의 중요한 역할 중 하나가 단순히 "미래의 행동 순서를 만드는 것"이
아니라고 생각한다. 오히려 다음에 가깝다.

> **Criticize the user's instruction before executing it.**

사용자의 instruction을 그대로 실행하지 않고 먼저 묻는다.

- 사용자가 정말 원하는 것이 이것인가?
- 이 instruction에는 빠진 조건이 없는가?
- 지금 내가 하려고 하는 implementation이 실제 goal과 일치하는가?
- 더 좋은 방법은 없는가?

이렇게 보면 planning과 criticism의 경계가 흐려진다.

예를 들어 사용자가 이렇게 말했다고 하자.

> "User table에 `is_admin` boolean column을 추가하고 admin 기능을 만들어줘."

좋은 agent라면 migration부터 만들기보다 이렇게 생각할 수 있다.

> 잠깐. 이 서비스에는 organization별 role이 이미 존재한다.
> `is_admin`을 추가하면 authorization source가 두 개가 된다.
> 기존 RBAC을 확장하는 것이 더 자연스럽지 않을까?

이것은 planning일까? Criticism일까? Architecture review일까?

사실 굳이 구분할 필요가 없다. 좋은 agent에게 필요한 것은 execution 전에 자신의
interpretation과 action을 비판할 수 있는 능력이다.

## Plan Mode는 일종의 UX적 safety net이었다

그렇다면 왜 처음부터 이것을 agent 내부에서 처리하지 않고 Plan Mode라는 명시적인
feature로 만들었을까?

나는 이것의 일부가 **모델 성능과 신뢰의 문제를 UX로 해결하는 과정**이었다고
생각한다.

Agent가 사용자의 instruction을 임의로 재해석했다고 해보자. 사용자는

> "DB query 좀 빠르게 만들어줘."

라고 했는데 agent는 "schema를 바꾸는 게 가장 좋은 방법이다"라고 판단해서
migration까지 실행해버린다.

결과가 좋다면 문제가 없다. 하지만 틀렸다면 사용자는 이렇게 느낀다.

> "나는 그런 걸 시킨 적이 없는데?"

그래서 중간에 하나의 checkpoint를 넣는다.

```text
User Intent → Agent Interpretation → Human Approval → Execution
```

Agent가 "제가 이해하기로는 A, B, C를 하려고 합니다"라고 보여주고 사람이 Yes를
누른다.

Agent의 intelligence가 갑자기 좋아진 것은 아니다. 하지만 사용자가 agent의
interpretation을 한 번 검증했기 때문에, 실패했을 때의 surprise가 크게 줄어든다.

이런 의미에서 Plan Mode는 단순한 reasoning feature라기보다
**imperfect agent를 위한 soft-landing UX**였다고 볼 수도 있다.

## 우리는 이미 비슷한 transition을 보고 있다

흥미롭게도 permission에서도 거의 같은 현상이 일어나고 있다.

Anthropic에 따르면 Claude Code 사용자들은 permission prompt의 **93%**를 승인하고
있었다. 문제는 approval이 너무 많아지면 사람이 더 이상 제대로 읽지 않는다는 것이다.
Anthropic은 이를 *approval fatigue*라고 부르며, 사람 대신 model-based classifier가
각 action을 실행 전에 평가하는 **auto mode**를 만들었다
([Anthropic Engineering](https://www.anthropic.com/engineering/claude-code-auto-mode)).
그리고 현재 Claude Code 문서 기준으로 auto mode는 terminal과 VS Code
interactive session의 기본 시작 permission mode가 되었다
([Claude Code Docs](https://code.claude.com/docs/en/permission-modes)).

즉 이것이

```text
AI Action → Human Approval → Execute
```

이렇게 바뀌고 있다.

```text
AI Action → Internal Risk Evaluation → Execute
```

단, 위험도가 높은 순간에는 여전히 인간에게 묻는다.

OpenAI도 비슷한 방향을 보인다. OpenAI가 설명한 내부 Codex 운영 방식에서는
planned action과 최근 context를 별도의 auto-approval subagent가 검토하고, 낮은
위험의 행동은 사용자를 방해하지 않고 승인한다. 높은 위험이나 의도치 않은 결과가
가능한 행동에서는 멈추도록 설계되어 있다
([OpenAI, Running Codex safely at OpenAI](https://openai.com/index/running-codex-safely/)).

이 evolution을 planning에도 그대로 적용해볼 수 있다. 실제로 Codex repository에는
agent가 구현 도중 스스로 Plan Mode에 들어가고 나올 수 있게 해달라는
feature request도 올라와 있다
([openai/codex#35858](https://github.com/openai/codex/issues/35858)).
Planning에 들어갈지를 사람이 아니라 agent가 결정하게 하자는 요구다.

## Plan Mode가 아니라 Dynamic Planning

현재는 대략 이렇다.

```text
Plan Mode ON  → 많이 생각한다.
Plan Mode OFF → 바로 실행한다.
```

하지만 이 binary distinction이 꼭 필요한가? Agent가 스스로 판단하면 된다.

예를 들어:

> "Button color를 blue에서 green으로 바꿔줘."

여기에는 거의 planning이 필요 없다. 바로 실행하면 된다.

반대로:

> "우리 payment architecture를 Stripe Connect 기반으로 migration해줘."

라고 하면 이야기가 완전히 다르다. Agent는 repository를 탐색하고, dependency를
파악하고, migration risk를 분석하고, alternative architecture를 비교해야 한다.

그리고 중간에 중요한 결정이 발견될 수도 있다.

> Existing customers를 migration할 것인가?

이것은 agent가 임의로 결정하기에는 impact가 크다. 그 순간에만 사람에게 물으면 된다.

즉 미래의 agent loop는

```text
Plan → Approve → Execute
```

보다 다음과 가까울 것이다.

```text
Observe → Act → Critique → Act → Observe → Replan → Ask Human → Act
```

Planning이 하나의 phase가 아니라, 계속해서 조절되는 internal behavior가 된다.

## Planning의 정도 자체가 Hyperparameter가 될 수 있다

여기서 한 단계 더 나아갈 수 있다고 생각한다.

Planning을 ON/OFF로 생각하지 않고, **얼마나 멀리 발산할 것인가**라는 continuous
variable로 보는 것이다. Agent 내부에 개념적으로 이런 값들이 있다고 생각해볼 수 있다.

```text
exploration_depth
reasoning_budget
uncertainty_tolerance
autonomy_horizon
human_escalation_threshold
```

간단한 task에서는 exploration depth가 거의 0이다.

```text
Query → Execute
```

조금 애매하면 올라간다.

```text
Query → Inspect → Critique → Execute
```

복잡한 task라면 훨씬 커진다.

```text
Query → Explore → Generate alternatives → Critique → Gather context → Re-plan → Execute
```

그리고 irreversible하거나 가치 판단이 필요한 순간에는 human escalation threshold를
넘는다.

```text
→ Ask Human
```

이 관점에서는 Plan Mode라는 UI가 조금 이상해진다. 사용자가 왜 agent에게 매번
"이번에는 깊게 생각해"라고 알려줘야 하는가?

좋은 agent라면 문제의 complexity와 uncertainty를 보고, 얼마나 planning해야 하는지
스스로 판단해야 한다.

## Context Window도 Planning Budget의 일부가 될 수 있다

여기에는 context size 문제도 연결된다.

현재 agent에게 긴 task를 맡기면 context가 계속 쌓인다. 그래서 인간이 Plan Mode,
subagent, `/clear`, task decomposition 같은 방법으로 context를 관리한다.

하지만 이것 역시 점점 agent runtime 내부의 문제로 이동할 가능성이 있다.
Agent가 스스로 판단하는 것이다.

- 이 문제를 해결하는 데 과거 50K tokens가 정말 필요한가?
- 어떤 context를 유지해야 하는가?
- 어떤 정보는 summary로 압축할 수 있는가?
- 새로운 subagent에게 어떤 subset만 전달해야 하는가?
- 언제 global context를 다시 읽어야 하는가?

그러면 context window 역시 단순한 model specification이 아니라
**planning resource**가 된다. Agent는 task마다 필요한 reasoning depth, exploration
breadth, context budget을 동적으로 배분한다.

## 그러면 인간은 Query를 작성할 필요도 없어지는가?

이 생각을 끝까지 밀어붙이면 조금 이상한 질문에 도달한다.

Agent가 알아서

- intent를 해석하고,
- 부족한 정보를 찾고,
- alternative를 탐색하고,
- 자신의 계획을 criticize하고,
- context를 관리하고,
- 결과를 보고 다시 planning한다면,

인간은 왜 정확한 query를 작성해야 할까?

어쩌면 앞으로 인간이 agent에게 전달하는 것은 task specification이 아니라 훨씬
가벼운 direction일 수도 있다. 예를 들어:

> "요즘 onboarding conversion이 좀 안 좋은 것 같아."

현재의 AI라면 질문을 기다릴 가능성이 높다. 미래의 agent라면 여기에서 여러
hypothesis를 발산할 수 있다. Analytics를 확인하고, 최근 deployment를 보고,
drop-off point를 찾고, customer feedback을 읽고, 몇 가지 hypothesis를 만들고,
low-risk experiment를 설계한다.

그리고 정말 중요한 결정이 생겼을 때만 인간에게 돌아온다.

> "Signup step을 하나 제거하면 conversion 개선 가능성이 있지만 fraud protection이
> 약해질 수 있습니다. 이 trade-off는 어떤 방향으로 가져갈까요?"

인간의 역할은 점점

> *How should this be done?*

에서

> *What do we actually value?*

쪽으로 이동한다.

## Planning은 사라지지 않는다. Plan Mode가 사라질 수 있다.

그래서 내가 생각하는 미래는 planning이 없는 agent가 아니다. 오히려 반대다.
Agent는 지금보다 훨씬 더 많이 planning할 것이다.

다만 사용자가 매번 버튼을 눌러 "지금부터 planning하세요"라고 말할 필요가 없어질
가능성이 높다. Planning은 execution과 분리된 feature가 아니라, agent loop 안에
자연스럽게 녹아든다.

- 필요하면 2초 생각하고 바로 행동한다.
- 필요하면 repository 전체를 탐색한다.
- 필요하면 여러 alternative를 만든다.
- 필요하면 자기 생각을 criticize한다.
- 필요하면 context를 버리고 다시 시작한다.

그리고 **자신이 결정해서는 안 되는 순간에만 인간을 호출한다.**

## 한 가지 확인해보고 싶은 데이터

여기서 개인적으로 가장 궁금한 데이터가 있다.

**Codex와 Claude Code에서 explicit Plan Mode로 시작되는 task의 비율이 시간에 따라
어떻게 변하고 있을까?**

현재 공개 자료로는 이 usage ratio의 historical trend를 확인하기 어렵다. 그래서
"Plan Mode 사용률이 이미 감소하고 있다"라고 주장할 근거는 아직 없다. 이 글의
주장은 관찰과 가설이지, 측정된 사실이 아니다.

오히려 지금도 Claude Code는 explore → plan → implement → commit을 권장 workflow로
가르치고 있고
([Claude Code best practices](https://code.claude.com/docs/en/best-practices)),
Codex에서도 planning과 execution을 분리하는 Plan Mode가 제공된다.

다만 같은 문서에는 흥미로운 문장이 하나 있다. Plan mode는 overhead가 있으니,
*"diff를 한 문장으로 설명할 수 있다면 plan을 건너뛰라"*는 것이다. 지금은 이
판단을 사람이 한다. 이 글의 주장은 결국 그 판단이 agent로 넘어간다는 것이다.

하지만 모델의 self-critique와 autonomy가 좋아질수록, 나는 이 숫자가 장기적으로
내려갈 것이라고 예상한다. Permission prompt가 auto mode로 넘어간 흐름이 그
선행 지표일 수 있다.

만약 실제 데이터가 그렇다면 꽤 재미있는 의미를 갖는다.

Plan Mode가 실패한 feature여서 사라지는 것이 아니다.
**너무 성공해서 agent 내부로 흡수되는 것이다.**

1. 처음에는 planning을 인간에게 보여줘야 했다.
2. 그다음에는 인간이 planning을 승인했다.
3. 그다음에는 agent가 planning을 스스로 critique한다.
4. 그리고 결국 인간에게는 중요한 decision만 올라온다.

그래서 내가 궁금한 질문은 이것이다.

> *Will Plan Mode exist forever?*

내 생각에는,

**Planning will.**

**Plan Mode probably won't.**
