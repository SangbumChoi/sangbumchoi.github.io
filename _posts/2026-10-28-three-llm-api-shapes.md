---
title: "chat/completions, messages, responses: why three LLM API shapes"
title_ko: "chat/completions, messages, responses: LLM API 는 왜 세 가지 모양일까"
permalink: /posts/three-llm-api-shapes/
date: 2026-10-28
last_modified_at: 2026-10-28
dek: "A tracing proxy that only understood /v1/chat/completions broke on Claude Code and Codex traffic. Here's how the three request shapes got where they are, and what that means for anything sitting in the middle."
dek_ko: "/v1/chat/completions 만 알던 추적 프록시가 Claude Code 와 Codex 트래픽에서 깨졌어요. 세 가지 요청 모양이 어떻게 지금 모습이 됐는지, 그리고 중간에서 트래픽을 읽는 쪽은 무엇을 챙겨야 하는지 정리했어요."
read_time: true
comments: false
share: false
related: false
---

<div class="lang-block" lang="en" markdown="1">

I was fixing a gateway that sits between coding agents and our model servers
and writes every call to a tracing tool. Plain chat calls looked fine. Calls
from Claude Code (`/v1/messages`) and Codex (`/v1/responses`) came out with an
empty output and no token counts. The code assumed every stream was a
Chat Completions stream.

Fixing it meant learning why the three shapes differ. They didn't drift apart
by accident. Each one answers a problem the one before it couldn't handle.

## Both companies took the same three steps {#steps}

| Step | OpenAI | Anthropic | What changed |
| --- | --- | --- | --- |
| 1. Text in, text out | `/v1/completions` (GPT-3) | Text Completions (`\n\nHuman: … \n\nAssistant:`) | The model only continues text. You fake a chat by writing the turn markers yourself. |
| 2. Messages with roles | Chat Completions (March 2023, `gpt-3.5-turbo`) | Messages API (now the standard; Text Completions is legacy) | The server applies the chat template. A user can't type "Assistant:" to pose as another role. |
| 3. Typed blocks or items | Responses API (March 2025) | Messages had `content` blocks from the start | One call now returns reasoning, tool calls and an answer mixed together. |

The third step is the one that matters for anything in the middle. Once a
response is a list of typed parts, "the assistant's text" stops being one
string.

## Why OpenAI built Responses {#responses}

Chat Completions assumes one request makes one assistant message. OpenAI kept
adding to it: function calling, then `tools` and `tool_calls`, image input,
the `developer` role. Three needs finally didn't fit.

1. **Reasoning had nowhere to go.** A reasoning model thinks before it
   answers. Chat Completions has no field to carry that thinking into the next
   turn, so it was dropped every turn. In an agent loop with many tool calls,
   that wastes work. Responses returns reasoning as its own output item. If you
   don't let OpenAI store data (`store: false`), you ask for
   `reasoning.encrypted_content` and send that item back in the next `input`.
2. **Tools that run on the server.** With built-in web search or file search,
   one response holds "search call → result → answer". A single assistant
   message can't hold that, so `output` became an array of typed items.
3. **State.** The Assistants API managed threads and runs, but you had to
   poll. Responses folds that into `previous_response_id` and `store`. OpenAI
   set the Assistants API to shut down on August 26, 2026, and said it would
   keep supporting Chat Completions.

The streams changed for the same reason. Chat Completions streams diffs of one
object. Responses streams named events such as `response.output_text.delta`
and `response.function_call_arguments.delta`. The last event,
`response.completed`, carries the whole final response.

Codex follows this all the way. It needs to pass reasoning from turn to turn,
so it dropped the `wire_api = "chat"` option. It added a deprecation warning
in December 2025, and newer versions reject it.

## Why Anthropic's Messages looks the way it does {#messages}

- **`system` is a top-level field, not a role.** The system prompt is a header
  for the whole conversation. `messages` holds only `user` and `assistant`
  turns, so the API shape itself keeps a system message from showing up in the
  middle.
- **`content` was a list of blocks from day one.** Text, image, `tool_use`,
  `tool_result`, document, and later `thinking`. New features add a block type
  instead of a new field.
- **Tool results go in the user turn.** Your code runs the tool, so its result
  comes from your side of the conversation, as a `tool_result` block.
- **`max_tokens` is required.** There's no default, so the caller decides how
  long, and how expensive, an answer can be.
- **The stream is block by block.** `content_block_start` → deltas →
  `content_block_stop` mark where each block begins and ends. A client can
  render text, tool calls and thinking separately. Tool input arrives as
  `input_json_delta` fragments, and the JSON is complete when its block
  closes.
- **Errors can arrive mid-stream.** HTTP 200 has already been sent, so the
  stream carries an `event: error` instead, for example `overloaded_error`.
- **`thinking` blocks have a `signature`.** It lets the server check that the
  thinking you send back is the thinking it produced. This solves the same
  problem as OpenAI's `encrypted_content`: neither side keeps the reasoning on
  the server, so the client carries a sealed copy.

## Same idea, different streams {#streams}

This table is what a proxy has to deal with. It's the part I got wrong first.

| | Chat Completions | Messages | Responses |
| --- | --- | --- | --- |
| Event names | none, only `data:` lines | `event:` line on every event | `event:` line on every event |
| Unit of a delta | a diff of `choices[0].delta` | a delta for one content block (by `index`) | a delta for one output item |
| End of stream | `data: [DONE]` | `message_stop` | `response.completed` (full response inside) |
| Where usage is | last chunk, only with `stream_options.include_usage` | `message_start`, then cumulative counts in `message_delta` | inside `response.completed` |
| Errors mid-stream | an error chunk | `event: error` | `error` or `response.failed` |

So the logging code needs one assembler per shape:

- **Chat Completions:** append each `delta.content` and merge `tool_calls` by
  index.
- **Messages:** open a block on `content_block_start`, append its deltas, and
  close it on `content_block_stop`. Read token counts from the last
  `message_delta`, because those counts are cumulative.
- **Responses:** you can skip the deltas. Take the response object out of
  `response.completed`. If the stream breaks before that event, keep the raw
  events so the call isn't lost.

Pick the assembler from the request path. Guessing from the first chunk works
until a new event type shows up.

## "Input tokens" means two different things {#usage}

This one doesn't throw an error. It just makes your numbers wrong.

| | Cache tokens | Total input |
| --- | --- | --- |
| OpenAI | `cached_tokens` is part of `prompt_tokens` | `prompt_tokens` |
| Anthropic | `cache_read_input_tokens` and `cache_creation_input_tokens` are separate from `input_tokens` | the sum of all three |

The difference follows from how each company bills caching. OpenAI caches
automatically and gives a discount, so cached tokens are a subset of the
prompt. Anthropic makes cache writes cost more (1.25× for a 5-minute cache)
and cache reads cost less (0.1× on most models), so each gets its own field.
If you copy `input_tokens` into a dashboard as "prompt tokens", a well-cached
agent session looks almost free.

## The shapes are converging {#converge}

Responses is closer to Messages than to Chat Completions:

- Output is a list of typed parts.
- Usage is named `input_tokens` and `output_tokens`.
- The system prompt has its own field (`instructions`).
- Stream events have names.

Open-source servers followed the clients. vLLM added a `/v1/messages`
endpoint in October 2025 so tools like Claude Code could talk to it directly.
Before that, people put translation proxies in front. On a server like that,
the model sees the same chat template whichever shape you send. The shapes
differ only on the wire.

That leaves Chat Completions as the one shape still built around a data-only
diff stream from 2023. A lot of tooling was written against it. That's why it
breaks first when agent traffic shows up.

## What I'd check in any LLM proxy {#checklist}

1. Choose the parser from the request path, not from the payload.
2. Write one stream assembler per shape, and test it with thinking and tool
   call blocks, not only text.
3. Normalize usage on purpose. Write down whether "input" includes cache
   tokens.
4. Keep the raw events when a stream ends early.
5. Check every place that reads the request, not only the logger. Fields like
   `system` and `instructions` live outside `messages`, so code that only reads
   `messages` misses them.

## Sources {#sources}

- OpenAI, [New tools for building agents](https://openai.com/index/new-tools-for-building-agents/) (Responses API launch)
- OpenAI, [Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses)
- OpenAI developer forum, [Assistants API beta deprecation, August 26, 2026 sunset](https://community.openai.com/t/assistants-api-beta-deprecation-august-26-2026-sunset/1354666)
- OpenAI developer forum, [Using `reasoning.encrypted_content` with `store=False`](https://community.openai.com/t/how-to-use-reasoning-encrypted-content-with-store-false-stateless/1286934)
- OpenAI, [Prompt caching in the API](https://openai.com/index/api-prompt-caching)
- Anthropic, [Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- Anthropic, [Prompt caching](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching)
- Anthropic, [Text Completions (legacy)](https://platform.claude.com/docs/en/api/http/completions.md)
- openai/codex, [Deprecating chat/completions support in Codex](https://github.com/openai/codex/discussions/7782)
- vllm-project/vllm, [Support Anthropic API /v1/messages endpoint (#22627)](https://github.com/vllm-project/vllm/pull/22627)

</div>

<div class="lang-block" lang="ko" markdown="1">

코딩 에이전트와 모델 서버 사이에 있는 게이트웨이를 고치고 있었어요. 이
게이트웨이는 호출을 하나도 빠짐없이 추적 도구에 남겨요. 일반 채팅 호출은 잘
남았어요. 그런데 Claude Code(`/v1/messages`)와 Codex(`/v1/responses`)에서 온
호출은 출력이 비어 있고 토큰 수도 없었어요. 코드가 모든 스트림을 Chat
Completions 스트림이라고 가정하고 있었거든요.

고치려면 세 가지 모양이 왜 다른지부터 알아야 했어요. 우연히 갈라진 게
아니에요. 새 모양은 앞의 모양이 못 풀던 문제를 하나씩 풀려고 나왔어요.

## 두 회사가 같은 세 단계를 거쳤어요 {#steps-ko}

| 단계 | OpenAI | Anthropic | 바뀐 점 |
| --- | --- | --- | --- |
| 1. 글을 넣고 글을 받기 | `/v1/completions` (GPT-3) | Text Completions (`\n\nHuman: … \n\nAssistant:`) | 모델은 글을 이어 쓰기만 해요. 대화처럼 보이게 하려면 차례 표시를 직접 써 넣어야 했어요. |
| 2. 역할이 붙은 메시지 | Chat Completions (2023년 3월, `gpt-3.5-turbo`) | Messages API (지금 표준. Text Completions 는 레거시) | 대화 템플릿은 서버가 붙여요. 사용자가 본문에 "Assistant:" 를 써서 다른 역할인 척할 수 없어요. |
| 3. 종류가 붙은 블록이나 아이템 | Responses API (2025년 3월) | Messages 는 처음부터 `content` 블록 구조였어요 | 한 번 호출하면 추론, 도구 호출, 답이 섞여서 나와요. |

중간에서 트래픽을 읽는 쪽에는 3단계가 중요해요. 응답이 종류가 붙은 조각들의
목록이 되면, "어시스턴트가 쓴 글"이 더는 문자열 하나가 아니에요.

## OpenAI 가 Responses 를 따로 만든 이유 {#responses-ko}

Chat Completions 는 "요청 하나에 어시스턴트 메시지 하나"를 전제로 만들어졌어요.
OpenAI 는 여기에 기능을 계속 붙였어요. function calling, `tools` 와
`tool_calls`, 이미지 입력, `developer` 역할이 차례로 붙었어요. 그러다 세 가지가
이 틀에 안 들어갔어요.

1. **추론을 담을 자리가 없었어요.** 추론 모델은 답하기 전에 먼저 생각해요.
   Chat Completions 에는 이 생각을 다음 턴으로 넘길 칸이 없어서, 턴마다
   버려졌어요. 도구를 여러 번 부르는 에이전트 루프에서는 한 일을 버리는
   셈이에요. Responses 는 추론을 별도 출력 아이템으로 돌려줘요. OpenAI 에
   데이터를 저장하지 않게 하면(`store: false`), `reasoning.encrypted_content`
   (암호화된 추론)를 받아서 다음 요청의 `input` 에 그대로 다시 넣으면 돼요.
2. **서버에서 도는 도구.** 내장 웹 검색이나 파일 검색을 쓰면 응답 하나에 "검색
   호출 → 결과 → 답"이 같이 들어가요. 어시스턴트 메시지 하나로는 이걸 담을 수
   없어서, `output` 을 종류가 붙은 아이템 배열로 바꿨어요.
3. **상태 관리.** Assistants API 는 thread 와 run 으로 대화 상태를 관리했지만,
   결과를 계속 물어봐야(폴링) 했어요. Responses 는 이 기능을
   `previous_response_id` 와 `store` 로 흡수했어요. OpenAI 는 Assistants API 를
   2026년 8월 26일에 종료하기로 했고, Chat Completions 는 계속 지원한다고
   밝혔어요.

스트림이 바뀐 것도 같은 이유예요. Chat Completions 는 객체 하나의 변경분(diff)을
흘려보내요. Responses 는 `response.output_text.delta`,
`response.function_call_arguments.delta` 처럼 이름이 붙은 이벤트를 보내요. 마지막
이벤트인 `response.completed` 에는 최종 응답 전체가 들어 있어요.

Codex 는 이 방향을 끝까지 따라갔어요. 추론을 턴마다 넘겨야 해서
`wire_api = "chat"` 설정을 없앴어요. 2025년 12월에 지원 중단 경고를 넣었고, 최근
버전은 이 설정을 아예 받지 않아요.

## Anthropic Messages 가 이렇게 생긴 이유 {#messages-ko}

- **`system` 은 역할이 아니라 맨 위 필드예요.** 시스템 프롬프트는 대화 전체의
  머리말이에요. `messages` 에는 `user` 와 `assistant` 턴만 들어가요. 그래서
  대화 중간에 시스템 메시지가 끼어드는 일을 API 구조가 막아요.
- **`content` 는 처음부터 블록 목록이었어요.** 텍스트, 이미지, `tool_use`,
  `tool_result`, 문서, 나중에는 `thinking` 까지 있어요. 기능이 늘면 새 필드
  대신 블록 종류만 늘리면 돼요.
- **도구 결과는 user 턴에 들어가요.** 도구를 실행하는 건 내 코드라서, 결과도
  내 쪽 차례에 `tool_result` 블록으로 넣어요.
- **`max_tokens` 는 꼭 넣어야 해요.** 기본값이 없어요. 답이 얼마나 길어도
  되는지, 그래서 비용이 얼마까지 나와도 되는지를 호출하는 쪽이 정해요.
- **스트림은 블록 단위예요.** `content_block_start` → 변경분 →
  `content_block_stop` 이 블록의 시작과 끝을 알려 줘요. 그래서 클라이언트는
  텍스트, 도구 호출, thinking 을 따로 그릴 수 있어요. 도구 입력은
  `input_json_delta` 조각으로 오다가, 블록이 닫힐 때 JSON 이 완성돼요.
- **오류가 스트림 중간에 올 수 있어요.** 이미 HTTP 200 을 보낸 뒤라 상태 코드로는
  알릴 수 없어요. 그래서 `event: error` 로 보내요. `overloaded_error` 가 한
  예예요.
- **`thinking` 블록에는 `signature`(서명)가 있어요.** 내가 다시 보낸 thinking 이
  서버가 만든 그대로인지 서버가 확인하는 데 써요. OpenAI 의
  `encrypted_content` 와 같은 문제를 풀어요. 두 회사 모두 추론을 서버에 두지
  않고, 클라이언트가 봉인된 사본을 들고 다니게 했어요.

## 생각은 같은데 스트림은 달라요 {#streams-ko}

프록시가 상대해야 하는 게 바로 이 표예요. 제가 처음에 틀린 부분이기도 해요.

| | Chat Completions | Messages | Responses |
| --- | --- | --- | --- |
| 이벤트 이름 | 없음. `data:` 줄만 와요 | 이벤트마다 `event:` 줄이 와요 | 이벤트마다 `event:` 줄이 와요 |
| 변경분 단위 | `choices[0].delta` 의 변경분 | 블록 하나(`index`)의 변경분 | 출력 아이템 하나의 변경분 |
| 스트림 끝 | `data: [DONE]` | `message_stop` | `response.completed` (응답 전체가 들어 있어요) |
| 토큰 수 위치 | 마지막 청크. `stream_options.include_usage` 를 켰을 때만 와요 | `message_start`, 그다음 `message_delta` 의 누적값 | `response.completed` 안 |
| 중간 오류 | 오류 청크 | `event: error` | `error` 또는 `response.failed` |

그래서 기록하는 코드는 모양마다 조립기(조각을 모아 응답 하나로 만드는 코드)가
따로 있어야 해요.

- **Chat Completions:** `delta.content` 를 계속 이어 붙이고, `tool_calls` 는
  index 별로 합쳐요.
- **Messages:** `content_block_start` 에서 블록을 열고, 변경분을 이어 붙이고,
  `content_block_stop` 에서 닫아요. 토큰 수는 마지막 `message_delta` 에서
  읽어요. 여기 숫자가 누적값이에요.
- **Responses:** 변경분은 건너뛰어도 돼요. `response.completed` 에서 응답 객체를
  꺼내면 돼요. 그 이벤트가 오기 전에 스트림이 끊기면, 원본 이벤트를 그대로
  남겨서 호출 기록을 잃지 않게 해요.

조립기는 요청 경로를 보고 골라요. 첫 청크를 보고 짐작하는 방식은 새 이벤트
종류가 나오는 순간 깨져요.

## "입력 토큰"이 회사마다 뜻이 달라요 {#usage-ko}

이건 오류가 나지 않아요. 숫자만 조용히 틀려요.

| | 캐시 토큰 | 전체 입력 |
| --- | --- | --- |
| OpenAI | `cached_tokens` 가 `prompt_tokens` 안에 들어 있어요 | `prompt_tokens` |
| Anthropic | `cache_read_input_tokens`, `cache_creation_input_tokens` 가 `input_tokens` 와 따로 있어요 | 셋을 더한 값 |

차이는 캐싱에 돈을 매기는 방식에서 나와요. OpenAI 는 자동으로 캐시하고 값을
깎아 줘요. 그래서 캐시 토큰이 프롬프트 토큰의 일부예요. Anthropic 은 캐시 쓰기가
더 비싸고(5분 캐시는 1.25배), 캐시 읽기가 더 싸요(대부분 모델에서 0.1배). 그래서
각각 따로 칸이 있어요. `input_tokens` 를 그대로 대시보드의 "프롬프트 토큰"에
넣으면, 캐시가 잘 맞는 에이전트 세션이 거의 공짜처럼 보여요.

## 세 모양이 서로 닮아 가요 {#converge-ko}

Responses 는 Chat Completions 보다 Messages 에 더 가까워요.

- 출력이 종류가 붙은 조각들의 목록이에요.
- 토큰 수 이름이 `input_tokens`, `output_tokens` 예요.
- 시스템 프롬프트가 따로 있어요(`instructions`).
- 스트림 이벤트에 이름이 붙어요.

오픈소스 서버도 클라이언트를 따라왔어요. vLLM 은 2025년 10월에 `/v1/messages`
엔드포인트를 넣었어요. Claude Code 같은 도구가 바로 붙을 수 있게 하려고요. 그
전에는 앞에 번역 프록시를 따로 세웠어요. 이런 서버에서는 어떤 모양으로 보내든
모델이 받는 대화 템플릿은 같아요. 모양 차이는 HTTP 로 주고받는 형태에만 있어요.

결국 2023년식 "`data:` 만 있는 변경분 스트림"에 남아 있는 건 Chat Completions
하나예요. 많은 도구가 이 모양을 기준으로 만들어졌어요. 그래서 에이전트 트래픽이
들어오면 거기서 먼저 깨져요.

## LLM 프록시라면 확인할 것 {#checklist-ko}

1. 파서는 본문 말고 요청 경로를 보고 골라요.
2. 스트림 조립기를 모양마다 따로 만들고, 텍스트만이 아니라 thinking 과 도구 호출
   블록으로도 시험해 봐요.
3. 토큰 수는 의도해서 맞춰요. "입력"에 캐시 토큰이 들어가는지 적어 둬요.
4. 스트림이 일찍 끊기면 원본 이벤트를 남겨요.
5. 기록 코드만 보지 말고, 요청을 읽는 곳을 다 확인해요. `system` 이나
   `instructions` 는 `messages` 밖에 있어요. `messages` 만 읽는 코드는 이걸
   놓쳐요.

## 출처 {#sources-ko}

- OpenAI, [New tools for building agents](https://openai.com/index/new-tools-for-building-agents/) (Responses API 출시)
- OpenAI, [Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses)
- OpenAI 개발자 포럼, [Assistants API beta 종료, 2026년 8월 26일](https://community.openai.com/t/assistants-api-beta-deprecation-august-26-2026-sunset/1354666)
- OpenAI 개발자 포럼, [`store=False` 에서 `reasoning.encrypted_content` 쓰기](https://community.openai.com/t/how-to-use-reasoning-encrypted-content-with-store-false-stateless/1286934)
- OpenAI, [Prompt caching in the API](https://openai.com/index/api-prompt-caching)
- Anthropic, [Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- Anthropic, [Prompt caching](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching)
- Anthropic, [Text Completions (레거시)](https://platform.claude.com/docs/en/api/http/completions.md)
- openai/codex, [Deprecating chat/completions support in Codex](https://github.com/openai/codex/discussions/7782)
- vllm-project/vllm, [Support Anthropic API /v1/messages endpoint (#22627)](https://github.com/vllm-project/vllm/pull/22627)

</div>
