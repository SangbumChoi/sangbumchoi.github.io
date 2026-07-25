---
title: "Open-sourcing my first game: AI grids improved the workflow, not the file size"
permalink: /posts/open-sourcing-absorbed-ai-asset-pipeline/
date: 2026-07-25
last_modified_at: 2026-07-25
eyebrow: "FIELD NOTE / GAME DEVELOPMENT"
dek: "What I learned while turning AI-generated images into stable game assets, then publishing the project without its private history."
read_time: true
comments: false
share: false
related: false
---

I open-sourced [Absorbed](https://github.com/SangbumChoi/Absorbed), the first
game I have built, as a clean Apache 2.0 repository. It is an ongoing
landscape iPhone and iPad roguelike inspired by *Darkest Dungeon*,
*Slay the Spire*, and creature-collecting games, reinterpreted through
Buddhist ideas about attachment, identity, and letting go.

The most reusable lesson did not come from combat design. It came from learning
that generating a good image is only the first step. A game also needs stable
names, predictable crops, consistent dimensions, safe margins, and a way to add
new art without breaking old art.

![A Korean combat victory screen from Absorbed]({{ '/assets/images/posts/open-sourcing-absorbed-ai-asset-pipeline/absorbed-gameplay.png' | relative_url }})
*Absorbed is a playable English and Korean SwiftUI prototype. This captured combat result shows the dark, restrained visual language that the asset pipeline must preserve.*

## Image generation is not an asset pipeline

I first treated each icon as an individual generation task. That sounded
sensible: one prompt, one image, and maximum control. It also created repeated
work. Every new skill needed another prompt, another review, another filename,
and another opportunity for the visual language to drift.

> **In simple words - Asset pipeline:** An asset pipeline is the repeatable path that turns source art into files the game can load. It is the difference between owning a box of illustrations and having labeled pieces that always fit the board.
{: .notice--info}

The opposite extreme was not better. A single growing strip with five, six, or
more icons made each icon occupy less of the generated canvas. It also made
cell boundaries harder to detect reliably. A huge atlas looked efficient from
a distance, but it reduced the usable detail inside each icon.

The useful middle ground was a small, fixed batch:

- use a 2×2 source sheet for three or four related images;
- use separate sources for one or two images unless a native wide canvas is
  available;
- keep accepted batches immutable;
- add later skills as a new batch instead of regenerating an old sheet.

![A fixed two-by-two AI-generated skill source sheet]({{ '/assets/images/posts/open-sourcing-absorbed-ai-asset-pipeline/vajrapani-source-batch.png' | relative_url }})
*One immutable source batch contains four related Vajrapani skills. The shared generation gives the set a common material, palette, and visual weight.*

> **In simple words - Immutable batch:** Once a batch is accepted, I do not regenerate it merely to append another item. It behaves like a printed page: new material goes on a new page, so the approved cells never move.
{: .notice--info}

This rule protects more than aesthetics. Regenerating a sheet can subtly replace
every accepted image, even when I only wanted one new skill. An immutable batch
makes change local and reviewable.

## The manifest gives every cell an identity

A grid is still ambiguous unless the code knows what each cell means. I store
that identity in a **manifest**, a small JSON file that maps each row-major slot
to a stable game ID.

> **In simple words - Manifest:** A manifest is a packing list. It says which picture is in each cell, so a script does not have to guess from the image itself.
{: .notice--info}

The real skill manifest contains entries like this:

```json
{
  "sheet": "imagegen_vajrapani_skills.png",
  "columns": 2,
  "rows": 2,
  "slots": [
    "vajra_strike",
    "gate_stance",
    "adamant_quake",
    "immovable_mountain"
  ]
}
```

The [committed manifest](https://github.com/SangbumChoi/Absorbed/blob/main/Absorb/Assets/GeneratedAtlases/imagegen_skill_batches.json)
defines 15 skill batches with 60 filled slots. The gear manifest adds 31
batches for 100 weapons and 24 armor pieces. Three facility batches produce 11
building images. Together, these 49 fixed batches describe 195 outputs.

Those numbers are not a benchmark for every project. They describe the current
repository on July 25, 2026. The important property is that every output has an
explicit source, cell, and destination.

## Source layout and runtime layout solve different problems

The source grid is convenient for generation and review. The game should not
load that grid and guess a crop at runtime. A preparation script splits each
batch into independent 512×512 PNG files before the app is built.

![Diagram of one source grid becoming four independent runtime icons]({{ '/assets/images/posts/open-sourcing-absorbed-ai-asset-pipeline/asset-pipeline.png' | relative_url }})
*The source sheet optimizes generation and review. The runtime files optimize stable lookup, layout, and validation.*

> **In simple words - Runtime asset:** A runtime asset is the final file the app loads while it is running. The source sheet is a workshop material; the runtime icon is the finished part installed in the product.
{: .notice--info}

The [preparation script](https://github.com/SangbumChoi/Absorbed/blob/main/tools/prepare_skill_assets.py)
performs four jobs:

1. read the manifest and preserve its row-major order;
2. crop the declared cell from the source sheet;
3. fit the visible art into a 512×512 transparent tile;
4. validate dimensions, safe margins, duplicates, and missing files.

The output reserves an 8% margin on each side around the occupied art. This
keeps important detail away from UI masks and card borders.

> **In simple words - Safe area:** A safe area is empty breathing room around the important pixels. It is like keeping text away from the edge of a printed page so another frame does not cut it off.
{: .notice--info}

At runtime, skill collections use a stable 1×N horizontal row. Four icons fit
without shrinking; a fifth continues in the same scrolling row. That layout is
deliberately unrelated to the 2×2 generation grid. One shape helps the model
produce art, while another helps the player read the interface.

## The grid did not compress the bytes

I initially described batching as a storage improvement. Measuring the files
corrected that statement.

| Representative Vajrapani files | Files | Bytes | Purpose |
| --- | ---: | ---: | --- |
| One 2×2 source sheet | 1 | 2,975,117 | Generation source and visual review |
| Four normalized runtime icons | 4 | 1,675,150 | Stable files loaded by the game |
| Source and outputs retained together | 5 | 4,650,267 | Reproducibility plus runtime use |

The source sheet alone is larger than the four optimized runtime icons in this
example. Keeping both layers uses more disk space, not less.

The grid compressed the **workflow**:

- fewer generation units to prompt and review;
- stronger visual consistency inside one family;
- one manifest entry can describe four related outputs;
- accepted art remains stable when the catalog grows;
- one validator can enforce the same contract across every batch.

It did not automatically compress the repository. That distinction matters
because an efficient process and a small archive are separate engineering
goals.

## Publishing the code required a second pipeline

Opening the repository was not just a visibility switch. The working files
contained an Apple development-team identifier, a personal bundle identifier,
and Git author metadata from the private history. I removed those values,
changed the public bundle ID to `org.example.absorbed`, expanded the ignore
rules for environment and signing files, and rewrote the README for a new
reader.

The old repository also contained seven pull requests. A normal force-push
would not guarantee that their referenced commits disappeared. GitHub's own
[sensitive-data removal guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
explains that old commits can remain reachable through pull requests, cached
views, forks, and existing clones.

I therefore kept the original repository as a private archive and created a
fresh public repository from one neutral root commit. The public result has:

- one `main` branch;
- one commit by `Absorbed Contributors`;
- no inherited pull requests;
- no Apple signing team or personal bundle identifier;
- an Apache 2.0 license and a contributor-oriented README.

This approach preserved the development record privately without publishing
the private identity embedded in that record.

> **In simple words - Clean root commit:** A root commit is the first snapshot in Git history. Creating a clean one is like publishing a finished book without attaching every private draft and margin note used to write it.
{: .notice--info}

## What failed, and what each failure taught me

Four wrong assumptions improved the final system.

**One image per generation must give the most control.** It did give local
control, but it multiplied prompting and review. Small batches offered enough
control while making related icons easier to compare.

**A wider sheet must be more efficient.** Beyond four slots, each icon lost
usable resolution and crop reliability. The source format needed a strict
maximum.

**Fewer source files must mean fewer bytes.** The measured example disproved
that. Batching improved coordination, while retaining source and runtime layers
increased storage.

**Rewriting `main` must clean a repository.** Existing pull-request references
made that incomplete. A new public root was easier to explain and verify.

## How I checked the result

I ran the repository's full `bash tools/check.sh` gate before publication. It
passed all nine content, shop, asset, localization, parity, and deterministic
balance checks.

I also built the SwiftUI target with Xcode for the iOS Simulator on the local
Mac. The build passed with code signing disabled and the new generic bundle
identifier. A final scan found no tracked personal paths, development-team ID,
personal email, private-key signature, or live credential pattern.

After publication, I checked GitHub's API without authentication. It reported
one public branch, one commit, zero pull requests, and Apache 2.0 as the detected
license. These checks do not prove that the game is finished. They prove that
the published snapshot matches the release contract I intended.

## Learn the system, then leave room to forget it

AI tools make procedural judgment more valuable. The model can generate many
images, but someone still has to decide the batch size, naming rule, review
boundary, failure condition, and runtime contract.

There is also a tension. Learning established patterns makes work faster, but a
pattern can become a fence around imagination. Sometimes not knowing the
"correct" method creates enough freedom to invent a different one.

I do not think the answer is permanent ignorance. A better loop is:

```text
explore without a rule
        ↓
notice repeated pain
        ↓
design a small system
        ↓
measure where it helps
        ↓
break or replace it when the idea changes
```

The 2×2 grid is not a universal law. It is the smallest rule that solved this
project's current problem. The deeper skill is knowing why the rule exists, so
I can also recognize the moment when it should stop applying.

## Limits and next steps

This method works well for families of icons with a shared visual language. A
large character portrait or narrative scene may still deserve an individual
generation. The four-slot limit also reflects this project's canvas sizes and
crop behavior; another model or art direction may support a different limit.

The repository currently retains both generation sources and normalized
runtime files for reproducibility. A later storage pass could move heavyweight
sources to versioned releases or another artifact store while keeping manifests
and runtime assets in Git. That is planned, not implemented.

Absorbed itself remains a playable prototype rather than an App Store release.
The [public repository](https://github.com/SangbumChoi/Absorbed) includes the
game, bilingual content, preparation scripts, manifests, tests, and the
decisions behind them.

## Key terms

- **Asset pipeline:** the repeatable process that converts source art into
  validated files a game can load.
- **Immutable batch:** a fixed generation sheet that is not regenerated when
  later items are added.
- **Manifest:** structured metadata that maps each grid cell to a stable ID.
- **Runtime asset:** the final independent file loaded by the application.
- **Safe area:** reserved space around important visual content that protects it
  from masks, borders, and cropping.

---

## 한국어 요약

제가 처음 만든 게임 **Absorbed(흡수)**를 오픈소스로 공개하면서, AI 생성
에셋을 관리하는 방식도 함께 정리했습니다.

처음에는 이미지를 하나씩 생성하는 편이 가장 정교하다고 생각했습니다.
하지만 관련 이미지를 최대 네 칸의 작은 고정 그리드로 생성하고, 매니페스트로
각 칸의 ID를 기록한 뒤, 빌드 전에 독립된 512×512 런타임 파일로 분리하는
방식이 생성·검수·일관성 관리에 더 효과적이었습니다.

다만 실제 파일 크기를 측정해 보니 그리드가 저장 용량을 줄인 것은
아니었습니다. 대표 원본 시트는 약 2.98MB였고, 네 개의 런타임 아이콘은
합계 약 1.68MB였습니다. 원본과 결과물을 모두 보관하면 오히려 용량은
늘어납니다. 그리드가 압축한 것은 **파일 크기가 아니라 작업 과정**이었습니다.

AI 시대에는 효율적인 규칙을 설계하는 감각이 중요합니다. 동시에 규칙을 너무
당연하게 받아들이면 새로운 개념을 상상할 여지가 줄어들 수 있습니다. 결국
중요한 것은 규칙을 깊이 배우고, 왜 필요한지 측정하고, 더 이상 맞지 않을
때는 기꺼이 버릴 수 있는 능력인지도 모릅니다.
