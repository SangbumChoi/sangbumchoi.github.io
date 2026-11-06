OCR models that turn table images into text have an odd habit. Give one a table where the same row repeats, and it writes one row too few. Think of a transaction list with the same date and amount on eight lines. Change a single value per row and it reads every row fine.

I looked into why [PaddleOCR-VL-1.5](https://arxiv.org/abs/2601.21957) (a 0.9B document-parsing model) does this. The method is attention masking. When the model writes a token, I block part of what it can look at, then see how the output changes. The weights stay the same; the masks apply only at inference.

Every table in these experiments is synthetic, and the values are made up.

## The symptom: identical-row tables come out a row short {#symptom}

The model writes tables as OTSL-style tokens:

- `<fcel>`: a cell with text
- `<ecel>`: an empty cell
- `<lcel>`, `<ucel>`: a cell merged with the one to its left or above
- `<nl>`: the end of a row

Writing EOS (the end-of-output token) after the last `<nl>` ends the table.

I made two kinds of 8-row table, 40 of each. In one kind every row is identical. In the other every row is different.

| Table | Wrote all 8 rows | Rows missing | Rows written |
| --- | --- | --- | --- |
| Every row different | 40 (100%) | 0 | 8 rows in all 40 |
| Every row identical | 1 (2.5%) | 39 (97.5%) | 7 rows in 35, 6 in 4, 8 in 1 |

On identical-row tables the model usually stops after 7 rows. The text it writes is right; it's the row count that's short.

## Its gaze runs ahead of what it writes {#gaze}

I measured which image row the model looks at while it writes each row.

- **How:** the correct output goes in as history (teacher forcing), and attention is averaged over all 18 layers.
- **Size:** 60 table pairs.

Switch the table type and the moment in the figure below.

<figure class="ocr-fig" data-ocr="heat" data-src="{{ '/assets/data/posts/ocr-row-masks.json' | relative_url }}"></figure>
<p class="ocr-caption">Each row's image attention is scaled to 100%. The last image row always gets a little attention, even on tables with unique rows.</p>

**On unique-row tables the pattern is a diagonal.** While writing row 3, the model looks at image row 3 most.

**On identical-row tables the gaze gets ahead from row 4 on.**
- While writing row 4, it looks at image row 5 most.
- While writing row 6, it's already on row 8, the last one.
- The row-end `<nl>` shows the same thing. At the `<nl>` that ends row 5, the most-watched row is already row 8. It should be row 6.

The reason is plausible. In an identical-row table, rows 4 and 5 look exactly the same in the image. So it's easy to lose track of where you are. The gaze reaches the last row early, so the model decides it's done and stops.

The probability of stopping, P(EOS), fits this picture:

| Table | After row 6 | After row 7 | After row 8 (the real end) |
| --- | --- | --- | --- |
| Every row different | 0.0% | 0.0% | 99.9% |
| Every row identical | 19.8% | 88.6% | 77.9% |

On identical-row tables the model stops after row 7 with 88.6% probability.

Attention only points at a cause, though. It doesn't prove one. So the next step was to apply masks.

## Fourteen masks {#masks}

There are two kinds of mask:

- Masks on every output token.
- Masks only on the row-end `<nl>` token. These touch only the moment the model decides whether to stop.

I added two controls that hide the same number of positions at random. They separate the effect of what's hidden from the effect of how much is hidden.

This time the model wrote the whole output on its own. I used 40 pairs of 8-row tables.
- **Exact:** the output matches the answer token for token.
- **Text similarity:** a character-level match that includes the cell separators.

| Mask | Identical rows: row count right | Short | Over | Exact | Text similarity | Unique rows: exact | Text similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No mask | 2.5 | 97.5 | 0 | 2.5 | 86.3 | 97.5 | 99.8 |
| Only the current row (needs row positions) | 100 | 0 | 0 | 80 | 99.1 | 85 | 99.3 |
| `<nl>` can't see earlier `<nl>` | 45 | 52.5 | 2.5 | 45 | 92.8 | 97.5 | 99.8 |
| Hide rows already read | 0 | 100 | 0 | 0 | 74.1 | 90 | 99.8 |
| At `<nl>`, hide rows already read | 0 | 100 | 0 | 0 | 82.2 | 97.5 | 99.8 |
| At `<nl>`, see only the next row | 0 | 100 | 0 | 0 | 49.8 | 97.5 | 99.8 |
| Hide rows not yet read | 0 | 75 | 25 | 0 | 35.4 | 0 | 33.8 |
| At `<nl>`, hide rows not yet read | 0 | 100 | 0 | 0 | 48.7 | 2.5 | 69.7 |
| At `<nl>`, hide the image | 5 | 95 | 0 | 0 | 34.2 | 0 | 61.5 |
| `<nl>` can't see earlier text | 0 | 97.5 | 2.5 | 0 | 56.7 | 0 | 66.7 |
| `<nl>` can't see any earlier output | 0 | 0 | 100 | 0 | 75.4 | 0 | 84.0 |
| Control: `<nl>` loses random earlier output | 0 | 0 | 100 | 0 | 75.5 | 0 | 84.0 |
| Hide earlier rows' output | 0 | 0 | 100 | 0 | 76.4 | 0 | 49.2 |
| Control: at `<nl>`, hide random image tokens | 0 | 100 | 0 | 0 | 78.2 | 27.5 | 91.0 |
| Hide the whole image | 0 | 95 | 5 | 0 | 23.0 | 0 | 22.9 |

All values are percentages of 40 tables, so a gap of a few points can be chance.

The figure below shows P(EOS) right after each row, per mask. The gray line is the identical-row table with nothing hidden. Switch masks and compare the points after rows 7 and 8.

<figure class="ocr-fig" data-ocr="eos" data-src="{{ '/assets/data/posts/ocr-row-masks.json' | relative_url }}"></figure>
<p class="ocr-caption">Average over 60 identical-row table pairs, measured with the correct output as history. A good line stays low through row 7 and jumps at row 8.</p>

## Pinning the gaze to the current row fixes every table {#oracle}

**"Only the current row" gets the row count right on all 40 identical-row tables.** Exact matches go from 2.5% to 80%. Here's what it does:

- While the model writes a row's cells, it can see only that row of the image.
- At `<nl>`, it can see only the next row.

P(EOS) is 0.1% after row 7 and 99.6% after row 8. When the gaze can't run ahead, the symptom goes away. That fits the "it loses its place" explanation.

But this mask is a diagnostic, not a fix.

- **It needs the answer.** You have to tell it where each row is in the image.
- **It hurts other tables.** On unique-row tables, exact matches drop from 97.5% to 85%.
- **It's worse on realistic tables.** On a separate, more realistic set, unique-row tables dropped from 72% to 28%.

## Hiding earlier `<nl>` tokens from `<nl>` brings back about half {#nl-nl}

**Of the masks that don't need row positions, this was the only one that helped.** It does one thing: the row-end `<nl>` token can't see the `<nl>` tokens written before it.

- Exact matches on identical-row tables go from 2.5% to 45%.
- Unique-row tables are unchanged: row count 100%, exact 97.5%.
- P(EOS) after row 7 falls from 88.6% to 46.4%.

I haven't confirmed why it works. My guess: earlier `<nl>` tokens may carry a running sense of which row the model is on. On identical-row tables that sense has already drifted a row or two ahead. A new `<nl>` that reads it inherits the error. Hiding the earlier `<nl>` tokens cuts that chain. Testing this needs its own experiment.

I also tried one synthetic table of each of a few other shapes. It's one table per shape, so don't read these as rates.

| Table shape | Rows | No mask | `<nl>` can't see earlier `<nl>` | Only the current row |
| --- | ---: | --- | --- | --- |
| Header + 8 identical rows | 9 | 8 rows (1 short) | exact | exact |
| Merged first column + identical rows | 9 | 8 rows (1 short) | exact | 9 rows, one cell wrong |
| Only the middle 4 rows identical | 8 | 7 rows (1 short) | 8 rows, one merge token differs | exact |
| Repayment schedule | 9 | exact | exact | exact |
| Every row different | 10 | exact | exact | exact |

## Three masks that only looked like fixes {#fake}

When `<nl>` can't see any earlier output, the "short" rate on identical-row tables drops to 0%. The same is true for the random control and for "hide earlier rows' output". If you count only short outputs, it looks solved.

But all three run over on all 40 tables. Most never stop and get cut off at the length limit. With every earlier output hidden from `<nl>`, P(EOS) never goes above 0.1% after any row. If the model can't see what it already wrote, it can't tell that it's done.

So in experiments like this, always count overshoot along with undershoot.

## Fixing only the stop decision makes it worse {#decision}

**When `<nl>` can see only the next image row, identical-row tables stop even earlier.**

- The model writes 3 to 5 rows, and text similarity falls from 86.3 to 49.8.
- P(EOS) is already 35% after row 3.

The next row looks exactly like the one just written, so the model seems to treat it as already done.

The other masks tell us a few more things:

- **Hiding rows already read changes nothing.** So confusing a row with the previous one isn't the cause.
- **Hiding rows not yet read makes it stop earlier.** The model keeps writing only while it can see rows left.
- **Hiding the whole image makes it invent tables.** It fills cells with the numbers 1 to 99, or repeats one word over and over.

## What this experiment can't tell you {#limits}

- **One model, synthetic tables only.** I didn't measure dropped rows on real documents. Real tables that lose rows without any repetition aren't explained by this.
- **The samples are small.** Rates come from 40 pairs, attention from 60 pairs, and the shape examples are one table each.
- **The attention runs and the full runs use different conditions.** Attention and P(EOS) were measured with the correct output as history. They may differ when the model writes on its own.
- **It isn't ready for serving.** To use "`<nl>` can't see earlier `<nl>`", the inference server (vLLM, for example) has to support a different attention mask per token.

## Takeaways {#takeaways}

1. When rows repeat, the model loses its place. Its gaze runs a row or two ahead, reaches the last row early, and it stops.
2. Pinning the gaze to the current row removes the symptom. It needs row positions, though, so it's a diagnostic.
3. Hiding earlier `<nl>` tokens from `<nl>` brought back about half the missing rows without row positions, and didn't hurt other tables. It's worth testing on real tables.
4. If you count only short outputs, a mask that never stops looks like a fix. Count overshoot too.
