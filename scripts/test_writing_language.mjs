import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolveWritingLanguage } from "../assets/js/writing-language.mjs";

test("new visitors start with English", () => {
  assert.equal(resolveWritingLanguage("", null), "en");
  assert.equal(resolveWritingLanguage("?lang=invalid", null), "en");
});
test("saved Korean preference is restored", () => {
  assert.equal(resolveWritingLanguage("", "ko"), "ko");
  assert.equal(resolveWritingLanguage("", "invalid"), "en");
});
test("a shared language URL takes precedence over preference", () => {
  assert.equal(resolveWritingLanguage("?lang=en", "ko"), "en");
  assert.equal(resolveWritingLanguage("?lang=ko", "en"), "ko");
});
test("attention translation preserves executable examples and benchmark numbers", () => {
  const source = readFileSync(new URL("../_posts/2026-10-07-pytorch-attention-variants.md", import.meta.url), "utf8");
  const translation = readFileSync(new URL("../_includes/translations/attention-variants.en.md", import.meta.url), "utf8");
  const blocks = text => [...text.matchAll(/```[^\n]*\n([\s\S]*?)```/g)].map(match => match[1]);
  assert.deepEqual(blocks(translation), blocks(source));
  const numbers = text => text.split("\n").filter(line => line.startsWith("|")).map(line => line.match(/\d+(?:\.\d+)?/g) || []).filter(values => values.length >= 3);
  assert.deepEqual(numbers(translation), numbers(source));
});
