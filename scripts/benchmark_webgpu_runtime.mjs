import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("@playwright/test"));
} catch (_) {
  ({ chromium } = require("playwright"));
}

const targetUrl = process.env.TARGET_URL || "http://localhost:4000/";
const fullModel = process.env.FULL_MODEL === "1";
const contention = process.env.CONTENTION === "1";
const emulateLowSpec = process.env.EMULATE_LOW_SPEC === "1";
const softwareWebgpu = process.env.SOFTWARE_WEBGPU === "1";
const headless = process.env.HEADLESS !== "0";
const timeout = Number(process.env.MODEL_TIMEOUT_MS || 300_000);
const benchmarkPrompt = "What pattern connects the problems Daniel chooses to solve across his career?";

if (contention && !fullModel) throw new Error("CONTENTION=1 requires FULL_MODEL=1.");

const args = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"];
if (softwareWebgpu) {
  args.push(
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--disable-vulkan-surface",
    "--enable-unsafe-swiftshader",
    "--use-webgpu-adapter=swiftshader",
  );
}

const browser = await chromium.launch({ headless, args });
const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });

if (emulateLowSpec) {
  await context.addInitScript(() => {
    Object.defineProperty(navigator, "deviceMemory", { configurable: true, get: () => 4 });
    Object.defineProperty(navigator, "hardwareConcurrency", { configurable: true, get: () => 4 });
  });
}

async function openRuntimePage() {
  const page = await context.newPage();
  const diagnostics = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) diagnostics.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => diagnostics.push(`pageerror: ${error.message}`));
  await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
  try {
    await page.waitForFunction(() => Boolean(window.__DANIEL_RUNTIME__), undefined, { timeout: 30_000 });
  } catch (error) {
    throw new Error(`Runtime probe did not initialize. ${diagnostics.join(" | ") || error.message}`);
  }
  const runtime = await page.evaluate(() => window.__DANIEL_RUNTIME__);
  return { page, runtime, diagnostics };
}

async function ensureModelReady(page) {
  const status = page.locator("#model-status");
  const initialStatus = (await status.textContent())?.trim();
  const startedAt = performance.now();
  if (["Personalized LFM2 available", "Development mock ready"].includes(initialStatus)) {
    await page.locator("#load-model").click();
  }
  await page.waitForFunction(
    () => ["Personalized LFM2 ready", "Local model unavailable"].includes(document.querySelector("#model-status")?.textContent),
    undefined,
    { timeout },
  );
  const finalStatus = (await status.textContent())?.trim();
  if (finalStatus !== "Personalized LFM2 ready") {
    throw new Error(`Model initialization failed: ${await page.locator("#model-detail").textContent()}`);
  }
  return Math.round(performance.now() - startedAt);
}

async function runGeneration(page, prompt) {
  await page.locator("#clear-chat").click();
  const previous = await page.locator('.message--assistant:not([data-source="profile-index"])').count();
  const startedAt = performance.now();
  await page.locator("#prompt-input").fill(prompt);
  await page.locator("#send-button").click();
  await page.waitForFunction(
    (count) => {
      const messages = document.querySelectorAll('.message--assistant:not([data-source="profile-index"])');
      const latest = messages[messages.length - 1];
      return messages.length > count && latest && !latest.classList.contains("is-streaming") && latest.textContent.trim().length > 20;
    },
    previous,
    { timeout },
  );
  return {
    elapsedMs: Math.round(performance.now() - startedAt),
    workerElapsedMs: await page.evaluate(() => window.__DANIEL_RUNTIME__.session.lastGenerationElapsed || null),
    answer: (await page.locator('.message--assistant:not([data-source="profile-index"])').last().innerText()).trim(),
  };
}

try {
  const first = await openRuntimePage();
  const modelResponse = await fetch(new URL("/assets/js/lfm-worker.js", targetUrl));
  const workerSource = await modelResponse.text();
  const modelId = workerSource.match(/const MODEL_ID = "([^"]+)"/)?.[1];
  const revision = workerSource.match(/const MODEL_REVISION = "([0-9a-f]{40})"/)?.[1];
  const weightUrl = modelId && revision
    ? `https://huggingface.co/${modelId}/resolve/${revision}/onnx/model_q4.onnx_data`
    : null;
  const rangeResponse = weightUrl ? await fetch(weightUrl, { headers: { Range: "bytes=0-0" } }) : null;
  const contentRange = rangeResponse?.headers.get("content-range") || "";
  const weightBytes = Number(contentRange.match(/\/(\d+)$/)?.[1] || 0) || null;

  const report = {
    targetUrl,
    emulatedLowSpec: emulateLowSpec,
    softwareWebgpu,
    headless,
    capabilities: first.runtime.capabilities,
    policy: first.runtime.policy,
    observedWeightBytes: weightBytes,
    modelId,
    modelRevision: revision,
    note: emulateLowSpec
      ? "Policy emulation only; latency and memory are not low-end hardware measurements."
      : "Capabilities and timings come from this browser process.",
  };

  if (fullModel) {
    report.firstLoadMs = await ensureModelReady(first.page);
    report.warmGeneration = await runGeneration(first.page, benchmarkPrompt);
  }

  if (contention) {
    const second = await openRuntimePage();
    report.secondLoadMs = await ensureModelReady(second.page);
    const simultaneousStartedAt = performance.now();
    report.concurrentGeneration = await Promise.all([
      runGeneration(first.page, benchmarkPrompt),
      runGeneration(second.page, benchmarkPrompt),
    ]);
    report.concurrentWallMs = Math.round(performance.now() - simultaneousStartedAt);
  }

  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}
