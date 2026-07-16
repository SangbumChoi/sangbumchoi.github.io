import { chromium } from "@playwright/test";

const targetUrl = process.env.TARGET_URL || "https://sangbumchoi.github.io/";
const timeout = Number(process.env.MODEL_TIMEOUT_MS || 900_000);
const consoleMessages = [];
let progressTimer;

function logEvent(event, details = {}) {
  console.log(JSON.stringify({ event, at: new Date().toISOString(), ...details }));
}

const browser = await chromium.launch({
  headless: true,
  args: [
    "--no-sandbox",
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--disable-vulkan-surface",
    "--enable-unsafe-webgpu",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--use-webgpu-adapter=swiftshader",
  ],
});

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("console", (message) => {
    const entry = `${message.type()}: ${message.text()}`;
    if (["error", "warning"].includes(message.type())) consoleMessages.push(entry);
    logEvent("browser-console", { message: entry });
  });
  page.on("pageerror", (error) => consoleMessages.push(`pageerror: ${error.message}`));
  page.on("requestfailed", (request) => {
    const entry = `requestfailed: ${request.url()} (${request.failure()?.errorText || "unknown"})`;
    consoleMessages.push(entry);
    logEvent("request-failed", { message: entry });
  });

  logEvent("navigation-start", { targetUrl });
  await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
  const adapterAvailable = await page.evaluate(async () => {
    if (!("gpu" in navigator)) return false;
    return Boolean(await navigator.gpu.requestAdapter());
  });
  if (!adapterAvailable) throw new Error("No WebGPU adapter is available in remote Chromium.");
  logEvent("webgpu-adapter-ready");

  await page.locator('label[for="voice-output"]').click();
  progressTimer = setInterval(async () => {
    try {
      const state = await page.evaluate(() => ({
        status: document.querySelector("#model-status")?.textContent,
        detail: document.querySelector("#model-detail")?.textContent,
        progress: document.querySelector(".progress-track")?.getAttribute("aria-valuenow"),
      }));
      logEvent("model-progress", state);
    } catch (error) {
      logEvent("progress-read-failed", { message: error.message });
    }
  }, 15_000);
  await page.waitForFunction(
    () => document.querySelector("#model-status")?.textContent === "Personalized LFM2 ready",
    undefined,
    { timeout },
  );
  clearInterval(progressTimer);
  progressTimer = undefined;

  const readyState = await page.evaluate(() => ({
    runtime: document.querySelector("#runtime-label")?.textContent,
    detail: document.querySelector("#model-detail")?.textContent,
    progress: document.querySelector(".progress-track")?.getAttribute("aria-valuenow"),
  }));
  if (readyState.runtime !== "webgpu / private") {
    throw new Error(`Expected WebGPU runtime, received ${readyState.runtime}: ${readyState.detail}`);
  }

  const prompt = page.getByLabel("Ask Daniel a question");
  await prompt.fill("In one short sentence, describe your role as Daniel's portfolio assistant.");
  await page.getByRole("button", { name: "Send question" }).click();
  await page.waitForFunction(
    () => document.querySelector("#latency-label")?.textContent?.startsWith("WEBGPU"),
    undefined,
    { timeout: 180_000 },
  );

  const result = await page.evaluate(() => {
    const messages = [...document.querySelectorAll("#chat-log .message--assistant .message__body")];
    return {
      answer: messages.at(-1)?.textContent?.trim(),
      latency: document.querySelector("#latency-label")?.textContent,
      status: document.querySelector("#model-status")?.textContent,
    };
  });
  if (!result.answer || result.answer.length < 10) {
    throw new Error("Personalized model generation returned an empty response.");
  }

  logEvent("generation-complete", { adapterAvailable, readyState, result, consoleMessages });
} catch (error) {
  console.error(JSON.stringify({ error: error.message, consoleMessages }, null, 2));
  throw error;
} finally {
  clearInterval(progressTimer);
  await browser.close();
}
