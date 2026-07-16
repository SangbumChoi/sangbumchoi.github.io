import { chromium } from "@playwright/test";

const targetUrl = process.env.TARGET_URL || "https://sangbumchoi.github.io/";
const timeout = Number(process.env.MODEL_TIMEOUT_MS || 900_000);
const consoleMessages = [];

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
    if (["error", "warning"].includes(message.type())) {
      consoleMessages.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleMessages.push(`pageerror: ${error.message}`));

  await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
  const adapterAvailable = await page.evaluate(async () => {
    if (!("gpu" in navigator)) return false;
    return Boolean(await navigator.gpu.requestAdapter());
  });
  if (!adapterAvailable) throw new Error("No WebGPU adapter is available in remote Chromium.");

  await page.locator('label[for="voice-output"]').click();
  await page.getByRole("button", { name: "Enable local AI" }).click();
  await page.waitForFunction(
    () => document.querySelector("#model-status")?.textContent === "Personalized LFM2 ready",
    undefined,
    { timeout },
  );

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

  console.log(JSON.stringify({ adapterAvailable, readyState, result, consoleMessages }, null, 2));
} catch (error) {
  console.error(JSON.stringify({ error: error.message, consoleMessages }, null, 2));
  throw error;
} finally {
  await browser.close();
}
