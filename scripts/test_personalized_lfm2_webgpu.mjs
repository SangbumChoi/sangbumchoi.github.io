import { chromium } from "@playwright/test";

const targetUrl = process.env.TARGET_URL || "https://sangbumchoi.github.io/";
const timeout = Number(process.env.MODEL_TIMEOUT_MS || 120_000);
const consoleMessages = [];
const modelAssetUrl = "https://media.githubusercontent.com/media/SangbumChoi/sangbumchoi.github.io/7792cb8a5dbde55140a40658e7d9d6605d2c63d9/models/daniel-lfm2-350m-ONNX/onnx/model_q4.onnx_data";

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
  let resolveModelRequest;
  const modelRequestPromise = new Promise((resolve) => {
    resolveModelRequest = resolve;
  });
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
  page.on("request", (request) => {
    if (request.url().includes("models/daniel-lfm2-350m-ONNX")) {
      resolveModelRequest(request.url());
    }
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
  await page.waitForFunction(
    () => ["Loading personalized LFM2", "Personalized LFM2 ready"].includes(document.querySelector("#model-status")?.textContent),
    undefined,
    { timeout },
  );

  const loadingState = await page.evaluate(() => ({
    runtime: document.querySelector("#runtime-label")?.textContent,
    status: document.querySelector("#model-status")?.textContent,
    detail: document.querySelector("#model-detail")?.textContent,
    progress: document.querySelector(".progress-track")?.getAttribute("aria-valuenow"),
  }));
  if (!loadingState.runtime?.startsWith("webgpu /")) {
    throw new Error(`Expected WebGPU autoload, received ${loadingState.runtime}: ${loadingState.detail}`);
  }

  const requestedModelFile = await Promise.race([
    modelRequestPromise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("The browser worker did not request a personalized model asset.")), timeout)),
  ]);
  const rangeResponse = await fetch(modelAssetUrl, { headers: { Range: "bytes=0-1023" } });
  const rangeBytes = new Uint8Array(await rangeResponse.arrayBuffer());
  if (![200, 206].includes(rangeResponse.status) || rangeBytes.length < 1024) {
    throw new Error(`Model asset range request failed: ${rangeResponse.status}, ${rangeBytes.length} bytes.`);
  }

  logEvent("autoload-complete", {
    adapterAvailable,
    loadingState,
    requestedModelFile,
    rangeStatus: rangeResponse.status,
    rangeBytes: rangeBytes.length,
    consoleMessages,
  });
} catch (error) {
  console.error(JSON.stringify({ error: error.message, consoleMessages }, null, 2));
  throw error;
} finally {
  await browser.close();
}
