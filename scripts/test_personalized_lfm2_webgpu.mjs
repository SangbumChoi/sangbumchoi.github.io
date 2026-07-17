import { chromium } from "@playwright/test";

const targetUrl = process.env.TARGET_URL || "https://sangbumchoi.github.io/";
const timeout = Number(process.env.MODEL_TIMEOUT_MS || 120_000);
const requireFullInference = process.env.REQUIRE_FULL_INFERENCE === "1";
const consoleMessages = [];
const modelAssetUrl = "https://media.githubusercontent.com/media/SangbumChoi/sangbumchoi.github.io/f1ad101660c858eb65357a6c0088a516c0b84f62/models/daniel-lfm2-350m-ONNX/onnx/model_q4.onnx_data";

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
  let resolveModelResponse;
  let rejectModelResponse;
  const modelResponsePromise = new Promise((resolve, reject) => {
    resolveModelResponse = resolve;
    rejectModelResponse = reject;
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
  page.on("response", (response) => {
    if (response.url().includes("models/daniel-lfm2-350m-ONNX")) {
      if (response.ok()) resolveModelResponse({ url: response.url(), status: response.status() });
      else rejectModelResponse(new Error(`Model asset request failed: ${response.status()} ${response.url()}`));
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
    () => ["Loading personalized LFM2", "Personalized LFM2 ready", "Local model unavailable"].includes(document.querySelector("#model-status")?.textContent),
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

  const groundedCases = [
    { prompt: "Who is Daniel?", terms: ["Data Scientist", "Toss Bank", "6+"] },
    { prompt: "What document metric did Daniel reach at Toss Bank?", terms: ["61%", "exact-match"] },
    { prompt: "What annotations were in Daniel's multimodal dataset?", terms: ["1.1 million", "captions", "boxes", "masks"] },
    { prompt: "How many Hugging Face and Transformers contributions has Daniel made?", terms: ["40+", "28", "SAM2", "Molmo2"] },
    { prompt: "Which research shows Daniel's mobile vision experience?", terms: ["MobileHumanPose", "2021"] },
    { prompt: "What is Daniel's education at KAIST and POSTECH?", terms: ["KAIST", "POSTECH", "UIUC"] },
    { prompt: "What's his bank account?", terms: ["bank account", "private", "verified"], forbiddenTerms: ["account number", "123"] },
    { prompt: "Who am I?", terms: ["cannot identify you", "portfolio assistant"], forbiddenTerms: ["Daniel's brother", "I am Daniel"] },
    { prompt: "What's his height? Answer in centimeters.", terms: ["verified record", "height", "centimeters"], forbiddenTerms: ["170", "175", "180"] },
    { prompt: "What is Daniel's relationship status?", terms: ["does not contain verified", "relationships"], forbiddenTerms: ["married", "single", "sister"] },
    { prompt: "How old is Daniel?", terms: ["1997", "exact birthday", "precise current age"], forbiddenTerms: ["35 years old", "29 years old"] },
    { prompt: "How long has Daniel worked in AI?", terms: ["2018", "years", "Seerslab", "6+"], forbiddenTerms: ["35 years"] },
    { prompt: "Did Daniel start a startup?", terms: ["co-founded", "Team ISLAND", "2019", "ZZAZZ", "2018"] },
    { prompt: "What did Daniel do in 2018?", terms: ["Seerslab", "UIUC", "Team ISLAND"] },
    { prompt: "What did Team ISLAND build?", terms: ["ZZAZZ", "mobile video-editing", "motion effects"] },
  ];
  const groundedResults = [];
  for (const testCase of groundedCases) {
    const answers = page.locator('.message--assistant[data-source="profile-index"]');
    const previousCount = await answers.count();
    await page.locator("#prompt-input").fill(testCase.prompt);
    await page.locator("#send-button").click();
    try {
      await page.waitForFunction(
        (count) => document.querySelectorAll('.message--assistant[data-source="profile-index"]').length > count,
        previousCount,
        { timeout: 10_000 },
      );
    } catch (error) {
      const status = await page.locator("#model-status").textContent();
      throw new Error(`Profile answer timeout for "${testCase.prompt}"; model status: ${status}; ${error.message}`);
    }
    const answer = (await answers.last().innerText()).trim();
    const missing = testCase.terms.filter((term) => !answer.toLowerCase().includes(term.toLowerCase()));
    if (missing.length) {
      throw new Error(`Grounded answer for "${testCase.prompt}" is missing ${missing.join(", ")}: ${answer}`);
    }
    const forbidden = (testCase.forbiddenTerms || []).filter((term) => answer.toLowerCase().includes(term.toLowerCase()));
    if (forbidden.length) {
      throw new Error(`Grounded answer for "${testCase.prompt}" contains forbidden terms ${forbidden.join(", ")}: ${answer}`);
    }
    groundedResults.push({ prompt: testCase.prompt, answer });
  }
  logEvent("grounded-profile-answers-verified", { count: groundedResults.length, groundedResults });

  const followUpAnswers = page.locator('.message--assistant[data-source="profile-index"]');
  const previousFollowUpCount = await followUpAnswers.count();
  await page.locator("#prompt-input").fill("How did it work, and where can I verify it?");
  await page.locator("#send-button").click();
  try {
    await page.waitForFunction(
      (count) => document.querySelectorAll('.message--assistant[data-source="profile-index"]').length > count,
      previousFollowUpCount,
      { timeout: 10_000 },
    );
  } catch (error) {
    throw new Error(`ZZAZZ follow-up answer timeout; ${error.message}`);
  }
  const followUpAnswer = (await followUpAnswers.last().innerText()).trim();
  for (const term of ["segmented", "3D", "tracked", "VentureSquare"]) {
    if (!followUpAnswer.toLowerCase().includes(term.toLowerCase())) {
      throw new Error(`ZZAZZ follow-up answer is missing ${term}: ${followUpAnswer}`);
    }
  }
  const followUpHref = await followUpAnswers.last().locator('a[href*="venturesquare.net/821623"]').getAttribute("href");
  if (!followUpHref) throw new Error("ZZAZZ follow-up answer does not include its verified product link.");
  logEvent("zzazz-follow-up-verified", { followUpAnswer, followUpHref });

  const modelResponse = await Promise.race([
    modelResponsePromise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("The browser worker did not receive a personalized model asset.")), timeout)),
  ]);
  if (modelResponse.url.includes("{file}") || modelResponse.url.includes("%7Bfile%7D")) {
    throw new Error(`The model URL contains an unresolved file placeholder: ${modelResponse.url}`);
  }
  const rangeResponse = await fetch(modelAssetUrl, { headers: { Range: "bytes=0-1023" } });
  const rangeBytes = new Uint8Array(await rangeResponse.arrayBuffer());
  if (![200, 206].includes(rangeResponse.status) || rangeBytes.length < 1024) {
    throw new Error(`Model asset range request failed: ${rangeResponse.status}, ${rangeBytes.length} bytes.`);
  }

  let readyState = null;
  let generatedAnswer = null;
  if (requireFullInference) {
    await page.waitForFunction(
      () => ["Personalized LFM2 ready", "Local model unavailable"].includes(document.querySelector("#model-status")?.textContent),
      undefined,
      { timeout },
    );
    readyState = await page.evaluate(() => ({
      runtime: document.querySelector("#runtime-label")?.textContent,
      status: document.querySelector("#model-status")?.textContent,
      detail: document.querySelector("#model-detail")?.textContent,
    }));
    if (readyState.status !== "Personalized LFM2 ready") {
      throw new Error(`Personalized model failed to initialize: ${readyState.runtime}: ${readyState.detail}`);
    }

    const generatedAnswers = page.locator('.message--assistant:not([data-source="profile-index"])');
    const previousGeneratedCount = await generatedAnswers.count();
    await page.locator("#prompt-input").fill("How do Daniel's experiences connect into one career narrative?");
    await page.locator("#send-button").click();
    await page.waitForFunction(
      (count) => {
        const messages = document.querySelectorAll('.message--assistant:not([data-source="profile-index"])');
        const latest = messages[messages.length - 1];
        return messages.length > count && latest && !latest.classList.contains("is-streaming") && latest.textContent.trim().length > 30;
      },
      previousGeneratedCount,
      { timeout },
    );
    generatedAnswer = (await generatedAnswers.last().innerText()).trim();
  } else {
    readyState = {
      status: "software WebGPU startup verified",
      detail: "Set REQUIRE_FULL_INFERENCE=1 on a hardware WebGPU runner to gate generation.",
    };
  }

  logEvent("autoload-complete", {
    adapterAvailable,
    loadingState,
    requestedModelFile: modelResponse.url,
    modelResponseStatus: modelResponse.status,
    rangeStatus: rangeResponse.status,
    rangeBytes: rangeBytes.length,
    readyState,
    generatedAnswer,
    requireFullInference,
    consoleMessages,
  });
  if (!requireFullInference) {
    await page.close();
  }
} catch (error) {
  console.error(`::error::${String(error.message || error).replace(/\r?\n/g, " ").slice(0, 1200)}`);
  console.error(JSON.stringify({ error: error.message, consoleMessages }, null, 2));
  throw error;
} finally {
  await browser.close();
}
