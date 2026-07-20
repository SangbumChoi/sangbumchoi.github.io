import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("@playwright/test"));
} catch (_) {
  ({ chromium } = require("playwright"));
}

const targetUrl = process.env.TARGET_URL || "https://sangbumchoi.github.io/";
const timeout = Number(process.env.MODEL_TIMEOUT_MS || 120_000);
const requireFullInference = process.env.REQUIRE_FULL_INFERENCE === "1";
const consoleMessages = [];
const workerSource = await (await fetch(new URL(`/assets/js/lfm-worker.js?smoke=${Date.now()}`, targetUrl))).text();
const modelId = workerSource.match(/const MODEL_ID = "([^"]+)"/)?.[1];
const modelRevision = workerSource.match(/const MODEL_REVISION = "([0-9a-f]{40})"/)?.[1];
if (!modelId || !modelRevision) throw new Error("Could not read the deployed model identity from lfm-worker.js.");
const modelAssetUrl = `https://huggingface.co/${modelId}/resolve/${modelRevision}/onnx/model_q4.onnx_data`;

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
    if (response.url().includes(modelId) && response.url().includes("model_q4")) {
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
    () => ["Loading personalized LFM2", "Personalized LFM2 ready", "Personalized LFM2 available", "Development mock ready", "Local model unavailable"].includes(document.querySelector("#model-status")?.textContent),
    undefined,
    { timeout },
  );

  const runtimePolicy = await page.evaluate(() => window.__DANIEL_RUNTIME__?.policy);
  if (!runtimePolicy || runtimePolicy.maxResidentGpuModels !== 1) {
    throw new Error(`Missing single-residency runtime policy: ${JSON.stringify(runtimePolicy)}`);
  }
  const onDemandStatus = ["Personalized LFM2 available", "Development mock ready"]
    .includes(await page.locator("#model-status").textContent());
  if (onDemandStatus && requireFullInference) {
    await page.locator("#load-model").click();
    await page.waitForFunction(
      () => ["Loading personalized LFM2", "Personalized LFM2 ready", "Local model unavailable"].includes(document.querySelector("#model-status")?.textContent),
      undefined,
      { timeout },
    );
  }

  const loadingState = await page.evaluate(() => ({
    runtime: document.querySelector("#runtime-label")?.textContent,
    status: document.querySelector("#model-status")?.textContent,
    detail: document.querySelector("#model-detail")?.textContent,
    progress: document.querySelector(".progress-track")?.getAttribute("aria-valuenow"),
  }));
  const expectedRuntime = runtimePolicy.autoLoadModel ? "LLM webgpu /" : null;
  if (expectedRuntime && !loadingState.runtime?.startsWith(expectedRuntime)) {
    throw new Error(`Expected WebGPU autoload, received ${loadingState.runtime}: ${loadingState.detail}`);
  }
  if (!runtimePolicy.autoLoadModel && !onDemandStatus) {
    throw new Error(`Expected an on-demand model on constrained WebGPU, received ${loadingState.status}: ${loadingState.detail}`);
  }

  const groundedCases = [
    { prompt: "Who is Daniel?", terms: ["Data Scientist", "Toss Bank", "6+"] },
    { prompt: "What document metric did Daniel reach at Toss Bank?", terms: ["61%", "exact-match"] },
    { prompt: "What annotations were in Daniel's multimodal dataset?", terms: ["1.1 million", "captions", "boxes", "masks"] },
    { prompt: "How many Hugging Face and Transformers contributions has Daniel made?", terms: ["40+", "28", "SAM2", "Molmo2"] },
    { prompt: "Which research shows Daniel's mobile vision experience?", terms: ["MobileHumanPose", "2021"] },
    { prompt: "What is Daniel's education at KAIST and POSTECH?", terms: ["KAIST", "POSTECH", "UIUC"] },
    {
      prompt: "What is RT-DETR?",
      source: "entity-index",
      terms: ["Real-Time DEtection TRansformer", "end-to-end", "object detector"],
      forbiddenTerms: ["Daniel's work", "few-shot", "negative sampling"],
    },
    {
      prompt: "What is ViTPose?",
      source: "entity-index",
      terms: ["human-pose-estimation", "Vision Transformer", "keypoints"],
      forbiddenTerms: ["Daniel's work", "few-shot", "position and orientation"],
    },
    {
      prompt: "Where is UIUC?",
      source: "entity-index",
      terms: ["University of Illinois Urbana-Champaign", "Champaign-Urbana", "Illinois"],
      forbiddenTerms: ["KAIST", "POSTECH", "exchange student"],
    },
    {
      prompt: "What did Daniel contribute to RT-DETR?",
      source: "profile-entity-index",
      terms: ["Hugging Face Transformers", "model implementation", "training"],
      forbiddenTerms: ["Daniel created RT-DETR", "negative sampling"],
    },
    {
      prompt: "What is his bank account number?",
      source: "privacy-policy",
      terms: ["does not disclose", "private personal information"],
      forbiddenTerms: ["account number is", "123"],
    },
    {
      prompt: "Who wrote Pride and Prejudice?",
      source: "wikipedia-evidence",
      terms: ["novel", "Jane Austen", "Wikipedia source"],
      forbiddenTerms: ["2005 period romance film", "Daniel"],
    },
    {
      prompt: "Who created the Python programming language?",
      source: "wikipedia-evidence",
      terms: ["Python", "Guido van Rossum", "Wikipedia source"],
      forbiddenTerms: ["Daniel", "portfolio"],
    },
  ];
  const groundedResults = [];
  for (const testCase of groundedCases) {
    const source = testCase.source || "profile-index";
    const answers = page.locator(`.message--assistant[data-source="${source}"]`);
    const previousCount = await answers.count();
    await page.locator("#prompt-input").fill(testCase.prompt);
    await page.locator("#send-button").click();
    try {
      await page.waitForFunction(
        ({ count, source }) => document.querySelectorAll(`.message--assistant[data-source="${source}"]`).length > count,
        { count: previousCount, source },
        { timeout: 10_000 },
      );
    } catch (error) {
      const status = await page.locator("#model-status").textContent();
      throw new Error(`Profile answer timeout for "${testCase.prompt}"; model status: ${status}; ${error.message}`);
    }
    const answer = (await answers.last().locator(".message__body").innerText()).trim();
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
  logEvent("grounded-routing-answers-verified", { count: groundedResults.length, groundedResults });

  const modelResponse = requireFullInference
    ? await Promise.race([
      modelResponsePromise,
      new Promise((_, reject) => setTimeout(() => reject(new Error("The browser worker did not receive a personalized model asset.")), timeout)),
    ])
    : await Promise.race([
      modelResponsePromise,
      new Promise((resolve) => setTimeout(() => resolve(null), 5_000)),
    ]);
  if (modelResponse && (modelResponse.url.includes("{file}") || modelResponse.url.includes("%7Bfile%7D"))) {
    throw new Error(`The model URL contains an unresolved file placeholder: ${modelResponse.url}`);
  }
  const rangeResponse = await fetch(modelAssetUrl, { headers: { Range: "bytes=0-1023" } });
  const rangeBytes = new Uint8Array(await rangeResponse.arrayBuffer());
  if (![200, 206].includes(rangeResponse.status) || rangeBytes.length < 1024) {
    throw new Error(`Model asset range request failed: ${rangeResponse.status}, ${rangeBytes.length} bytes.`);
  }

  let readyState = null;
  let generatedAnswers = [];
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

    const askModel = async (testCase, reset = true) => {
      if (reset) await page.locator("#clear-chat").click();
      const answers = page.locator('.message--assistant:not([data-source="profile-index"])');
      const previousCount = await answers.count();
      await page.locator("#prompt-input").fill(testCase.prompt);
      await page.locator("#send-button").click();
      await page.waitForFunction(
        (count) => {
          const messages = document.querySelectorAll('.message--assistant:not([data-source="profile-index"])');
          const latest = messages[messages.length - 1];
          return messages.length > count && latest && !latest.classList.contains("is-streaming") && latest.textContent.trim().length > 20;
        },
        previousCount,
        { timeout },
      );
      const answer = (await answers.last().innerText()).trim();
      const normalized = answer.toLowerCase();
      const missingGroups = testCase.groups.filter(
        (group) => !group.some((term) => normalized.includes(term.toLowerCase())),
      );
      const forbidden = (testCase.forbiddenTerms || []).filter(
        (term) => normalized.includes(term.toLowerCase()),
      );
      if (missingGroups.length || forbidden.length) {
        throw new Error(`Model answer failed for "${testCase.prompt}"; missing=${JSON.stringify(missingGroups)}, forbidden=${forbidden.join(", ")}: ${answer}`);
      }
      generatedAnswers.push({ prompt: testCase.prompt, answer });
    };

    const modelCases = [
      { prompt: "Who am I?", groups: [["cannot identify", "cannot recognize"], ["portfolio assistant"]], forbiddenTerms: ["Daniel's brother", "I am Daniel"] },
      { prompt: "How old is Daniel?", groups: [["1997"], ["exact birthday"], ["cannot verify", "does not contain"]], forbiddenTerms: ["35 years old", "29 years old", "28 years old"] },
      { prompt: "How long has Daniel worked in AI?", groups: [["2018"], ["8+", "8 years"], ["6+"]], forbiddenTerms: ["35 years"] },
      { prompt: "What did Daniel do in 2018?", groups: [["Seerslab"], ["UIUC"], ["2019"]], forbiddenTerms: ["startup in 2018"] },
    ];
    for (const testCase of modelCases) await askModel(testCase);

    await askModel({
      prompt: "Did Daniel start a startup?",
      groups: [["Team ISLAND"], ["co-founded", "cofounder"], ["2019"], ["ZZAZZ"]],
      forbiddenTerms: ["jazz band", "music startup"],
    });
    await askModel({
      prompt: "What exactly was that product?",
      groups: [["ZZAZZ"], ["mobile video"], ["motion effects"]],
      forbiddenTerms: ["music streaming"],
    }, false);
    await askModel({
      prompt: "How did it work technically?",
      groups: [["detect", "segment"], ["3D"], ["track"], ["mobile"]],
      forbiddenTerms: ["cloud-only", "text-to-video"],
    }, false);
  } else {
    readyState = {
      status: "software WebGPU startup verified",
      detail: "Set REQUIRE_FULL_INFERENCE=1 on a hardware WebGPU runner to gate generation.",
    };
  }

  logEvent("autoload-complete", {
    adapterAvailable,
    loadingState,
    requestedModelFile: modelResponse?.url || null,
    modelResponseStatus: modelResponse?.status || null,
    modelResponseObserved: Boolean(modelResponse),
    rangeStatus: rangeResponse.status,
    rangeBytes: rangeBytes.length,
    readyState,
    generatedAnswers,
    modelRevision,
    modelId,
    requireFullInference,
    runtimePolicy,
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
