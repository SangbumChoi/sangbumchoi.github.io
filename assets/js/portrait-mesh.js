const MOUTH_TARGETS = {
  rest: { width: 1, height: 1, open: 0, jaw: 0, smile: 0 },
  open: { width: 0.98, height: 1.05, open: 0.075, jaw: 0.018, smile: 0 },
  wide: { width: 1.16, height: 0.93, open: 0.035, jaw: 0.006, smile: 0.028 },
  round: { width: 0.78, height: 1.2, open: 0.09, jaw: 0.014, smile: 0 },
};

const MORPH_KEYS = Object.keys(MOUTH_TARGETS.rest);
const MOUTH_X_CORRECTION = 0.26;

function objectPositionFactor(value, fallback) {
  if (!value) return fallback;
  if (value === "left" || value === "top") return 0;
  if (value === "right" || value === "bottom") return 1;
  if (value === "center") return 0.5;
  if (value.endsWith("%")) return Number.parseFloat(value) / 100;
  return fallback;
}

function createGrid(bounds, columns, rows) {
  const points = [];
  const triangles = [];

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const u = column / (columns - 1);
      const v = row / (rows - 1);
      points.push({
        x: bounds.minX + (bounds.maxX - bounds.minX) * u,
        y: bounds.minY + (bounds.maxY - bounds.minY) * v,
        u,
        v,
      });
    }
  }

  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const topLeft = row * columns + column;
      const topRight = topLeft + 1;
      const bottomLeft = topLeft + columns;
      const bottomRight = bottomLeft + 1;
      if ((row + column) % 2 === 0) {
        triangles.push([topLeft, topRight, bottomRight], [topLeft, bottomRight, bottomLeft]);
      } else {
        triangles.push([topLeft, topRight, bottomLeft], [topRight, bottomRight, bottomLeft]);
      }
    }
  }

  return { bounds, points, triangles };
}

function boundedRegion(center, width, height, canvasWidth, canvasHeight, columns, rows) {
  const halfWidth = Math.min(width / 2, center.x, canvasWidth - center.x);
  const halfHeight = Math.min(height / 2, center.y, canvasHeight - center.y);
  return createGrid({
    minX: center.x - halfWidth,
    minY: center.y - halfHeight,
    maxX: center.x + halfWidth,
    maxY: center.y + halfHeight,
  }, columns, rows);
}

function edgeWeight({ u, v }) {
  return Math.max(0, Math.sin(Math.PI * u) * Math.sin(Math.PI * v));
}

function deformMouth(region, morph) {
  const { bounds } = region;
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;

  return region.points.map((point) => {
    const nx = (point.x - centerX) / Math.max(1, width / 2);
    const ny = (point.y - centerY) / Math.max(1, height / 2);
    const influence = Math.pow(edgeWeight(point), 0.72);
    const lipSide = Math.tanh(ny * 7);
    const lowerFace = Math.max(0, ny);
    const corner = Math.max(0, Math.abs(nx) - 0.35) / 0.65;
    return {
      x: point.x + (point.x - centerX) * (morph.width - 1) * influence,
      y: point.y
        + (point.y - centerY) * (morph.height - 1) * influence
        + lipSide * morph.open * height * influence
        + lowerFace * morph.jaw * height * influence
        - corner * morph.smile * height * influence,
    };
  });
}

function deformEye(region, blink, gaze) {
  const { bounds } = region;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  const width = bounds.maxX - bounds.minX;

  return region.points.map((point) => {
    const influence = Math.pow(edgeWeight(point), 0.78);
    return {
      x: point.x + gaze * width * 0.045 * influence,
      y: point.y + (centerY - point.y) * blink * 0.86 * influence,
    };
  });
}

// Map each source triangle onto its deformed destination while retaining the portrait pixels.
function affineTransform(source, destination) {
  const [s0, s1, s2] = source;
  const [d0, d1, d2] = destination;
  const denominator = s0.x * (s1.y - s2.y) + s1.x * (s2.y - s0.y) + s2.x * (s0.y - s1.y);
  if (Math.abs(denominator) < 0.001) return null;

  const a = (d0.x * (s1.y - s2.y) + d1.x * (s2.y - s0.y) + d2.x * (s0.y - s1.y)) / denominator;
  const b = (d0.y * (s1.y - s2.y) + d1.y * (s2.y - s0.y) + d2.y * (s0.y - s1.y)) / denominator;
  const c = (d0.x * (s2.x - s1.x) + d1.x * (s0.x - s2.x) + d2.x * (s1.x - s0.x)) / denominator;
  const d = (d0.y * (s2.x - s1.x) + d1.y * (s0.x - s2.x) + d2.y * (s1.x - s0.x)) / denominator;
  const e = (
    d0.x * (s1.x * s2.y - s2.x * s1.y)
    + d1.x * (s2.x * s0.y - s0.x * s2.y)
    + d2.x * (s0.x * s1.y - s1.x * s0.y)
  ) / denominator;
  const f = (
    d0.y * (s1.x * s2.y - s2.x * s1.y)
    + d1.y * (s2.x * s0.y - s0.x * s2.y)
    + d2.y * (s0.x * s1.y - s1.x * s0.y)
  ) / denominator;
  return { a, b, c, d, e, f };
}

function expandedTriangle(points) {
  const centerX = (points[0].x + points[1].x + points[2].x) / 3;
  const centerY = (points[0].y + points[1].y + points[2].y) / 3;
  return points.map((point) => ({
    x: centerX + (point.x - centerX) * 1.012,
    y: centerY + (point.y - centerY) * 1.012,
  }));
}

function drawTexturedTriangle(context, texture, source, destination) {
  const transform = affineTransform(source, destination);
  if (!transform) return;
  const clip = expandedTriangle(destination);

  context.save();
  context.beginPath();
  context.moveTo(clip[0].x, clip[0].y);
  context.lineTo(clip[1].x, clip[1].y);
  context.lineTo(clip[2].x, clip[2].y);
  context.closePath();
  context.clip();
  context.setTransform(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f);
  context.drawImage(texture, 0, 0);
  context.restore();
}

function renderRegion(context, texture, region, destinationPoints) {
  region.triangles.forEach((indices) => {
    const source = indices.map((index) => region.points[index]);
    const destination = indices.map((index) => destinationPoints[index]);
    drawTexturedTriangle(context, texture, source, destination);
  });
}

function blendMorph(current, target, amount) {
  MORPH_KEYS.forEach((key) => {
    current[key] += (target[key] - current[key]) * amount;
  });
}

function morphDistance(current, target) {
  return MORPH_KEYS.reduce((distance, key) => distance + Math.abs(current[key] - target[key]), 0);
}

export function createPortraitMeshAnimator({ canvas, image, media, features }) {
  const context = canvas.getContext("2d", { alpha: true });
  const texture = document.createElement("canvas");
  const textureContext = texture.getContext("2d", { alpha: true });
  if (!context || !textureContext || !features?.lips || !features?.leftEye || !features?.rightEye) {
    throw new Error("Portrait texture mesh could not initialize.");
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const currentMorph = { ...MOUTH_TARGETS.rest };
  let state = "idle";
  let viseme = "rest";
  let mouthRegion = null;
  let eyeRegions = [];
  let frameRequest = 0;
  let lastFrameAt = performance.now();
  let lastRenderAt = 0;
  let nextBlinkAt = performance.now() + 3800;
  let blinkStartedAt = 0;
  let ready = false;

  function mapBounds(bounds, fit, ratio) {
    const centerX = fit.offsetX + ((bounds.minX + bounds.maxX) / 2) * fit.width;
    const centerY = fit.offsetY + ((bounds.minY + bounds.maxY) / 2) * fit.height;
    return {
      x: centerX * ratio,
      y: centerY * ratio,
      width: (bounds.maxX - bounds.minX) * fit.width * ratio,
      height: (bounds.maxY - bounds.minY) * fit.height * ratio,
    };
  }

  function resize() {
    const cssWidth = Math.max(1, media.clientWidth);
    const cssHeight = Math.max(1, media.clientHeight);
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const pixelWidth = Math.max(1, Math.round(cssWidth * ratio));
    const pixelHeight = Math.max(1, Math.round(cssHeight * ratio));
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
    texture.width = pixelWidth;
    texture.height = pixelHeight;

    const scale = Math.max(cssWidth / image.naturalWidth, cssHeight / image.naturalHeight);
    const fittedWidth = image.naturalWidth * scale;
    const fittedHeight = image.naturalHeight * scale;
    const position = getComputedStyle(image).objectPosition.trim().split(/\s+/);
    const offsetX = (cssWidth - fittedWidth) * objectPositionFactor(position[0], 0.5);
    const offsetY = (cssHeight - fittedHeight) * objectPositionFactor(position[1], 0.5);
    const fit = { offsetX, offsetY, width: fittedWidth, height: fittedHeight };

    textureContext.setTransform(1, 0, 0, 1, 0, 0);
    textureContext.clearRect(0, 0, pixelWidth, pixelHeight);
    textureContext.imageSmoothingEnabled = true;
    textureContext.imageSmoothingQuality = "high";
    textureContext.drawImage(
      image,
      offsetX * ratio,
      offsetY * ratio,
      fittedWidth * ratio,
      fittedHeight * ratio,
    );

    const lips = mapBounds(features.lips, fit, ratio);
    const mouthCenter = { x: lips.x + lips.width * MOUTH_X_CORRECTION, y: lips.y };
    const mouthWidth = Math.max(18 * ratio, lips.width * 1.7);
    const mouthHeight = Math.max(10 * ratio, lips.height * 4, lips.width * 0.62);
    mouthRegion = boundedRegion(mouthCenter, mouthWidth, mouthHeight, pixelWidth, pixelHeight, 8, 6);

    eyeRegions = [features.leftEye, features.rightEye].map((bounds) => {
      const eye = mapBounds(bounds, fit, ratio);
      const eyeWidth = Math.max(10 * ratio, eye.width * 1.65);
      const eyeHeight = Math.max(8 * ratio, eye.height * 4.2, eye.width * 0.72);
      return boundedRegion(eye, eyeWidth, eyeHeight, pixelWidth, pixelHeight, 6, 6);
    });

    ready = true;
    render(0, 0);
  }

  function blinkAmount(now) {
    if (reducedMotion) return 0;
    if (!blinkStartedAt && now >= nextBlinkAt) blinkStartedAt = now;
    if (!blinkStartedAt) return 0;

    const elapsed = now - blinkStartedAt;
    const duration = 170;
    if (elapsed >= duration) {
      blinkStartedAt = 0;
      nextBlinkAt = now + 4200 + Math.random() * 2800;
      return 0;
    }
    return Math.pow(Math.sin((elapsed / duration) * Math.PI), 1.35);
  }

  function render(blink, gaze) {
    if (!ready || !mouthRegion || eyeRegions.length !== 2) return;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(texture, 0, 0);
    eyeRegions.forEach((region) => renderRegion(context, texture, region, deformEye(region, blink, gaze)));
    renderRegion(context, texture, mouthRegion, deformMouth(mouthRegion, currentMorph));
    canvas.dataset.renderer = "texture-mesh";
    canvas.dataset.blink = blink.toFixed(3);
    canvas.dataset.viseme = viseme;
  }

  function tick(now) {
    const elapsed = Math.min(80, Math.max(0, now - lastFrameAt));
    lastFrameAt = now;
    const target = state === "speaking" && !reducedMotion
      ? MOUTH_TARGETS[viseme] || MOUTH_TARGETS.open
      : MOUTH_TARGETS.rest;
    const blend = 1 - Math.exp(-elapsed / 105);
    blendMorph(currentMorph, target, blend);

    const blink = blinkAmount(now);
    const gaze = reducedMotion ? 0 : Math.sin(now / 2600) * 0.32 + Math.sin(now / 4700) * 0.12;
    const active = state === "speaking" || blink > 0.001 || morphDistance(currentMorph, target) > 0.002;
    const renderInterval = active ? 32 : 110;
    if (!document.hidden && now - lastRenderAt >= renderInterval) {
      render(blink, gaze);
      lastRenderAt = now;
    }
    frameRequest = window.requestAnimationFrame(tick);
  }

  return {
    start() {
      resize();
      frameRequest = window.requestAnimationFrame(tick);
    },
    resize,
    setState(next) {
      state = next;
    },
    setViseme(next) {
      viseme = MOUTH_TARGETS[next] ? next : "open";
    },
    destroy() {
      window.cancelAnimationFrame(frameRequest);
    },
  };
}
