const THREE_MODULE = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js";
const DEPTH_MAP_SIZE = 128;
const OVERSCAN = 1.12;
const FOV = 28;

// Mouth shapes driven by the existing viseme timeline in jarvis.js.
const VISEMES = {
  rest: { open: 0, wide: 0, round: 0 },
  open: { open: 1, wide: 0, round: 0 },
  wide: { open: 0.45, wide: 1, round: 0 },
  round: { open: 0.85, wide: 0, round: 1 },
};

// Facial expression and head pose per conversation state.
const EXPRESSIONS = {
  idle: { brow: 0, smile: 0.18, pitch: 0, yaw: 0, roll: 0, lean: 0, gazeX: 0, gazeY: 0, squint: 0 },
  listening: { brow: 0.85, smile: 0.1, pitch: 0.02, yaw: 0, roll: 0.045, lean: 0.03, gazeX: 0, gazeY: 0, squint: 0 },
  thinking: { brow: 0.4, smile: 0, pitch: -0.07, yaw: 0.09, roll: -0.02, lean: 0, gazeX: 0.75, gazeY: -0.55, squint: 0.18 },
  speaking: { brow: 0.22, smile: 0.26, pitch: 0, yaw: 0, roll: 0, lean: 0.015, gazeX: 0, gazeY: 0, squint: 0 },
};

const VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uDepth;
  uniform vec4 uImgMap;
  uniform vec2 uPlaneSize;
  uniform float uOverscan;
  uniform float uDepthScale;
  uniform vec3 uPivot;
  uniform vec3 uHeadRot;
  uniform vec3 uHeadOffset;
  uniform float uBreath;
  varying vec2 vImg;
  varying float vHead;

  mat3 rotation(vec3 r) {
    float cx = cos(r.x), sx = sin(r.x);
    float cy = cos(r.y), sy = sin(r.y);
    float cz = cos(r.z), sz = sin(r.z);
    mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx);
    mat3 ry = mat3(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);
    mat3 rz = mat3(cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0);
    return rz * ry * rx;
  }

  void main() {
    vec2 viewport = vec2(0.5 + (uv.x - 0.5) * uOverscan, 0.5 - (uv.y - 0.5) * uOverscan);
    vec2 img = viewport * uImgMap.xy + uImgMap.zw;
    vImg = img;
    vec4 depth = texture2D(uDepth, clamp(img, 0.0, 1.0));
    float head = depth.g;
    vHead = head;

    vec3 world = vec3(position.xy * uPlaneSize, depth.r * uDepthScale);
    vec3 local = world - uPivot;
    local = rotation(uHeadRot * head) * local;
    world = uPivot + local + uHeadOffset * head;
    world.y += uBreath * (0.35 + 0.65 * head);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  uniform sampler2D uMap;
  uniform sampler2D uDepth;
  uniform float uImgAspect;
  uniform vec2 uMouthCenter;
  uniform vec2 uMouthAxis;
  uniform vec2 uMouthSize;
  uniform float uMouthCurve;
  uniform vec4 uMouth;
  uniform vec4 uEyeL;
  uniform vec4 uEyeR;
  uniform vec4 uEyes;
  uniform float uBrow;
  uniform float uShade;
  varying vec2 vImg;
  varying float vHead;

  vec2 iso(vec2 v) { return vec2(v.x * uImgAspect, v.y); }
  vec2 deiso(vec2 v) { return vec2(v.x / uImgAspect, v.y); }

  // Raise the brows by pulling image content upward above each eye.
  vec2 browWarp(vec2 img, vec4 eye) {
    vec2 rel = iso(img - eye.xy);
    vec2 center = vec2(0.0, -2.1 * eye.w);
    vec2 d = (rel - center) / vec2(1.7 * eye.z, 1.5 * eye.w);
    float w = exp(-dot(d, d));
    return img + vec2(0.0, uBrow * eye.w * 0.75 * w);
  }

  // Shift the iris region sideways and vertically for gaze.
  vec2 gazeWarp(vec2 img, vec4 eye) {
    vec2 rel = iso(img - eye.xy);
    float r = length(rel / vec2(eye.z, eye.w));
    float w = 1.0 - smoothstep(0.45, 1.0, r);
    vec2 shift = vec2(-uEyes.y * eye.z * 0.16, -uEyes.z * eye.w * 0.22) * w;
    return img + deiso(shift);
  }

  vec2 mouthWarp(vec2 img, out float interior, out float interiorT, out float gapSize) {
    interior = 0.0;
    interiorT = 0.0;
    vec2 rel = iso(img - uMouthCenter);
    vec2 perp = vec2(-uMouthAxis.y, uMouthAxis.x);
    float a = dot(rel, uMouthAxis);
    float b = dot(rel, perp);
    float halfW = uMouthSize.x;
    float lipH = uMouthSize.y;

    float widthScale = 1.0 + 0.16 * uMouth.y - 0.24 * uMouth.z;
    float fall = (1.0 - smoothstep(1.1, 1.8, abs(a) / halfW)) * (1.0 - smoothstep(1.6, 3.2, abs(b) / lipH));
    float scale = mix(1.0, widthScale, fall);
    float aSrc = a / scale;
    float dx = aSrc / halfW;
    float lens = sqrt(max(0.0, 1.0 - dx * dx));
    // Follow the drawn lip line: the corners sit above or below the lip center.
    float bLine = uMouthCurve * min(dx * dx, 1.0);
    b -= bLine;

    float gapHalf = uMouth.x * lens * lipH * 1.05;
    float gUp = gapHalf * 0.7;
    float gDown = gapHalf * 1.3;
    gapSize = gUp + gDown;

    float bSrc = b;
    if (b < 0.0) {
      float s = -b;
      float r = lipH * 3.0;
      bSrc = -(s - gUp * (1.0 - smoothstep(gUp, gUp + r, s)));
      if (s < gUp) interior = 1.0;
    } else {
      float r = lipH * 4.6;
      bSrc = b - gDown * (1.0 - smoothstep(gDown, gDown + r, b));
      if (b < gDown) interior = 1.0;
    }
    if (gapSize < 1e-5) interior = 0.0;
    interiorT = clamp((b + gUp) / max(gapSize, 1e-5), 0.0, 1.0);

    // Smile lifts the mouth corners.
    bSrc += uMouth.w * lipH * 0.9 * dx * dx * fall + bLine;
    vec2 src = uMouthAxis * aSrc + perp * bSrc;
    return uMouthCenter + deiso(src);
  }

  vec3 eyelid(vec3 color, vec2 img, vec4 eye) {
    float blink = clamp(uEyes.x, 0.0, 1.0);
    if (blink < 0.01) return color;
    vec2 rel = iso(img - eye.xy);
    vec2 d = rel / vec2(eye.z * 1.08, eye.w * 1.25);
    float inside = 1.0 - smoothstep(0.85, 1.05, length(d));
    if (inside <= 0.0) return color;
    float lidEdge = -1.0 + 2.0 * blink;
    float covered = 1.0 - smoothstep(lidEdge - 0.08, lidEdge + 0.08, d.y);
    vec2 skinSample = eye.xy + deiso(vec2(rel.x, -1.9 * eye.w));
    vec3 skin = texture2D(uMap, skinSample).rgb * 0.94;
    vec3 lidded = mix(color, skin, covered);
    float lash = (1.0 - smoothstep(0.0, 0.14, abs(d.y - lidEdge))) * smoothstep(0.35, 0.9, blink);
    lidded = mix(lidded, vec3(0.1, 0.08, 0.08), lash * 0.85);
    return mix(color, lidded, inside);
  }

  void main() {
    vec2 img = vImg;
    img = browWarp(img, uEyeL);
    img = browWarp(img, uEyeR);
    img = gazeWarp(img, uEyeL);
    img = gazeWarp(img, uEyeR);
    float interior;
    float interiorT;
    float gapSize;
    vec2 src = mouthWarp(img, interior, interiorT, gapSize);
    vec3 color = texture2D(uMap, clamp(src, 0.0, 1.0)).rgb;

    if (interior > 0.5) {
      vec3 cavity = mix(vec3(0.2, 0.07, 0.08), vec3(0.08, 0.03, 0.04), 1.0 - abs(interiorT - 0.5) * 2.0);
      float teethOn = smoothstep(0.35, 0.8, gapSize / uMouthSize.y);
      float teeth = (1.0 - smoothstep(0.18, 0.3, interiorT)) * teethOn;
      float tongue = smoothstep(0.62, 0.85, interiorT);
      cavity = mix(cavity, vec3(0.5, 0.2, 0.2), tongue * 0.7);
      cavity = mix(cavity, vec3(0.88, 0.86, 0.82), teeth);
      color = cavity;
    }

    color = eyelid(color, img, uEyeL);
    color = eyelid(color, img, uEyeR);

    // Light the depth relief so head turns read as volume.
    float e = 1.5 / 128.0;
    float dx = texture2D(uDepth, vImg + vec2(e, 0.0)).r - texture2D(uDepth, vImg - vec2(e, 0.0)).r;
    float dy = texture2D(uDepth, vImg + vec2(0.0, e)).r - texture2D(uDepth, vImg - vec2(0.0, e)).r;
    vec3 normal = normalize(vec3(-dx * uShade, dy * uShade, 1.0));
    float light = dot(normal, normalize(vec3(-0.45, 0.55, 1.0)));
    color *= mix(1.0, 0.86 + 0.2 * light, vHead);
    gl_FragColor = vec4(color, 1.0);
    #include <colorspace_fragment>
  }
`;

function smoothstep(edge0, edge1, value) {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function objectPositionFactor(value, fallback) {
  if (!value) return fallback;
  if (value === "left" || value === "top") return 0;
  if (value === "right" || value === "bottom") return 1;
  if (value === "center") return 0.5;
  if (value.endsWith("%")) return Number.parseFloat(value) / 100;
  return fallback;
}

function boxBlur(source, size, radius) {
  const temp = new Float32Array(source.length);
  const out = new Float32Array(source.length);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      let sum = 0;
      let count = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const xx = Math.min(size - 1, Math.max(0, x + k));
        sum += source[y * size + xx];
        count += 1;
      }
      temp[y * size + x] = sum / count;
    }
  }
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      let sum = 0;
      let count = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const yy = Math.min(size - 1, Math.max(0, y + k));
        sum += temp[yy * size + x];
        count += 1;
      }
      out[y * size + x] = sum / count;
    }
  }
  return out;
}

// Build an image-space depth map (R) and head-rotation weight (G) from MediaPipe's 478 points.
function buildDepthMap(features, imageAspect) {
  const size = DEPTH_MAP_SIZE;
  const { points, faceOval } = features;
  const cx = (faceOval.minX + faceOval.maxX) / 2;
  const cy = (faceOval.minY + faceOval.maxY) / 2;
  const rx = (faceOval.maxX - faceOval.minX) / 2;
  const ry = (faceOval.maxY - faceOval.minY) / 2;
  const sortedZ = points.map(({ z }) => z).sort((a, b) => a - b);
  const zNear = sortedZ[0];
  const zFar = sortedZ[Math.floor(sortedZ.length * 0.9)];
  const zRange = Math.max(1e-4, zFar - zNear);

  let depth = new Float32Array(size * size);
  let head = new Float32Array(size * size);
  for (let j = 0; j < size; j += 1) {
    const y = (j + 0.5) / size;
    for (let i = 0; i < size; i += 1) {
      const x = (i + 0.5) / size;
      const nx = (x - cx) / rx;
      const ny = (y - cy) / (y < cy ? ry * 1.22 : ry);
      const r = Math.hypot(nx, ny);
      const headWeight = 1 - smoothstep(0.95, 1.5, r);
      const dome = Math.sqrt(Math.max(0, 1 - Math.min(1, r) ** 2));
      const faceWeight = 1 - smoothstep(0.82, 1.08, r);

      let relief = 0;
      if (faceWeight > 0) {
        let weightSum = 0;
        let zSum = 0;
        for (let p = 0; p < points.length; p += 1) {
          const dxp = (points[p].x - x) * imageAspect;
          const dyp = points[p].y - y;
          const d2 = dxp * dxp + dyp * dyp + 2e-5;
          const w = 1 / (d2 * d2);
          weightSum += w;
          zSum += w * points[p].z;
        }
        relief = Math.min(1, Math.max(0, (zFar - zSum / weightSum) / zRange));
      }
      depth[j * size + i] = 0.55 * dome * headWeight + 0.45 * relief * faceWeight;
      head[j * size + i] = headWeight;
    }
  }
  depth = boxBlur(boxBlur(depth, size, 2), size, 1);
  head = boxBlur(head, size, 2);

  const data = new Uint8Array(size * size * 4);
  for (let k = 0; k < size * size; k += 1) {
    data[k * 4] = Math.round(Math.min(1, depth[k]) * 255);
    data[k * 4 + 1] = Math.round(Math.min(1, head[k]) * 255);
    data[k * 4 + 3] = 255;
  }
  return data;
}

function center(bounds) {
  return { x: (bounds.minX + bounds.maxX) / 2, y: (bounds.minY + bounds.maxY) / 2 };
}

function approach(current, target, amount) {
  Object.keys(target).forEach((key) => {
    current[key] += (target[key] - current[key]) * amount;
  });
}

export async function createPortraitThreeAnimator({ canvas, image, media, features }) {
  if (!features?.points?.length || !features.faceOval) throw new Error("3D portrait needs full face landmarks.");
  const THREE = await import(THREE_MODULE);
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "low-power" });
  if (!renderer.capabilities.isWebGL2) throw new Error("WebGL2 is unavailable.");
  renderer.setClearColor(0x000000, 0);

  const imageAspect = image.naturalWidth / image.naturalHeight;
  // Copy into a fixed-size canvas: sRGB mipmap generation is not supported on every GPU driver.
  const source = document.createElement("canvas");
  source.width = 1024;
  source.height = Math.round(1024 / imageAspect);
  const sourceContext = source.getContext("2d");
  sourceContext.imageSmoothingQuality = "high";
  sourceContext.drawImage(image, 0, 0, source.width, source.height);
  const map = new THREE.CanvasTexture(source);
  map.flipY = false;
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = THREE.ClampToEdgeWrapping;
  map.wrapT = THREE.ClampToEdgeWrapping;
  map.generateMipmaps = false;
  map.minFilter = THREE.LinearFilter;
  map.magFilter = THREE.LinearFilter;

  const depthTexture = new THREE.DataTexture(buildDepthMap(features, imageAspect), DEPTH_MAP_SIZE, DEPTH_MAP_SIZE);
  depthTexture.magFilter = THREE.LinearFilter;
  depthTexture.minFilter = THREE.LinearFilter;
  depthTexture.needsUpdate = true;

  const p = features.points;
  const mouthLeft = p[61];
  const mouthRight = p[291];
  const mouthCenter = { x: (p[13].x + p[14].x) / 2, y: (p[13].y + p[14].y) / 2 };
  const axis = { x: (mouthRight.x - mouthLeft.x) * imageAspect, y: mouthRight.y - mouthLeft.y };
  const axisLength = Math.hypot(axis.x, axis.y) || 1;
  const lipHeight = Math.max(0.006, Math.abs(p[17].y - p[0].y) / 2);
  const perp = { x: -axis.y / axisLength, y: axis.x / axisLength };
  const cornerMid = { x: ((mouthLeft.x + mouthRight.x) / 2 - mouthCenter.x) * imageAspect, y: (mouthLeft.y + mouthRight.y) / 2 - mouthCenter.y };
  const mouthCurve = cornerMid.x * perp.x + cornerMid.y * perp.y;
  const eyeUniform = (bounds) => {
    const c = center(bounds);
    return new THREE.Vector4(
      c.x,
      c.y,
      Math.max(0.01, ((bounds.maxX - bounds.minX) / 2) * imageAspect),
      Math.max(0.006, (bounds.maxY - bounds.minY) / 2),
    );
  };

  const uniforms = {
    uMap: { value: map },
    uDepth: { value: depthTexture },
    uImgMap: { value: new THREE.Vector4(1, 1, 0, 0) },
    uPlaneSize: { value: new THREE.Vector2(1, 1) },
    uOverscan: { value: OVERSCAN },
    uDepthScale: { value: 0.1 },
    uPivot: { value: new THREE.Vector3() },
    uHeadRot: { value: new THREE.Vector3() },
    uHeadOffset: { value: new THREE.Vector3() },
    uBreath: { value: 0 },
    uImgAspect: { value: imageAspect },
    uMouthCenter: { value: new THREE.Vector2(mouthCenter.x, mouthCenter.y) },
    uMouthAxis: { value: new THREE.Vector2(axis.x / axisLength, axis.y / axisLength) },
    uMouthSize: { value: new THREE.Vector2(axisLength / 2, lipHeight) },
    uMouthCurve: { value: mouthCurve },
    uMouth: { value: new THREE.Vector4() },
    uEyeL: { value: eyeUniform(features.leftEye) },
    uEyeR: { value: eyeUniform(features.rightEye) },
    uEyes: { value: new THREE.Vector4() },
    uBrow: { value: 0 },
    uShade: { value: 9 },
  };

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 20);
  camera.position.set(0, 0, 0.5 / Math.tan(THREE.MathUtils.degToRad(FOV / 2)));
  const geometry = new THREE.PlaneGeometry(1, 1, 180, 180);
  const material = new THREE.ShaderMaterial({ uniforms, vertexShader: VERTEX_SHADER, fragmentShader: FRAGMENT_SHADER });
  scene.add(new THREE.Mesh(geometry, material));

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mouth = { open: 0, wide: 0, round: 0 };
  const expression = { ...EXPRESSIONS.idle };
  const look = { x: 0, y: 0 };
  const pointer = { x: 0, y: 0 };
  let conversationState = "idle";
  let viseme = "rest";
  let frameRequest = 0;
  let lastFrameAt = performance.now();
  let lastRenderAt = 0;
  let nextBlinkAt = performance.now() + 2600;
  let blinkStartedAt = 0;
  let viewport = { width: 1, height: 1, aspect: 1, imgScaleX: 1, imgScaleY: 1, imgBiasX: 0, imgBiasY: 0 };

  function imageToWorld(x, y) {
    const vx = (x - viewport.imgBiasX) / viewport.imgScaleX;
    const vy = (y - viewport.imgBiasY) / viewport.imgScaleY;
    return { x: (vx - 0.5) * viewport.aspect, y: 0.5 - vy };
  }

  function resize() {
    const cssWidth = Math.max(1, media.clientWidth);
    const cssHeight = Math.max(1, media.clientHeight);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(cssWidth, cssHeight, false);
    camera.aspect = cssWidth / cssHeight;
    camera.updateProjectionMatrix();

    const scale = Math.max(cssWidth / image.naturalWidth, cssHeight / image.naturalHeight);
    const fittedWidth = image.naturalWidth * scale;
    const fittedHeight = image.naturalHeight * scale;
    const position = getComputedStyle(image).objectPosition.trim().split(/\s+/);
    const offsetX = (cssWidth - fittedWidth) * objectPositionFactor(position[0], 0.5);
    const offsetY = (cssHeight - fittedHeight) * objectPositionFactor(position[1], 0.5);
    viewport = {
      width: cssWidth,
      height: cssHeight,
      aspect: cssWidth / cssHeight,
      imgScaleX: cssWidth / fittedWidth,
      imgScaleY: cssHeight / fittedHeight,
      imgBiasX: -offsetX / fittedWidth,
      imgBiasY: -offsetY / fittedHeight,
    };
    uniforms.uImgMap.value.set(viewport.imgScaleX, viewport.imgScaleY, viewport.imgBiasX, viewport.imgBiasY);
    uniforms.uPlaneSize.value.set(viewport.aspect * OVERSCAN, OVERSCAN);

    const oval = features.faceOval;
    const faceWidthWorld = ((oval.maxX - oval.minX) / viewport.imgScaleX) * viewport.aspect;
    uniforms.uDepthScale.value = faceWidthWorld * 0.42;
    const pivot = imageToWorld((oval.minX + oval.maxX) / 2, oval.maxY);
    uniforms.uPivot.value.set(pivot.x, pivot.y, -faceWidthWorld * 0.35);
    render();
  }

  function blinkAmount(now) {
    if (reducedMotion) return 0;
    if (!blinkStartedAt && now >= nextBlinkAt) blinkStartedAt = now;
    if (!blinkStartedAt) return 0;
    const elapsed = now - blinkStartedAt;
    const duration = 180;
    if (elapsed >= duration) {
      blinkStartedAt = 0;
      // Occasionally double-blink, otherwise wait a natural few seconds.
      nextBlinkAt = now + (Math.random() < 0.18 ? 220 : 3200 + Math.random() * 3200);
      return 0;
    }
    return Math.pow(Math.sin((elapsed / duration) * Math.PI), 1.2);
  }

  function update(now, elapsed) {
    const blend = (ms) => 1 - Math.exp(-elapsed / ms);
    const target = conversationState === "speaking" && !reducedMotion ? VISEMES[viseme] || VISEMES.open : VISEMES.rest;
    approach(mouth, target, blend(70));
    approach(expression, EXPRESSIONS[conversationState] || EXPRESSIONS.idle, blend(260));
    approach(look, reducedMotion ? { x: 0, y: 0 } : pointer, blend(420));

    const t = now / 1000;
    const sway = reducedMotion ? 0 : 1;
    const speakingBob = conversationState === "speaking" ? mouth.open * 0.022 + Math.sin(t * 5.3) * 0.006 : 0;
    const yaw = expression.yaw + look.x * 0.11 + sway * (Math.sin(t * 0.45) * 0.03 + Math.sin(t * 0.21) * 0.015);
    const pitch = expression.pitch + look.y * 0.07 + speakingBob + sway * Math.sin(t * 0.37 + 1.2) * 0.018;
    const roll = expression.roll + sway * Math.sin(t * 0.29 + 0.4) * 0.012;
    uniforms.uHeadRot.value.set(pitch, yaw, roll);
    uniforms.uHeadOffset.value.set(0, 0, expression.lean);
    uniforms.uBreath.value = sway * Math.sin(t * 1.35) * 0.0035;

    const blink = Math.max(blinkAmount(now), expression.squint);
    const idleGaze = sway * Math.sin(t * 0.38) * 0.18;
    uniforms.uEyes.value.set(blink, expression.gazeX + look.x * 0.6 + idleGaze, expression.gazeY + look.y * 0.4, 0);
    const speechBrow = conversationState === "speaking" ? mouth.open * 0.2 : 0;
    uniforms.uBrow.value = expression.brow + speechBrow;
    uniforms.uMouth.value.set(mouth.open, mouth.wide, mouth.round, expression.smile);
    return conversationState !== "idle" || blinkStartedAt > 0;
  }

  function render() {
    renderer.render(scene, camera);
    canvas.dataset.renderer = "three";
    canvas.dataset.viseme = viseme;
  }

  function tick(now) {
    const elapsed = Math.min(80, Math.max(0, now - lastFrameAt));
    lastFrameAt = now;
    const active = update(now, elapsed);
    const interval = active ? 16 : 33;
    if (!document.hidden && now - lastRenderAt >= interval) {
      render();
      lastRenderAt = now;
    }
    frameRequest = window.requestAnimationFrame(tick);
  }

  function onPointerMove(event) {
    const rect = media.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const x = (event.clientX - (rect.left + rect.width / 2)) / Math.max(rect.width, window.innerWidth * 0.5);
    const y = (event.clientY - (rect.top + rect.height * 0.4)) / Math.max(rect.height, window.innerHeight * 0.5);
    pointer.x = Math.max(-1, Math.min(1, x));
    pointer.y = Math.max(-1, Math.min(1, y));
  }

  function onPointerLeave() {
    pointer.x = 0;
    pointer.y = 0;
  }

  return {
    start() {
      resize();
      window.addEventListener("pointermove", onPointerMove, { passive: true });
      document.documentElement.addEventListener("pointerleave", onPointerLeave);
      frameRequest = window.requestAnimationFrame(tick);
    },
    resize,
    setState(next) {
      conversationState = EXPRESSIONS[next] ? next : "idle";
    },
    setViseme(next) {
      viseme = VISEMES[next] ? next : "open";
    },
    destroy() {
      window.cancelAnimationFrame(frameRequest);
      window.removeEventListener("pointermove", onPointerMove);
      document.documentElement.removeEventListener("pointerleave", onPointerLeave);
      geometry.dispose();
      material.dispose();
      map.dispose();
      depthTexture.dispose();
      renderer.dispose();
    },
  };
}
