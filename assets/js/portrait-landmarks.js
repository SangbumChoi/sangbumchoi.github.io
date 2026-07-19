const MEDIAPIPE_VERSION = "0.10.35";
const MEDIAPIPE_MODULE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/+esm`;
const MEDIAPIPE_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const FACE_LANDMARK_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

function boundsForConnections(landmarks, connections) {
  const indices = [...new Set(connections.flatMap(({ start, end }) => [start, end]))];
  const points = indices.map((index) => landmarks[index]).filter(Boolean);
  if (!points.length) return null;
  const xs = points.map(({ x }) => x);
  const ys = points.map(({ y }) => y);
  return {
    minX: Math.min(...xs),
    minY: Math.min(...ys),
    maxX: Math.max(...xs),
    maxY: Math.max(...ys),
  };
}

function sampleSkinTone(image, eyeBounds) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return "rgb(240 177 132)";
  context.drawImage(image, 0, 0, canvas.width, canvas.height);

  const minX = Math.max(0, Math.floor((eyeBounds.minX - 0.018) * canvas.width));
  const maxX = Math.min(canvas.width, Math.ceil((eyeBounds.maxX + 0.018) * canvas.width));
  const minY = Math.max(0, Math.floor((eyeBounds.minY - 0.018) * canvas.height));
  const maxY = Math.min(canvas.height, Math.ceil((eyeBounds.maxY + 0.018) * canvas.height));
  const pixels = context.getImageData(minX, minY, maxX - minX, maxY - minY).data;
  let red = 0;
  let green = 0;
  let blue = 0;
  let count = 0;

  for (let index = 0; index < pixels.length; index += 4) {
    const r = pixels[index];
    const g = pixels[index + 1];
    const b = pixels[index + 2];
    const luminance = r * 0.299 + g * 0.587 + b * 0.114;
    if (luminance < 120 || r < g || g < b) continue;
    red += r;
    green += g;
    blue += b;
    count += 1;
  }

  if (!count) return "rgb(240 177 132)";
  return `rgb(${Math.round(red / count)} ${Math.round(green / count)} ${Math.round(blue / count)})`;
}

export async function detectPortraitFeatures(image) {
  await image.decode();
  const { FaceLandmarker, FilesetResolver } = await import(MEDIAPIPE_MODULE);
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: FACE_LANDMARK_MODEL,
      delegate: "CPU",
    },
    minFaceDetectionConfidence: 0.1,
    minFacePresenceConfidence: 0.1,
    numFaces: 1,
    runningMode: "IMAGE",
  });

  try {
    const landmarks = landmarker.detect(image).faceLandmarks[0];
    if (!landmarks?.length) throw new Error("No face landmarks detected in the portrait.");
    const leftEye = boundsForConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
    const rightEye = boundsForConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
    const lips = boundsForConnections(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS);
    if (!leftEye || !rightEye || !lips) throw new Error("Eye or lip landmarks were incomplete.");

    return {
      source: "mediapipe-478",
      leftEye,
      rightEye,
      lips,
      skinColor: sampleSkinTone(image, {
        minX: Math.min(leftEye.minX, rightEye.minX),
        minY: Math.min(leftEye.minY, rightEye.minY),
        maxX: Math.max(leftEye.maxX, rightEye.maxX),
        maxY: Math.max(leftEye.maxY, rightEye.maxY),
      }),
    };
  } finally {
    landmarker.close();
  }
}
