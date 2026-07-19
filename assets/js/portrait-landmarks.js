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
    };
  } finally {
    landmarker.close();
  }
}
