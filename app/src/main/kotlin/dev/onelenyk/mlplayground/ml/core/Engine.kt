package dev.onelenyk.mlplayground.ml.core

import android.graphics.Bitmap

enum class DelegateType { CPU, GPU }

enum class EngineType(val title: String, val subtitle: String) {
  MEDIAPIPE("MediaPipe", "EfficientDet-Lite0 · COCO 80"),
  LITERT("LiteRT", "YOLOE prompt-free"),
  ONNX("ONNX Runtime", "YOLOE prompt-free"),
}

/** Normalized 0..1 bounding box in upright input bitmap coordinates. */
data class NormBox(val left: Float, val top: Float, val right: Float, val bottom: Float)

/** [box] is normalized (0..1) in coordinates of the upright input bitmap. */
data class Detection(val box: NormBox, val label: String, val score: Float)

data class DetectionResult(
  val detections: List<Detection>,
  val preprocessMs: Double,
  val inferenceMs: Double,
  val postprocessMs: Double,
  val imageWidth: Int,
  val imageHeight: Int,
) {
  val totalMs: Double get() = preprocessMs + inferenceMs + postprocessMs
}

/** Common contract implemented by every ML runtime integration (Strategy pattern). */
interface ObjectDetectorEngine : AutoCloseable {
  val type: EngineType
  /** Delegate actually in use (may differ from requested if GPU init failed). */
  val activeDelegate: DelegateType
  /** Short description of the loaded model, e.g. "yoloe_pf_320.tflite (4585 classes)". */
  val modelInfo: String

  /** Heavy: loads model. Must be called off the main thread. */
  fun initialize(delegate: DelegateType)

  /** [bitmap] must be upright (rotation already applied), ARGB_8888. */
  fun detect(bitmap: Bitmap, threshold: Float): DetectionResult
}

fun nowMs(): Double = System.nanoTime() / 1_000_000.0
