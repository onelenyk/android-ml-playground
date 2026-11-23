package dev.onelenyk.mlplayground.ml.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.Detection
import dev.onelenyk.mlplayground.ml.core.DetectionResult
import dev.onelenyk.mlplayground.ml.core.EngineType
import dev.onelenyk.mlplayground.ml.core.NormBox
import dev.onelenyk.mlplayground.ml.core.ObjectDetectorEngine
import dev.onelenyk.mlplayground.ml.core.nowMs

/**
 * Option 1: MediaPipe Tasks Vision. A "black box": resize, normalization, inference and NMS
 * all happen inside the native task. We therefore can only measure prep (Bitmap -> MPImage)
 * and the whole detect() call as "inference".
 */
class MediaPipeEngine(private val context: Context) : ObjectDetectorEngine {
  override val type = EngineType.MEDIAPIPE
  override var activeDelegate = DelegateType.CPU
    private set
  override val modelInfo = "$MODEL (COCO 80)"

  private var detector: ObjectDetector? = null
  private var currentThreshold = -1f
  private var requestedDelegate = DelegateType.CPU

  override fun initialize(delegate: DelegateType) {
    requestedDelegate = delegate
    build(delegate, DEFAULT_THRESHOLD)
  }

  private fun build(delegate: DelegateType, threshold: Float) {
    detector?.close()
    detector = try {
      create(delegate, threshold).also { activeDelegate = delegate }
    } catch (t: Throwable) {
      if (delegate == DelegateType.GPU) {
        Log.w(TAG, "GPU init failed, falling back to CPU", t)
        create(DelegateType.CPU, threshold).also { activeDelegate = DelegateType.CPU }
      } else throw t
    }
    currentThreshold = threshold
  }

  private fun create(delegate: DelegateType, threshold: Float): ObjectDetector {
    val base = BaseOptions.builder()
      .setModelAssetPath(MODEL)
      .setDelegate(if (delegate == DelegateType.GPU) Delegate.GPU else Delegate.CPU)
      .build()
    val options = ObjectDetector.ObjectDetectorOptions.builder()
      .setBaseOptions(base)
      .setRunningMode(RunningMode.IMAGE)
      .setScoreThreshold(threshold)
      .setMaxResults(MAX_RESULTS)
      .build()
    return ObjectDetector.createFromOptions(context, options)
  }

  override fun detect(bitmap: Bitmap, threshold: Float): DetectionResult {
    // Score threshold is baked into options -> rebuild only when it changes noticeably.
    if (kotlin.math.abs(threshold - currentThreshold) > 0.01f) build(activeDelegate, threshold)
    val det = detector ?: error("Not initialized")

    val t0 = nowMs()
    val image = BitmapImageBuilder(bitmap).build()
    val t1 = nowMs()
    val result = det.detect(image)
    val t2 = nowMs()
    val w = bitmap.width.toFloat()
    val h = bitmap.height.toFloat()
    val detections = result.detections().mapNotNull { d ->
      val c = d.categories().firstOrNull() ?: return@mapNotNull null
      val b = d.boundingBox()
      Detection(NormBox(b.left / w, b.top / h, b.right / w, b.bottom / h), c.categoryName(), c.score())
    }
    val t3 = nowMs()
    return DetectionResult(detections, t1 - t0, t2 - t1, t3 - t2, bitmap.width, bitmap.height)
  }

  override fun close() {
    detector?.close()
    detector = null
  }

  companion object {
    private const val TAG = "MediaPipeEngine"
    const val MODEL = "efficientdet_lite0.tflite"
    private const val DEFAULT_THRESHOLD = 0.4f
    private const val MAX_RESULTS = 25
  }
}
