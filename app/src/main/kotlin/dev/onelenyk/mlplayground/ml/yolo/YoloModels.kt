package dev.onelenyk.mlplayground.ml.yolo

import android.content.Context
import dev.onelenyk.mlplayground.ml.core.Assets

/**
 * Resolves the YOLO model asset for a given runtime/input size and infers the output layout
 * from the number of output floats, so exports with/without baked-in argmax both work.
 */
object YoloModels {
  const val LABELS = "yoloe_pf_labels.txt"
  val SUPPORTED_SIZES = listOf(320, 640)

  fun assetName(context: Context, ext: String, inputSize: Int): String? =
    listOf("yoloe_pf_$inputSize.$ext", if (inputSize == 640) "yoloe_pf.$ext" else null)
      .filterNotNull()
      .firstOrNull { Assets.exists(context, it) }

  fun anchors(inputSize: Int): Int =
    listOf(8, 16, 32).sumOf { s -> (inputSize / s) * (inputSize / s) }

  /**
   * @param coordsNormalized whether this runtime's export emits 0..1 box coords
   *   (Ultralytics TFLite) or input-pixel coords (ONNX).
   */
  fun inferSpec(outputSize: Int, inputSize: Int, numClasses: Int, coordsNormalized: Boolean): YoloOutputSpec {
    val n = anchors(inputSize)
    return when {
      outputSize == n * 6 -> YoloOutputSpec.Compact(n, coordsNormalized)
      outputSize % n == 0 && outputSize / n >= 4 + numClasses ->
        YoloOutputSpec.Raw(n, numClasses, coordsNormalized, channelsFirst = true)
      else -> error("Unexpected YOLO output size $outputSize for input $inputSize (anchors=$n, classes=$numClasses)")
    }
  }

  @Volatile private var cachedLabels: List<String>? = null

  fun labels(context: Context): List<String> =
    cachedLabels ?: Assets.readLines(context, LABELS).filter { it.isNotEmpty() }.also { cachedLabels = it }
}
