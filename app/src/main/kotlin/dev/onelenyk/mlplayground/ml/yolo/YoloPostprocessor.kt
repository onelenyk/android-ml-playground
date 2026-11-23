package dev.onelenyk.mlplayground.ml.yolo

import dev.onelenyk.mlplayground.ml.core.Detection
import dev.onelenyk.mlplayground.ml.core.NormBox

/**
 * Describes the YOLO output tensor so the same postprocessor works for every export variant.
 *
 * - [Compact]: rows of (cx, cy, w, h, score, classIndex) — argmax already done inside the graph.
 * - [Raw]: channels-first [4 + numClasses (+extra), numAnchors] — classic Ultralytics head.
 */
sealed interface YoloOutputSpec {
  val coordsNormalized: Boolean

  data class Compact(val numAnchors: Int, override val coordsNormalized: Boolean, val rowSize: Int = 6) :
    YoloOutputSpec

  data class Raw(
    val numAnchors: Int,
    val numClasses: Int,
    override val coordsNormalized: Boolean,
    val channelsFirst: Boolean = true,
  ) : YoloOutputSpec
}

private class Candidate(
  val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float, val cls: Int,
) {
  val area get() = (x2 - x1).coerceAtLeast(0f) * (y2 - y1).coerceAtLeast(0f)
}

/** Shared postprocessing: decode -> threshold -> NMS -> undo letterbox. */
class YoloPostprocessor(
  private val spec: YoloOutputSpec,
  private val labels: List<String>,
  private val inputSize: Int,
  private val iouThreshold: Float = 0.5f,
  private val maxDetections: Int = 50,
  private val classAgnosticNms: Boolean = true,
) {
  fun process(out: FloatArray, threshold: Float, lb: Letterbox): List<Detection> {
    val coordScale = if (spec.coordsNormalized) inputSize.toFloat() else 1f
    val candidates = ArrayList<Candidate>(64)

    when (spec) {
      is YoloOutputSpec.Compact -> {
        val r = spec.rowSize
        for (i in 0 until spec.numAnchors) {
          val o = i * r
          val score = out[o + 4]
          if (score < threshold) continue
          candidates += box(out[o], out[o + 1], out[o + 2], out[o + 3], coordScale, score, out[o + 5].toInt())
        }
      }
      is YoloOutputSpec.Raw -> {
        val n = spec.numAnchors
        val nc = spec.numClasses
        for (i in 0 until n) {
          var best = -1f
          var bestC = -1
          for (c in 0 until nc) {
            val s = if (spec.channelsFirst) out[(4 + c) * n + i] else out[i * (4 + nc) + 4 + c]
            if (s > best) { best = s; bestC = c }
          }
          if (best < threshold) continue
          if (spec.channelsFirst) {
            candidates += box(out[i], out[n + i], out[2 * n + i], out[3 * n + i], coordScale, best, bestC)
          } else {
            val o = i * (4 + nc)
            candidates += box(out[o], out[o + 1], out[o + 2], out[o + 3], coordScale, best, bestC)
          }
        }
      }
    }

    return nms(candidates).map { c -> toDetection(c, lb) }
  }

  private fun box(cx: Float, cy: Float, w: Float, h: Float, s: Float, score: Float, cls: Int): Candidate {
    val x = cx * s; val y = cy * s; val hw = w * s / 2f; val hh = h * s / 2f
    return Candidate(x - hw, y - hh, x + hw, y + hh, score, cls)
  }

  private fun nms(input: List<Candidate>): List<Candidate> {
    val sorted = input.sortedByDescending { it.score }
    val kept = ArrayList<Candidate>()
    val removed = BooleanArray(sorted.size)
    for (i in sorted.indices) {
      if (removed[i]) continue
      val a = sorted[i]
      kept += a
      if (kept.size >= maxDetections) break
      for (j in i + 1 until sorted.size) {
        if (!removed[j] && (classAgnosticNms || sorted[j].cls == a.cls) && iou(a, sorted[j]) > iouThreshold) removed[j] = true
      }
    }
    return kept
  }

  private fun iou(a: Candidate, b: Candidate): Float {
    val ix = (minOf(a.x2, b.x2) - maxOf(a.x1, b.x1)).coerceAtLeast(0f)
    val iy = (minOf(a.y2, b.y2) - maxOf(a.y1, b.y1)).coerceAtLeast(0f)
    val inter = ix * iy
    val union = a.area + b.area - inter
    return if (union <= 0f) 0f else inter / union
  }

  /** Model-input pixels -> normalized coords of the original (upright) bitmap. */
  private fun toDetection(c: Candidate, lb: Letterbox): Detection {
    fun nx(x: Float) = ((x - lb.padX) / lb.scale / lb.srcW).coerceIn(0f, 1f)
    fun ny(y: Float) = ((y - lb.padY) / lb.scale / lb.srcH).coerceIn(0f, 1f)
    return Detection(NormBox(nx(c.x1), ny(c.y1), nx(c.x2), ny(c.y2)), labels.getOrElse(c.cls) { "#${c.cls}" }, c.score)
  }
}
