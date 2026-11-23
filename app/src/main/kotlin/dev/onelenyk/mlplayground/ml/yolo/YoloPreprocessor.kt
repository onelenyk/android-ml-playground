package dev.onelenyk.mlplayground.ml.yolo

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect

enum class TensorLayout { NHWC, NCHW }

/** How the original image maps into the square model input (letterbox). */
data class Letterbox(val scale: Float, val padX: Float, val padY: Float, val srcW: Int, val srcH: Int)

/**
 * Shared preprocessing for YOLO models (used by both LiteRT and ONNX engines so the comparison
 * is apples-to-apples): letterbox resize with gray (114) padding, RGB, float 0..1.
 */
class YoloPreprocessor(val inputSize: Int, private val layout: TensorLayout) {
  private val canvasBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
  private val canvas = Canvas(canvasBitmap)
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
  private val pixels = IntArray(inputSize * inputSize)

  /** Input tensor data reused across frames (NHWC or NCHW, float 0..1). */
  val input = FloatArray(3 * inputSize * inputSize)

  fun process(src: Bitmap): Letterbox {
    val scale = minOf(inputSize.toFloat() / src.width, inputSize.toFloat() / src.height)
    val newW = Math.round(src.width * scale)
    val newH = Math.round(src.height * scale)
    val padX = (inputSize - newW) / 2f
    val padY = (inputSize - newH) / 2f

    canvas.drawColor(Color.rgb(114, 114, 114))
    canvas.drawBitmap(
      src, null,
      Rect(padX.toInt(), padY.toInt(), padX.toInt() + newW, padY.toInt() + newH), paint,
    )
    canvasBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

    val inv = 1f / 255f
    val area = inputSize * inputSize
    if (layout == TensorLayout.NHWC) {
      var j = 0
      for (i in 0 until area) {
        val p = pixels[i]
        input[j++] = ((p shr 16) and 0xFF) * inv
        input[j++] = ((p shr 8) and 0xFF) * inv
        input[j++] = (p and 0xFF) * inv
      }
    } else {
      for (i in 0 until area) {
        val p = pixels[i]
        input[i] = ((p shr 16) and 0xFF) * inv
        input[area + i] = ((p shr 8) and 0xFF) * inv
        input[2 * area + i] = (p and 0xFF) * inv
      }
    }
    return Letterbox(scale, padX, padY, src.width, src.height)
  }
}
