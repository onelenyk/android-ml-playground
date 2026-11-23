package dev.onelenyk.mlplayground.ui.common

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.onelenyk.mlplayground.ml.core.Detection
import kotlin.math.abs

private val BOX_PALETTE = listOf(
  Color(0xFF00E5FF),
  Color(0xFFFF6D00),
  Color(0xFF76FF03),
  Color(0xFFFF4081),
  Color(0xFFFFEA00),
  Color(0xFF7C4DFF),
  Color(0xFF1DE9B6),
  Color(0xFFFF5252),
  Color(0xFF40C4FF),
  Color(0xFFEEFF41),
)

private fun colorForLabel(label: String): Color =
  BOX_PALETTE[abs(label.hashCode()) % BOX_PALETTE.size]

/**
 * Draws bounding boxes and confidence pills aligned with either a `PreviewView` using
 * `ScaleType.FILL_CENTER` ([fillCenter] = true) or an `Image` with `ContentScale.Fit` ([fillCenter] = false).
 */
@Composable
fun DetectionOverlay(
  detections: List<Detection>,
  imageWidth: Int,
  imageHeight: Int,
  modifier: Modifier = Modifier,
  highlightQuery: String = "",
  fillCenter: Boolean = true,
) {
  val textMeasurer = rememberTextMeasurer()
  val labelStyle = TextStyle(color = Color.Black, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
  val query = highlightQuery.trim()

  Canvas(modifier = modifier) {
    if (imageWidth <= 0 || imageHeight <= 0 || detections.isEmpty()) return@Canvas

    val scale = if (fillCenter) {
      maxOf(size.width / imageWidth, size.height / imageHeight)
    } else {
      minOf(size.width / imageWidth, size.height / imageHeight)
    }
    val drawnW = imageWidth * scale
    val drawnH = imageHeight * scale
    val ox = (size.width - drawnW) / 2f
    val oy = (size.height - drawnH) / 2f
    val corner = CornerRadius(6.dp.toPx())
    val padH = 6.dp.toPx()
    val padV = 2.dp.toPx()

    for (det in detections) {
      val matches = query.isNotEmpty() && det.label.contains(query, ignoreCase = true)
      val alpha = when {
        query.isEmpty() -> 0.95f
        matches -> 1.0f
        else -> 0.35f
      }
      val baseColor = if (matches) Color(0xFF00E676) else colorForLabel(det.label)
      val strokePx = when {
        matches -> 4.dp.toPx()
        query.isNotEmpty() -> 1.5.dp.toPx()
        else -> 2.5.dp.toPx()
      }

      val left = ox + det.box.left * drawnW
      val top = oy + det.box.top * drawnH
      val right = ox + det.box.right * drawnW
      val bottom = oy + det.box.bottom * drawnH
      val boxW = (right - left).coerceAtLeast(1f)
      val boxH = (bottom - top).coerceAtLeast(1f)

      drawRoundRect(
        color = baseColor.copy(alpha = alpha),
        topLeft = Offset(left, top),
        size = Size(boxW, boxH),
        cornerRadius = corner,
        style = Stroke(width = strokePx),
      )

      val text = "${det.label} ${(det.score * 100).toInt()}%"
      val measured = textMeasurer.measure(text, style = labelStyle)
      val pillW = measured.size.width + padH * 2
      val pillH = measured.size.height + padV * 2
      val pillLeft = left.coerceIn(0f, (size.width - pillW).coerceAtLeast(0f))
      val pillTop = (top - pillH).coerceAtLeast(0f)

      drawRoundRect(
        color = baseColor.copy(alpha = alpha),
        topLeft = Offset(pillLeft, pillTop),
        size = Size(pillW, pillH),
        cornerRadius = CornerRadius(4.dp.toPx()),
      )
      drawText(
        textLayoutResult = measured,
        color = Color.Black.copy(alpha = alpha),
        topLeft = Offset(pillLeft + padH, pillTop + padV),
      )
    }
  }
}
