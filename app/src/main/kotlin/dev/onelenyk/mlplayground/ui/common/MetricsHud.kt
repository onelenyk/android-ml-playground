package dev.onelenyk.mlplayground.ui.common

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.DetectionResult
import dev.onelenyk.mlplayground.ml.core.EngineType
import java.util.Locale

@Composable
fun MetricsHud(
  engineType: EngineType,
  requestedDelegate: DelegateType,
  activeDelegate: DelegateType,
  modelInfo: String,
  fps: Double,
  result: DetectionResult?,
  modifier: Modifier = Modifier,
) {
  val isFallback = requestedDelegate == DelegateType.GPU && activeDelegate == DelegateType.CPU
  Surface(
    modifier = modifier.fillMaxWidth(),
    color = Color(0xBB101418),
    shape = RoundedCornerShape(14.dp),
  ) {
    Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
      ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
          Text(
            text = engineType.title,
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
            color = Color.White,
          )
          Spacer(Modifier.width(8.dp))
          val chipColor = when {
            isFallback -> Color(0xFFFFB300)
            activeDelegate == DelegateType.GPU -> Color(0xFF00E676)
            else -> Color(0xFF40C4FF)
          }
          Text(
            text = if (isFallback) "GPU → CPU fallback" else activeDelegate.name,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold,
            color = Color.Black,
            modifier = Modifier
              .background(chipColor, RoundedCornerShape(6.dp))
              .padding(horizontal = 6.dp, vertical = 2.dp),
          )
        }
        Text(
          text = String.format(Locale.US, "%.1f FPS", fps),
          style = MaterialTheme.typography.titleSmall,
          fontFamily = FontFamily.Monospace,
          fontWeight = FontWeight.Bold,
          color = Color(0xFF76FF03),
        )
      }

      if (result != null) {
        Text(
          text = String.format(
            Locale.US,
            "Всього: %.1f мс  |  Pre %.1f · Inf %.1f · Post %.1f мс  |  Об'єктів: %d",
            result.totalMs,
            result.preprocessMs,
            result.inferenceMs,
            result.postprocessMs,
            result.detections.size,
          ),
          style = MaterialTheme.typography.labelSmall,
          fontFamily = FontFamily.Monospace,
          color = Color(0xFFE0E6ED),
        )
      } else {
        Text(
          text = "Очікування кадру…",
          style = MaterialTheme.typography.labelSmall,
          color = Color(0xFFB0BAC5),
        )
      }

      if (modelInfo.isNotBlank() && modelInfo != "-") {
        Text(
          text = modelInfo,
          style = MaterialTheme.typography.labelSmall,
          color = Color(0xFF9AA5B1),
          maxLines = 1,
          overflow = TextOverflow.Ellipsis,
        )
      }
    }
  }
}
