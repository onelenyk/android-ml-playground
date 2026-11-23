package dev.onelenyk.mlplayground.ui.benchmark

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.onelenyk.mlplayground.camera.CameraPreview
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.EngineType
import dev.onelenyk.mlplayground.ml.yolo.YoloModels
import dev.onelenyk.mlplayground.ui.common.DetectionOverlay
import java.util.Locale

@Composable
fun BenchmarkScreen(
  viewModel: BenchmarkViewModel,
  isActive: Boolean,
  modifier: Modifier = Modifier,
) {
  val state by viewModel.uiState.collectAsStateWithLifecycle()
  val context = LocalContext.current

  Column(
    modifier = modifier
      .fillMaxSize()
      .verticalScroll(rememberScrollState())
      .padding(12.dp),
    verticalArrangement = Arrangement.spacedBy(10.dp),
  ) {
    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
      BenchmarkSubMode.entries.forEachIndexed { idx, mode ->
        SegmentedButton(
          selected = state.subMode == mode,
          onClick = { viewModel.setSubMode(mode) },
          shape = SegmentedButtonDefaults.itemShape(idx, BenchmarkSubMode.entries.size),
          label = { Text(mode.label) },
        )
      }
    }

    // Frame capture card
    CaptureFrameCard(
      capturedBitmap = state.capturedBitmap,
      isActive = isActive,
      isRunning = state.isRunning,
      onLiveFrame = viewModel::onLiveFrame,
      onCapture = viewModel::captureFrame,
      onRetake = viewModel::retakeFrame,
    )

    // Shared config chips
    Row(
      modifier = Modifier
        .fillMaxWidth()
        .horizontalScroll(rememberScrollState()),
      horizontalArrangement = Arrangement.spacedBy(8.dp),
      verticalAlignment = Alignment.CenterVertically,
    ) {
      Text("YOLO:", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
      YoloModels.SUPPORTED_SIZES.forEach { sz ->
        FilterChip(
          selected = state.inputSize == sz,
          onClick = { if (!state.isRunning) viewModel.setInputSize(sz) },
          label = { Text("${sz}px") },
        )
      }

      if (state.subMode == BenchmarkSubMode.SAME_FRAME) {
        Text("Делегат:", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
        DelegateType.entries.forEach { del ->
          FilterChip(
            selected = state.sameFrameDelegate == del,
            onClick = { if (!state.isRunning) viewModel.setSameFrameDelegate(del) },
            label = { Text(del.name) },
          )
        }
      } else {
        Text("Прогонів:", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
        listOf(20, 50, 100).forEach { count ->
          FilterChip(
            selected = state.autoRunCount == count,
            onClick = { if (!state.isRunning) viewModel.setAutoRunCount(count) },
            label = { Text("$count") },
          )
        }
      }
    }

    if (state.isRunning) {
      Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        LinearProgressIndicator(
          progress = { state.progressFraction.coerceIn(0f, 1f) },
          modifier = Modifier.fillMaxWidth(),
        )
        Row(
          modifier = Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically,
        ) {
          Text(
            text = state.progressText,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.weight(1f),
          )
          OutlinedButton(onClick = viewModel::cancelRun) {
            Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(4.dp))
            Text("Стоп")
          }
        }
      }
    }

    when (state.subMode) {
      BenchmarkSubMode.SAME_FRAME -> {
        Button(
          onClick = viewModel::runSameFrameCompare,
          enabled = state.capturedBitmap != null && !state.isRunning,
          modifier = Modifier.fillMaxWidth(),
        ) {
          Icon(Icons.Default.PlayArrow, contentDescription = null)
          Spacer(Modifier.width(6.dp))
          Text("Порівняти всі 3 на цьому кадрі")
        }

        if (state.sameFrameResults.isNotEmpty()) {
          SameFrameSection(
            bitmap = state.capturedBitmap,
            entries = state.sameFrameResults,
            selectedEngine = state.selectedPreviewEngine,
            onSelectEngine = viewModel::selectPreviewEngine,
          )
        }
      }

      BenchmarkSubMode.AUTO_RUN -> {
        Button(
          onClick = viewModel::runAutoBenchmark,
          enabled = state.capturedBitmap != null && !state.isRunning,
          modifier = Modifier.fillMaxWidth(),
        ) {
          Icon(Icons.Default.PlayArrow, contentDescription = null)
          Spacer(Modifier.width(6.dp))
          Text("Запустити автопрогін (всі 6 конфігурацій)")
        }

        if (state.autoResults.isNotEmpty()) {
          AutoBenchmarkTable(state.autoResults)
          ThrottlingChartCard(state.autoResults)

          Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
          ) {
            OutlinedButton(
              onClick = { viewModel.buildShareIntent(context, ExportFormat.CSV)?.let(context::startActivity) },
              modifier = Modifier.weight(1f),
            ) {
              Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(16.dp))
              Spacer(Modifier.width(6.dp))
              Text("Експорт CSV")
            }
            OutlinedButton(
              onClick = { viewModel.buildShareIntent(context, ExportFormat.JSON)?.let(context::startActivity) },
              modifier = Modifier.weight(1f),
            ) {
              Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(16.dp))
              Spacer(Modifier.width(6.dp))
              Text("Експорт JSON")
            }
          }

          state.savedReportPath?.let { path ->
            Text(
              text = "Збережено: $path",
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
          }
        }
      }
    }
  }
}

@Composable
private fun CaptureFrameCard(
  capturedBitmap: Bitmap?,
  isActive: Boolean,
  isRunning: Boolean,
  onLiveFrame: (Bitmap) -> Unit,
  onCapture: () -> Unit,
  onRetake: () -> Unit,
) {
  Card(modifier = Modifier.fillMaxWidth()) {
    if (capturedBitmap == null) {
      Column(modifier = Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Box(
          modifier = Modifier
            .fillMaxWidth()
            .height(210.dp)
            .clip(RoundedCornerShape(10.dp)),
        ) {
          if (isActive) {
            CameraPreview(
              modifier = Modifier.fillMaxSize(),
              analysisEnabled = true,
              onFrame = onLiveFrame,
            )
          }
        }
        Button(onClick = onCapture, modifier = Modifier.fillMaxWidth()) {
          Icon(Icons.Default.CameraAlt, contentDescription = null)
          Spacer(Modifier.width(6.dp))
          Text("Зняти тестовий кадр")
        }
      }
    } else {
      Row(
        modifier = Modifier
          .fillMaxWidth()
          .padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
      ) {
        Image(
          bitmap = capturedBitmap.asImageBitmap(),
          contentDescription = "Тестовий кадр",
          contentScale = ContentScale.Crop,
          modifier = Modifier
            .size(72.dp)
            .clip(RoundedCornerShape(8.dp)),
        )
        Column(modifier = Modifier.weight(1f)) {
          Text("Кадр зафіксовано", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
          Text(
            "${capturedBitmap.width}×${capturedBitmap.height} px",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
          )
        }
        OutlinedButton(onClick = onRetake, enabled = !isRunning) {
          Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
          Spacer(Modifier.width(4.dp))
          Text("Перезняти")
        }
      }
    }
  }
}

@Composable
private fun SameFrameSection(
  bitmap: Bitmap?,
  entries: List<SameFrameEntry>,
  selectedEngine: EngineType,
  onSelectEngine: (EngineType) -> Unit,
) {
  val selected = entries.firstOrNull { it.engineType == selectedEngine } ?: entries.first()

  Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      entries.forEach { e ->
        FilterChip(
          selected = e.engineType == selected.engineType,
          onClick = { onSelectEngine(e.engineType) },
          label = { Text(e.engineType.title) },
        )
      }
    }

    if (bitmap != null) {
      Card(modifier = Modifier.fillMaxWidth()) {
        Box(
          modifier = Modifier
            .fillMaxWidth()
            .height(260.dp)
            .background(Color.Black),
        ) {
          Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = null,
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxSize(),
          )
          selected.result?.let { res ->
            DetectionOverlay(
              detections = res.detections,
              imageWidth = res.imageWidth,
              imageHeight = res.imageHeight,
              fillCenter = false,
              modifier = Modifier.fillMaxSize(),
            )
          }
        }
      }
    }

    Card(modifier = Modifier.fillMaxWidth()) {
      Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Результати на одному кадрі", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
        for (e in entries) {
          val fb = if (e.requestedDelegate == DelegateType.GPU && e.activeDelegate == DelegateType.CPU) " (fallback)" else ""
          val res = e.result
          Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
              Text("${e.engineType.title} [${e.activeDelegate.name}$fb]", fontWeight = FontWeight.SemiBold)
              if (res != null) {
                Text(
                  String.format(Locale.US, "%.1f мс (%d об'єктів)", res.totalMs, res.detections.size),
                  fontFamily = FontFamily.Monospace,
                  style = MaterialTheme.typography.bodySmall,
                )
              } else {
                Text("Помилка", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
              }
            }
            if (res != null) {
              Text(
                String.format(
                  Locale.US,
                  "Pre %.1f · Inf %.1f · Post %.1f мс | Топ: %s",
                  res.preprocessMs,
                  res.inferenceMs,
                  res.postprocessMs,
                  res.detections.take(4).joinToString { "${it.label} ${(it.score * 100).toInt()}%" }.ifEmpty { "—" },
                ),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
              )
            } else if (e.error != null) {
              Text(e.error, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            }
          }
          HorizontalDivider()
        }
      }
    }
  }
}

@Composable
private fun AutoBenchmarkTable(rows: List<BenchmarkRow>) {
  Card(modifier = Modifier.fillMaxWidth()) {
    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
      Text("Таблиця бенчмарку", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
      Row(modifier = Modifier.fillMaxWidth()) {
        Text("Рушій", modifier = Modifier.weight(1.4f), style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
        Text("Mean", modifier = Modifier.weight(0.8f), style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
        Text("p50/p95", modifier = Modifier.weight(1.1f), style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
        Text("FPS", modifier = Modifier.weight(0.7f), style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
        Text("RAM", modifier = Modifier.weight(0.7f), style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
      }
      HorizontalDivider()
      for (r in rows) {
        val delTag = if (r.isFallback) "GPU→CPU" else r.activeDelegate.name
        Row(
          modifier = Modifier.fillMaxWidth(),
          verticalAlignment = Alignment.CenterVertically,
        ) {
          Text(
            "${r.engineType.title}\n$delTag",
            modifier = Modifier.weight(1.4f),
            style = MaterialTheme.typography.bodySmall,
          )
          if (r.error != null && r.measuredRuns == 0) {
            Text(
              r.error,
              modifier = Modifier.weight(3.3f),
              style = MaterialTheme.typography.bodySmall,
              color = MaterialTheme.colorScheme.error,
            )
          } else {
            Text(
              String.format(Locale.US, "%.1f", r.meanTotalMs),
              modifier = Modifier.weight(0.8f),
              fontFamily = FontFamily.Monospace,
              style = MaterialTheme.typography.bodySmall,
            )
            Text(
              String.format(Locale.US, "%.0f/%.0f", r.p50TotalMs, r.p95TotalMs),
              modifier = Modifier.weight(1.1f),
              fontFamily = FontFamily.Monospace,
              style = MaterialTheme.typography.bodySmall,
            )
            Text(
              String.format(Locale.US, "%.1f", r.fps),
              modifier = Modifier.weight(0.7f),
              fontFamily = FontFamily.Monospace,
              style = MaterialTheme.typography.bodySmall,
            )
            Text(
              String.format(Locale.US, "%.0fM", r.memoryMb),
              modifier = Modifier.weight(0.7f),
              fontFamily = FontFamily.Monospace,
              style = MaterialTheme.typography.bodySmall,
            )
          }
        }
        HorizontalDivider()
      }
    }
  }
}

private val CHART_COLORS = listOf(
  Color(0xFF29B6F6),
  Color(0xFF00E676),
  Color(0xFFFF7043),
  Color(0xFFAB47BC),
  Color(0xFFFFCA28),
  Color(0xFF26A69A),
)

@Composable
private fun ThrottlingChartCard(rows: List<BenchmarkRow>) {
  val valid = rows.filter { it.perRunInfMs.size >= 2 }
  if (valid.isEmpty()) return

  val maxInf = valid.maxOf { r -> r.perRunInfMs.maxOrNull() ?: 1.0 }.coerceAtLeast(1.0)

  Card(modifier = Modifier.fillMaxWidth()) {
    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
      Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text("Троттлінг / Час інференсу (мс)", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
        Text(
          String.format(Locale.US, "max %.0f мс", maxInf),
          style = MaterialTheme.typography.labelSmall,
          fontFamily = FontFamily.Monospace,
        )
      }

      Canvas(
        modifier = Modifier
          .fillMaxWidth()
          .height(140.dp)
          .background(Color(0xFF12161B), RoundedCornerShape(8.dp))
          .padding(8.dp),
      ) {
        valid.forEachIndexed { idx, r ->
          val series = r.perRunInfMs
          val color = CHART_COLORS[idx % CHART_COLORS.size]
          val path = Path()
          val denom = (series.size - 1).coerceAtLeast(1).toFloat()
          series.forEachIndexed { i, v ->
            val x = (i / denom) * size.width
            val y = size.height - ((v / maxInf).toFloat().coerceIn(0f, 1f) * size.height)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
          }
          drawPath(path, color = color, style = Stroke(width = 2.dp.toPx()))
          // Subtle baseline dots for short series
          if (series.size <= 30) {
            series.forEachIndexed { i, v ->
              val x = (i / denom) * size.width
              val y = size.height - ((v / maxInf).toFloat().coerceIn(0f, 1f) * size.height)
              drawCircle(color = color, radius = 2.dp.toPx(), center = Offset(x, y))
            }
          }
        }
      }

      Row(
        modifier = Modifier
          .fillMaxWidth()
          .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
      ) {
        valid.forEachIndexed { idx, r ->
          val c = CHART_COLORS[idx % CHART_COLORS.size]
          Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            Box(
              modifier = Modifier
                .size(10.dp)
                .background(c, RoundedCornerShape(2.dp)),
            )
            Text(
              "${r.engineType.title} ${r.requestedDelegate.name}",
              style = MaterialTheme.typography.labelSmall,
            )
          }
        }
      }
    }
  }
}
