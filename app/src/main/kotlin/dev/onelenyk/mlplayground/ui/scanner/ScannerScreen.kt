package dev.onelenyk.mlplayground.ui.scanner

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
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.onelenyk.mlplayground.camera.CameraPreview
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.EngineType
import dev.onelenyk.mlplayground.ml.yolo.YoloModels
import dev.onelenyk.mlplayground.ui.common.DetectionOverlay
import dev.onelenyk.mlplayground.ui.common.MetricsHud
import java.util.Locale

@Composable
fun ScannerScreen(
  viewModel: ScannerViewModel,
  isActive: Boolean,
  modifier: Modifier = Modifier,
) {
  val state by viewModel.uiState.collectAsStateWithLifecycle()
  val haptic = LocalHapticFeedback.current
  var showSearchField by rememberSaveable { mutableStateOf(false) }

  LaunchedEffect(state.findMatched) {
    if (state.findMatched) {
      haptic.performHapticFeedback(HapticFeedbackType.LongPress)
    }
  }

  Column(modifier = modifier.fillMaxSize()) {
    Box(
      modifier = Modifier
        .weight(1f)
        .fillMaxWidth(),
    ) {
      if (isActive) {
        CameraPreview(
          modifier = Modifier.fillMaxSize(),
          analysisEnabled = !state.isLoadingEngine,
          onFrame = viewModel::onFrame,
        )
      }

      val res = state.result
      if (res != null) {
        DetectionOverlay(
          detections = res.detections,
          imageWidth = res.imageWidth,
          imageHeight = res.imageHeight,
          highlightQuery = state.findQuery,
          fillCenter = true,
          modifier = Modifier.fillMaxSize(),
        )
      }

      MetricsHud(
        engineType = state.engineType,
        requestedDelegate = state.requestedDelegate,
        activeDelegate = state.activeDelegate,
        modelInfo = state.modelInfo,
        fps = state.fps,
        result = res,
        modifier = Modifier
          .align(Alignment.TopCenter)
          .padding(12.dp),
      )

      if (state.isLoadingEngine) {
        Surface(
          modifier = Modifier.align(Alignment.Center),
          shape = RoundedCornerShape(16.dp),
          color = Color(0xCC101418),
        ) {
          Row(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
          ) {
            CircularProgressIndicator(modifier = Modifier.size(22.dp), strokeWidth = 2.5.dp)
            Text(
              text = "Завантаження ${state.engineType.title} (${state.requestedDelegate.name})…",
              color = Color.White,
              style = MaterialTheme.typography.bodyMedium,
            )
          }
        }
      }

      val err = state.error
      if (err != null) {
        Surface(
          modifier = Modifier
            .align(Alignment.BottomCenter)
            .padding(12.dp),
          color = MaterialTheme.colorScheme.errorContainer,
          shape = RoundedCornerShape(12.dp),
        ) {
          Text(
            text = err,
            color = MaterialTheme.colorScheme.onErrorContainer,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
          )
        }
      }
    }

    Surface(
      tonalElevation = 3.dp,
      modifier = Modifier.fillMaxWidth(),
    ) {
      Column(
        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
      ) {
        // Live desk inventory row + "Find My Item" search toggle
        Row(
          modifier = Modifier.fillMaxWidth(),
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
          Text(
            text = "На столі:",
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
          )
          if (state.inventory.isEmpty()) {
            Text(
              text = "Наведіть камеру на стіл…",
              style = MaterialTheme.typography.bodySmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
              modifier = Modifier.weight(1f),
            )
          } else {
            LazyRow(
              modifier = Modifier.weight(1f),
              horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
              items(state.inventory, key = { it.label }) { item ->
                val selected = state.findQuery.equals(item.label, ignoreCase = true)
                AssistChip(
                  onClick = { viewModel.toggleInventoryChip(item.label) },
                  label = { Text("${item.label} ×${item.count}") },
                  colors = if (selected) {
                    AssistChipDefaults.assistChipColors(
                      containerColor = MaterialTheme.colorScheme.primaryContainer,
                      labelColor = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                  } else {
                    AssistChipDefaults.assistChipColors()
                  },
                )
              }
            }
          }
          IconButton(
            onClick = {
              showSearchField = !showSearchField
              if (!showSearchField) viewModel.setFindQuery("")
            },
            modifier = Modifier.size(34.dp),
          ) {
            Icon(
              imageVector = if (showSearchField) Icons.Default.Close else Icons.Default.Search,
              contentDescription = "Знайти предмет",
            )
          }
        }

        if (showSearchField) {
          OutlinedTextField(
            value = state.findQuery,
            onValueChange = viewModel::setFindQuery,
            singleLine = true,
            placeholder = { Text("Знайти предмет (англ., напр. keys, watch)") },
            trailingIcon = {
              if (state.findQuery.isNotEmpty()) {
                IconButton(onClick = { viewModel.setFindQuery("") }) {
                  Icon(Icons.Default.Close, contentDescription = "Очистити")
                }
              }
            },
            modifier = Modifier.fillMaxWidth(),
          )
        }

        // Engine selector
        Row(
          modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
          horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          EngineType.entries.forEach { eng ->
            FilterChip(
              selected = state.engineType == eng,
              onClick = { viewModel.selectEngine(eng) },
              label = { Text(eng.title) },
            )
          }
        }

        // Delegate + input size + threshold slider
        Row(
          modifier = Modifier.fillMaxWidth(),
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
          DelegateType.entries.forEach { del ->
            FilterChip(
              selected = state.requestedDelegate == del,
              onClick = { viewModel.selectDelegate(del) },
              label = { Text(del.name) },
            )
          }
          if (state.engineType != EngineType.MEDIAPIPE) {
            YoloModels.SUPPORTED_SIZES.forEach { sz ->
              FilterChip(
                selected = state.inputSize == sz,
                onClick = { viewModel.selectInputSize(sz) },
                label = { Text("${sz}px") },
              )
            }
          }
          Spacer(Modifier.weight(1f))
          Text(
            text = String.format(Locale.US, "Поріг %.2f", state.threshold),
            style = MaterialTheme.typography.labelMedium,
          )
        }

        Slider(
          value = state.threshold,
          onValueChange = viewModel::setThreshold,
          valueRange = 0.15f..0.80f,
          modifier = Modifier
            .fillMaxWidth()
            .height(28.dp),
        )
      }
    }
  }
}
