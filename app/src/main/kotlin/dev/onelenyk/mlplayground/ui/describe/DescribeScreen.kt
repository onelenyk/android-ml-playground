package dev.onelenyk.mlplayground.ui.describe

import androidx.compose.foundation.Image
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
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.onelenyk.mlplayground.camera.CameraPreview
import dev.onelenyk.mlplayground.ml.core.DelegateType
import java.util.Locale

@Composable
fun DescribeScreen(
  viewModel: DescribeViewModel,
  isActive: Boolean,
  modifier: Modifier = Modifier,
) {
  val state by viewModel.uiState.collectAsStateWithLifecycle()

  Column(
    modifier = modifier
      .fillMaxSize()
      .verticalScroll(rememberScrollState())
      .padding(12.dp),
    verticalArrangement = Arrangement.spacedBy(10.dp),
  ) {
    if (state.modelFile == null) {
      ModelMissingCard(
        expectedPath = state.expectedPath,
        onRefresh = viewModel::refreshModelAvailability,
      )
    } else {
      Surface(
        shape = RoundedCornerShape(10.dp),
        color = MaterialTheme.colorScheme.secondaryContainer,
        modifier = Modifier.fillMaxWidth(),
      ) {
        val status = when {
          state.isLoadingEngine -> "Завантаження Gemma 4 E2B…"
          state.isEngineLoaded -> {
            val fb = if (state.requestedDelegate == DelegateType.GPU && state.activeDelegate == DelegateType.CPU) {
              " (GPU→CPU)"
            } else {
              ""
            }
            val loadStr = state.loadTimeSec?.let { String.format(Locale.US, " за %.1f с", it) } ?: ""
            "Готова [${state.activeDelegate.name}$fb$loadStr]"
          }
          else -> String.format(Locale.US, "%s (%.2f ГБ) — очікує запуску", state.modelFile!!.name, state.modelSizeGb)
        }
        Text(
          text = status,
          style = MaterialTheme.typography.labelMedium,
          color = MaterialTheme.colorScheme.onSecondaryContainer,
          modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
        )
      }
    }

    // Camera or frozen photo preview
    Card(modifier = Modifier.fillMaxWidth()) {
      Column(modifier = Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val bmp = state.capturedBitmap
        Box(
          modifier = Modifier
            .fillMaxWidth()
            .height(220.dp)
            .clip(RoundedCornerShape(10.dp)),
        ) {
          if (bmp == null) {
            if (isActive) {
              CameraPreview(
                modifier = Modifier.fillMaxSize(),
                analysisEnabled = true,
                onFrame = viewModel::onLiveFrame,
              )
            }
          } else {
            Image(
              bitmap = bmp.asImageBitmap(),
              contentDescription = "Знятий кадр",
              contentScale = ContentScale.Crop,
              modifier = Modifier.fillMaxSize(),
            )
          }
        }

        if (bmp == null) {
          Button(
            onClick = viewModel::captureAndDescribe,
            enabled = state.modelFile != null && !state.isGenerating && !state.isLoadingEngine,
            modifier = Modifier.fillMaxWidth(),
          ) {
            Icon(Icons.Default.CameraAlt, contentDescription = null)
            Spacer(Modifier.width(6.dp))
            Text("Зняти й описати")
          }
        } else {
          Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
              onClick = viewModel::retakePhoto,
              enabled = !state.isGenerating,
              modifier = Modifier.weight(1f),
            ) {
              Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
              Spacer(Modifier.width(4.dp))
              Text("Нове фото")
            }
            Button(
              onClick = viewModel::describeCurrentPhoto,
              enabled = state.modelFile != null && !state.isGenerating && !state.isLoadingEngine,
              modifier = Modifier.weight(1f),
            ) {
              Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(16.dp))
              Spacer(Modifier.width(4.dp))
              Text("Описати знову")
            }
          }
        }
      }
    }

    // Preset prompt chips
    Row(
      modifier = Modifier
        .fillMaxWidth()
        .horizontalScroll(rememberScrollState()),
      horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
      DESCRIBE_PRESETS.forEach { preset ->
        FilterChip(
          selected = state.prompt == preset.prompt,
          onClick = { viewModel.setPrompt(preset.prompt) },
          label = { Text(preset.title) },
        )
      }
    }

    OutlinedTextField(
      value = state.prompt,
      onValueChange = viewModel::setPrompt,
      label = { Text("Промпт для Gemma 4 E2B") },
      maxLines = 3,
      modifier = Modifier.fillMaxWidth(),
    )

    // Backend, Thinking toggle, Unload button
    Row(
      modifier = Modifier
        .fillMaxWidth()
        .horizontalScroll(rememberScrollState()),
      verticalAlignment = Alignment.CenterVertically,
      horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
      DelegateType.entries.forEach { del ->
        FilterChip(
          selected = state.requestedDelegate == del,
          onClick = { viewModel.setDelegate(del) },
          label = { Text(del.name) },
        )
      }

      Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Thinking", style = MaterialTheme.typography.labelMedium)
        Switch(
          checked = state.enableThinking,
          onCheckedChange = viewModel::setEnableThinking,
          enabled = !state.isGenerating,
        )
      }

      if (state.isEngineLoaded && !state.isGenerating) {
        OutlinedButton(onClick = viewModel::unloadModel) {
          Icon(Icons.Default.DeleteOutline, contentDescription = null, modifier = Modifier.size(16.dp))
          Spacer(Modifier.width(4.dp))
          Text("Вивантажити модель")
        }
      }
    }

    if (state.isLoadingEngine || state.isGenerating) {
      Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
      ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.5.dp)
          Text(
            if (state.isLoadingEngine) "Ініціалізація Gemma 4 E2B (кілька секунд)…" else "Генерація опису…",
            style = MaterialTheme.typography.bodySmall,
          )
        }
        if (state.isGenerating && !state.isLoadingEngine) {
          OutlinedButton(onClick = viewModel::stopGeneration) {
            Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(4.dp))
            Text("Стоп")
          }
        }
      }
    }

    state.error?.let { err ->
      Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        modifier = Modifier.fillMaxWidth(),
      ) {
        Text(
          text = err,
          color = MaterialTheme.colorScheme.onErrorContainer,
          style = MaterialTheme.typography.bodySmall,
          modifier = Modifier.padding(12.dp),
        )
      }
    }

    if (state.streamedThought.isNotEmpty() || state.streamedText.isNotEmpty() || state.metrics != null) {
      Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
          Text("Відповідь Gemma 4 E2B", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
          if (state.streamedThought.isNotEmpty()) {
            Text(
              text = "Роздуми:\n${state.streamedThought}",
              style = MaterialTheme.typography.bodySmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
          }
          if (state.streamedText.isNotEmpty()) {
            Text(
              text = state.streamedText,
              style = MaterialTheme.typography.bodyMedium,
            )
          }
          state.metrics?.let { m ->
            Text(
              text = String.format(
                Locale.US,
                "TTFT: %.2f с | Prefill: %.1f tok/s (%d) | Decode: %.1f tok/s (%d) | Всього: %.2f с",
                m.ttftSec,
                m.prefillTokensPerSec,
                m.prefillTokens,
                m.decodeTokensPerSec,
                m.decodeTokens,
                m.totalSec,
              ),
              style = MaterialTheme.typography.labelSmall,
              fontFamily = FontFamily.Monospace,
              color = MaterialTheme.colorScheme.primary,
            )
          }
        }
      }
    }
  }
}

@Composable
private fun ModelMissingCard(expectedPath: String, onRefresh: () -> Unit) {
  Card(
    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
    modifier = Modifier.fillMaxWidth(),
  ) {
    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
      Text(
        "Модель Gemma 4 E2B не знайдена на пристрої",
        style = MaterialTheme.typography.titleSmall,
        fontWeight = FontWeight.Bold,
        color = MaterialTheme.colorScheme.onErrorContainer,
      )
      Text(
        "Потрібен файл gemma-4-E2B-it.litertlm (~2.59 ГБ). Покладіть його за шляхом:\n$expectedPath",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onErrorContainer,
      )
      OutlinedButton(onClick = onRefresh) {
        Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
        Spacer(Modifier.width(6.dp))
        Text("Перевірити знову")
      }
    }
  }
}
