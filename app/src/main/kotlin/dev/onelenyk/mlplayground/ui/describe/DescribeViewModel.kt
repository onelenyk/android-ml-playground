package dev.onelenyk.mlplayground.ui.describe

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.vlm.GemmaDescriber
import dev.onelenyk.mlplayground.ml.vlm.VlmMetrics
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

data class PromptPreset(val title: String, val prompt: String)

val DESCRIBE_PRESETS = listOf(
  PromptPreset(
    "Список предметів",
    "Перелічи всі предмети, які ти бачиш на цьому фото (короткий список українською, із зазначенням кольорів і деталей).",
  ),
  PromptPreset(
    "Детальний опис",
    "Детально опиши українською, що зображено на фото: предмети, їх розташування, кольори та стан.",
  ),
  PromptPreset(
    "Що зайве?",
    "Подивись на цей стіл або сцену і напиши українською, що тут лежить і чи є щось незвичне або зайве.",
  ),
)

data class DescribeUiState(
  val modelFile: File? = null,
  val expectedPath: String = "",
  val modelSizeGb: Double = 0.0,
  val isEngineLoaded: Boolean = false,
  val isLoadingEngine: Boolean = false,
  val loadTimeSec: Double? = null,
  val requestedDelegate: DelegateType = DelegateType.GPU,
  val activeDelegate: DelegateType = DelegateType.GPU,
  val capturedBitmap: Bitmap? = null,
  val prompt: String = DESCRIBE_PRESETS.first().prompt,
  val enableThinking: Boolean = false,
  val isGenerating: Boolean = false,
  val streamedText: String = "",
  val streamedThought: String = "",
  val metrics: VlmMetrics? = null,
  val error: String? = null,
)

class DescribeViewModel(application: Application) : AndroidViewModel(application) {
  private val executor = Executors.newSingleThreadExecutor { r ->
    Thread(r, "ml-vlm").apply { isDaemon = true }
  }
  private val vlmDispatcher = executor.asCoroutineDispatcher()
  private val describer = GemmaDescriber(application)
  private val latestLiveFrame = AtomicReference<Bitmap?>(null)
  private val captureAndRunNext = AtomicBoolean(false)

  private val _uiState = MutableStateFlow(DescribeUiState())
  val uiState: StateFlow<DescribeUiState> = _uiState.asStateFlow()

  init {
    refreshModelAvailability()
  }

  fun refreshModelAvailability() {
    val ctx = getApplication<Application>()
    val file = GemmaDescriber.findModelFile(ctx)
    val expected = GemmaDescriber.expectedExternalPath(ctx).absolutePath
    _uiState.update {
      it.copy(
        modelFile = file,
        expectedPath = expected,
        modelSizeGb = (file?.length() ?: 0L) / 1_000_000_000.0,
      )
    }
  }

  fun onLiveFrame(bitmap: Bitmap) {
    latestLiveFrame.set(bitmap)
    if (captureAndRunNext.compareAndSet(true, false)) {
      _uiState.update { it.copy(capturedBitmap = bitmap) }
      describeBitmap(bitmap)
    }
  }

  fun captureAndDescribe() {
    if (_uiState.value.isGenerating || _uiState.value.isLoadingEngine) return
    val snap = latestLiveFrame.get()
    if (snap != null) {
      _uiState.update { it.copy(capturedBitmap = snap) }
      describeBitmap(snap)
    } else {
      captureAndRunNext.set(true)
    }
  }

  fun describeCurrentPhoto() {
    val bmp = _uiState.value.capturedBitmap ?: return
    if (_uiState.value.isGenerating || _uiState.value.isLoadingEngine) return
    describeBitmap(bmp)
  }

  fun retakePhoto() {
    if (_uiState.value.isGenerating) return
    latestLiveFrame.set(null)
    captureAndRunNext.set(false)
    _uiState.update {
      it.copy(capturedBitmap = null, streamedText = "", streamedThought = "", metrics = null, error = null)
    }
  }

  fun setPrompt(prompt: String) {
    _uiState.update { it.copy(prompt = prompt) }
  }

  fun setEnableThinking(enabled: Boolean) {
    _uiState.update { it.copy(enableThinking = enabled) }
  }

  fun setDelegate(delegate: DelegateType) {
    if (_uiState.value.requestedDelegate == delegate || _uiState.value.isGenerating) return
    _uiState.update { it.copy(requestedDelegate = delegate) }
    if (_uiState.value.isEngineLoaded) {
      unloadModel()
    }
  }

  fun unloadModel() {
    if (_uiState.value.isGenerating) return
    viewModelScope.launch(vlmDispatcher) {
      describer.close()
      _uiState.update { it.copy(isEngineLoaded = false, loadTimeSec = null) }
    }
  }

  fun stopGeneration() {
    describer.cancel()
  }

  private fun describeBitmap(bitmap: Bitmap) {
    val file = _uiState.value.modelFile ?: GemmaDescriber.findModelFile(getApplication())
    if (file == null) {
      _uiState.update { it.copy(error = "Файл моделі .litertlm не знайдено") }
      return
    }
    val reqDelegate = _uiState.value.requestedDelegate
    val prompt = _uiState.value.prompt.ifBlank { DESCRIBE_PRESETS.first().prompt }
    val thinking = _uiState.value.enableThinking

    _uiState.update {
      it.copy(
        modelFile = file,
        streamedText = "",
        streamedThought = "",
        metrics = null,
        error = null,
        isGenerating = true,
      )
    }

    viewModelScope.launch(vlmDispatcher) {
      try {
        if (!describer.isInitialized || describer.loadedModelFile != file) {
          _uiState.update { it.copy(isLoadingEngine = true) }
          val loadSec = describer.initialize(file, reqDelegate)
          _uiState.update {
            it.copy(
              isLoadingEngine = false,
              isEngineLoaded = true,
              loadTimeSec = loadSec,
              activeDelegate = describer.activeDelegate,
            )
          }
        }

        val answerBuf = StringBuilder()
        val thoughtBuf = StringBuilder()
        val metrics = describer.describe(bitmap, prompt, thinking) { ansDelta, thoughtDelta ->
          if (ansDelta.isNotEmpty()) answerBuf.append(ansDelta)
          if (thoughtDelta.isNotEmpty()) thoughtBuf.append(thoughtDelta)
          _uiState.update {
            it.copy(
              streamedText = answerBuf.toString(),
              streamedThought = thoughtBuf.toString(),
            )
          }
        }
        _uiState.update { it.copy(isGenerating = false, metrics = metrics) }
      } catch (t: Throwable) {
        Log.e(TAG, "Gemma describe failed", t)
        _uiState.update {
          it.copy(
            isLoadingEngine = false,
            isGenerating = false,
            error = t.message ?: t::class.simpleName ?: "Помилка інференсу Gemma",
          )
        }
      }
    }
  }

  override fun onCleared() {
    super.onCleared()
    describer.cancel()
    executor.execute { describer.close() }
    executor.shutdown()
  }

  companion object {
    private const val TAG = "DescribeViewModel"
  }
}
