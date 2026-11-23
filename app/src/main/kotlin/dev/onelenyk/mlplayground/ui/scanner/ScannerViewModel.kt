package dev.onelenyk.mlplayground.ui.scanner

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.DetectionResult
import dev.onelenyk.mlplayground.ml.core.EngineFactory
import dev.onelenyk.mlplayground.ml.core.EngineType
import dev.onelenyk.mlplayground.ml.core.ObjectDetectorEngine
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

data class ScannerUiState(
  val engineType: EngineType = EngineType.LITERT,
  val requestedDelegate: DelegateType = DelegateType.CPU,
  val activeDelegate: DelegateType = DelegateType.CPU,
  val inputSize: Int = 320,
  val threshold: Float = 0.30f,
  val isLoadingEngine: Boolean = true,
  val error: String? = null,
  val modelInfo: String = "",
  val result: DetectionResult? = null,
  val fps: Double = 0.0,
  val inventory: List<InventoryItem> = emptyList(),
  val findQuery: String = "",
  val findMatched: Boolean = false,
)

class ScannerViewModel(application: Application) : AndroidViewModel(application) {
  private data class EngineConfigKey(
    val engineType: EngineType,
    val delegate: DelegateType,
    val inputSize: Int,
  )

  private val executor = Executors.newSingleThreadExecutor { r ->
    Thread(r, "ml-scanner").apply { isDaemon = true }
  }
  private val mlDispatcher = executor.asCoroutineDispatcher()

  private var engine: ObjectDetectorEngine? = null
  private val tracker = DeskInventoryTracker()
  private val isProcessing = AtomicBoolean(false)
  private val isActive = AtomicBoolean(true)
  private val desiredGeneration = AtomicLong(0L)
  private var lastFrameNs = 0L
  private var emaFps = 0.0

  private val _uiState = MutableStateFlow(ScannerUiState())
  val uiState: StateFlow<ScannerUiState> = _uiState.asStateFlow()

  init {
    rebuildEngine()
  }

  fun setActive(active: Boolean) {
    isActive.set(active)
    if (!active) {
      lastFrameNs = 0L
      emaFps = 0.0
    }
  }

  fun selectEngine(type: EngineType) {
    if (_uiState.value.engineType == type) return
    _uiState.update { it.copy(engineType = type, isLoadingEngine = true, error = null, result = null) }
    rebuildEngine()
  }

  fun selectDelegate(delegate: DelegateType) {
    if (_uiState.value.requestedDelegate == delegate) return
    _uiState.update { it.copy(requestedDelegate = delegate, isLoadingEngine = true, error = null, result = null) }
    rebuildEngine()
  }

  fun selectInputSize(size: Int) {
    if (_uiState.value.inputSize == size) return
    _uiState.update { it.copy(inputSize = size, isLoadingEngine = true, error = null, result = null) }
    rebuildEngine()
  }

  fun setThreshold(threshold: Float) {
    tracker.reset()
    _uiState.update { it.copy(threshold = threshold, inventory = emptyList()) }
  }

  fun setFindQuery(query: String) {
    _uiState.update { s ->
      val q = query.trim()
      val matched = q.isNotEmpty() && (s.result?.detections?.any { it.label.contains(q, ignoreCase = true) } == true)
      s.copy(findQuery = query, findMatched = matched)
    }
  }

  fun toggleInventoryChip(label: String) {
    val current = _uiState.value.findQuery
    setFindQuery(if (current.equals(label, ignoreCase = true)) "" else label)
  }

  private fun rebuildEngine() {
    val snap = _uiState.value
    val key = EngineConfigKey(snap.engineType, snap.requestedDelegate, snap.inputSize)
    val gen = desiredGeneration.incrementAndGet()

    viewModelScope.launch(mlDispatcher) {
      if (gen != desiredGeneration.get()) return@launch
      _uiState.update { it.copy(isLoadingEngine = true, error = null) }

      runCatching { engine?.close() }
      engine = null
      tracker.reset()
      lastFrameNs = 0L
      emaFps = 0.0

      try {
        val created = EngineFactory.create(getApplication(), key.engineType, key.inputSize)
        created.initialize(key.delegate)
        if (gen != desiredGeneration.get()) {
          runCatching { created.close() }
          return@launch
        }
        engine = created
        _uiState.update {
          it.copy(
            isLoadingEngine = false,
            activeDelegate = created.activeDelegate,
            modelInfo = created.modelInfo,
            error = null,
            inventory = emptyList(),
            fps = 0.0,
          )
        }
      } catch (t: Throwable) {
        Log.e(TAG, "Failed to initialize ${key.engineType} (${key.delegate})", t)
        if (gen == desiredGeneration.get()) {
          _uiState.update {
            it.copy(
              isLoadingEngine = false,
              error = t.message ?: t::class.simpleName ?: "Помилка ініціалізації",
            )
          }
        }
      }
    }
  }

  fun onFrame(bitmap: Bitmap) {
    if (!isActive.get() || _uiState.value.isLoadingEngine) return
    if (!isProcessing.compareAndSet(false, true)) return

    viewModelScope.launch(mlDispatcher) {
      try {
        val eng = engine
        if (eng == null || !isActive.get() || _uiState.value.isLoadingEngine) return@launch

        val threshold = _uiState.value.threshold
        val res = eng.detect(bitmap, threshold)

        val now = System.nanoTime()
        if (lastFrameNs > 0L) {
          val deltaSec = (now - lastFrameNs) / 1_000_000_000.0
          if (deltaSec > 0.0) {
            val instFps = 1.0 / deltaSec
            emaFps = if (emaFps == 0.0) instFps else emaFps * 0.8 + instFps * 0.2
          }
        }
        lastFrameNs = now

        val inv = tracker.update(res.detections)
        val q = _uiState.value.findQuery.trim()
        val matched = q.isNotEmpty() && res.detections.any { it.label.contains(q, ignoreCase = true) }

        _uiState.update {
          it.copy(
            result = res,
            fps = emaFps,
            inventory = inv,
            findMatched = matched,
            error = null,
          )
        }
      } catch (t: Throwable) {
        Log.e(TAG, "Detection error", t)
        _uiState.update { it.copy(error = t.message ?: t::class.simpleName ?: "Помилка детекції") }
      } finally {
        isProcessing.set(false)
      }
    }
  }

  override fun onCleared() {
    super.onCleared()
    desiredGeneration.incrementAndGet()
    executor.execute {
      runCatching { engine?.close() }
      engine = null
    }
    executor.shutdown()
  }

  companion object {
    private const val TAG = "ScannerViewModel"
  }
}
