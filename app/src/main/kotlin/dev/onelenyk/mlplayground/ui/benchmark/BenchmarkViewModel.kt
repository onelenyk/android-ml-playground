package dev.onelenyk.mlplayground.ui.benchmark

import android.app.Application
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Debug
import android.util.Log
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.EngineFactory
import dev.onelenyk.mlplayground.ml.core.EngineType
import dev.onelenyk.mlplayground.ml.core.nowMs
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

enum class BenchmarkSubMode(val label: String) {
  SAME_FRAME("1 кадр"),
  AUTO_RUN("Автопрогін"),
}

data class BenchmarkUiState(
  val subMode: BenchmarkSubMode = BenchmarkSubMode.SAME_FRAME,
  val capturedBitmap: Bitmap? = null,
  val inputSize: Int = 320,
  val threshold: Float = 0.30f,
  val sameFrameDelegate: DelegateType = DelegateType.CPU,
  val sameFrameResults: List<SameFrameEntry> = emptyList(),
  val selectedPreviewEngine: EngineType = EngineType.LITERT,
  val autoRunCount: Int = 30,
  val autoResults: List<BenchmarkRow> = emptyList(),
  val isRunning: Boolean = false,
  val progressText: String = "",
  val progressFraction: Float = 0f,
  val savedReportPath: String? = null,
  val error: String? = null,
)

class BenchmarkViewModel(application: Application) : AndroidViewModel(application) {
  private val executor = Executors.newSingleThreadExecutor { r ->
    Thread(r, "ml-benchmark").apply { isDaemon = true }
  }
  private val benchDispatcher = executor.asCoroutineDispatcher()
  private val latestLiveFrame = AtomicReference<Bitmap?>(null)
  private val captureNextFrame = AtomicBoolean(false)
  private val cancelRequested = AtomicBoolean(false)

  private val _uiState = MutableStateFlow(BenchmarkUiState())
  val uiState: StateFlow<BenchmarkUiState> = _uiState.asStateFlow()

  fun setSubMode(mode: BenchmarkSubMode) {
    _uiState.update { it.copy(subMode = mode) }
  }

  fun setInputSize(size: Int) {
    _uiState.update { it.copy(inputSize = size) }
  }

  fun setSameFrameDelegate(delegate: DelegateType) {
    _uiState.update { it.copy(sameFrameDelegate = delegate) }
  }

  fun setAutoRunCount(count: Int) {
    _uiState.update { it.copy(autoRunCount = count) }
  }

  fun selectPreviewEngine(type: EngineType) {
    _uiState.update { it.copy(selectedPreviewEngine = type) }
  }

  fun onLiveFrame(bitmap: Bitmap) {
    latestLiveFrame.set(bitmap)
    if (captureNextFrame.compareAndSet(true, false)) {
      _uiState.update { it.copy(capturedBitmap = bitmap, sameFrameResults = emptyList(), error = null) }
    }
  }

  fun captureFrame() {
    val snap = latestLiveFrame.get()
    if (snap != null) {
      _uiState.update { it.copy(capturedBitmap = snap, sameFrameResults = emptyList(), error = null) }
    } else {
      captureNextFrame.set(true)
    }
  }

  fun retakeFrame() {
    if (_uiState.value.isRunning) return
    latestLiveFrame.set(null)
    captureNextFrame.set(false)
    _uiState.update { it.copy(capturedBitmap = null, sameFrameResults = emptyList(), error = null) }
  }

  fun runSameFrameCompare() {
    val bmp = _uiState.value.capturedBitmap ?: return
    if (_uiState.value.isRunning) return
    val inputSize = _uiState.value.inputSize
    val delegate = _uiState.value.sameFrameDelegate
    val threshold = _uiState.value.threshold

    cancelRequested.set(false)
    _uiState.update {
      it.copy(isRunning = true, sameFrameResults = emptyList(), progressFraction = 0f, error = null)
    }

    viewModelScope.launch(benchDispatcher) {
      val list = ArrayList<SameFrameEntry>()
      val engines = EngineType.entries
      for ((idx, engType) in engines.withIndex()) {
        if (cancelRequested.get()) break
        _uiState.update {
          it.copy(
            progressText = "Перевірка ${engType.title} (${delegate.name})…",
            progressFraction = idx.toFloat() / engines.size,
          )
        }
        val entry = runCatching {
          val eng = EngineFactory.create(getApplication(), engType, inputSize)
          try {
            eng.initialize(delegate)
            eng.detect(bmp, threshold) // warmup
            val measured = eng.detect(bmp, threshold)
            SameFrameEntry(engType, delegate, eng.activeDelegate, eng.modelInfo, measured, null)
          } finally {
            runCatching { eng.close() }
          }
        }.getOrElse { t ->
          Log.e(TAG, "SameFrame error for $engType", t)
          SameFrameEntry(engType, delegate, delegate, "-", null, t.message ?: t::class.simpleName)
        }
        list += entry
        _uiState.update { it.copy(sameFrameResults = list.toList()) }
      }
      _uiState.update {
        it.copy(isRunning = false, progressText = "", progressFraction = 1f)
      }
    }
  }

  fun runAutoBenchmark() {
    val bmp = _uiState.value.capturedBitmap ?: return
    if (_uiState.value.isRunning) return
    val inputSize = _uiState.value.inputSize
    val runs = _uiState.value.autoRunCount
    val warmup = 5
    val threshold = _uiState.value.threshold
    val combos = EngineType.entries.flatMap { eng -> DelegateType.entries.map { del -> eng to del } }

    cancelRequested.set(false)
    _uiState.update {
      it.copy(
        isRunning = true,
        autoResults = emptyList(),
        progressFraction = 0f,
        progressText = "Підготовка бенчмарку…",
        savedReportPath = null,
        error = null,
      )
    }

    viewModelScope.launch(benchDispatcher) {
      val rows = ArrayList<BenchmarkRow>()
      for ((comboIdx, pair) in combos.withIndex()) {
        if (cancelRequested.get()) break
        val (engType, delegate) = pair

        System.gc()
        Thread.sleep(150)

        _uiState.update {
          it.copy(
            progressText = "${engType.title} ${delegate.name}: ініціалізація…",
            progressFraction = comboIdx.toFloat() / combos.size,
          )
        }

        val row = try {
          val eng = EngineFactory.create(getApplication(), engType, inputSize)
          try {
            val tInit0 = nowMs()
            eng.initialize(delegate)
            val initMs = nowMs() - tInit0

            for (w in 0 until warmup) {
              if (cancelRequested.get()) break
              eng.detect(bmp, threshold)
            }

            val preList = ArrayList<Double>(runs)
            val infList = ArrayList<Double>(runs)
            val postList = ArrayList<Double>(runs)
            val totalList = ArrayList<Double>(runs)

            for (r in 0 until runs) {
              if (cancelRequested.get()) break
              if (r % 5 == 0 || r == runs - 1) {
                _uiState.update {
                  it.copy(
                    progressText = "Тест ${engType.title} ${delegate.name} (${r + 1}/$runs)…",
                    progressFraction = (comboIdx + (r + 1f) / runs) / combos.size,
                  )
                }
              }
              val res = eng.detect(bmp, threshold)
              preList += res.preprocessMs
              infList += res.inferenceMs
              postList += res.postprocessMs
              totalList += res.totalMs
            }

            val memMb = currentMemoryMb()
            val sortedTotals = totalList.sorted()
            val n = sortedTotals.size
            val meanPre = preList.average()
            val meanInf = infList.average()
            val meanPost = postList.average()
            val meanTotal = if (n > 0) totalList.average() else 0.0
            val p50 = if (n > 0) sortedTotals[n / 2] else 0.0
            val p95 = if (n > 0) sortedTotals[((n - 1) * 0.95).toInt().coerceIn(0, n - 1)] else 0.0
            val fps = if (meanTotal > 0.0) 1000.0 / meanTotal else 0.0

            BenchmarkRow(
              engineType = engType,
              requestedDelegate = delegate,
              activeDelegate = eng.activeDelegate,
              inputSize = inputSize,
              modelInfo = eng.modelInfo,
              initMs = initMs,
              warmupRuns = warmup,
              measuredRuns = n,
              meanPreMs = meanPre,
              meanInfMs = meanInf,
              meanPostMs = meanPost,
              meanTotalMs = meanTotal,
              p50TotalMs = p50,
              p95TotalMs = p95,
              fps = fps,
              memoryMb = memMb,
              perRunInfMs = infList,
              error = if (cancelRequested.get()) "Скасовано" else null,
            )
          } finally {
            runCatching { eng.close() }
          }
        } catch (t: Throwable) {
          Log.e(TAG, "AutoBenchmark failed for $engType $delegate", t)
          BenchmarkRow(
            engineType = engType,
            requestedDelegate = delegate,
            activeDelegate = delegate,
            inputSize = inputSize,
            modelInfo = "-",
            initMs = 0.0,
            warmupRuns = warmup,
            measuredRuns = 0,
            meanPreMs = 0.0,
            meanInfMs = 0.0,
            meanPostMs = 0.0,
            meanTotalMs = 0.0,
            p50TotalMs = 0.0,
            p95TotalMs = 0.0,
            fps = 0.0,
            memoryMb = currentMemoryMb(),
            perRunInfMs = emptyList(),
            error = t.message ?: t::class.simpleName,
          )
        }
        rows += row
        _uiState.update { it.copy(autoResults = rows.toList()) }
      }

      val saved = if (rows.isNotEmpty()) BenchmarkExporter.saveToExternal(getApplication(), rows) else null
      _uiState.update {
        it.copy(
          isRunning = false,
          progressText = "",
          progressFraction = 1f,
          savedReportPath = saved?.second?.absolutePath,
        )
      }
    }
  }

  fun cancelRun() {
    cancelRequested.set(true)
  }

  fun buildShareIntent(context: Context, format: ExportFormat): Intent? {
    val rows = _uiState.value.autoResults
    if (rows.isEmpty()) return null
    val file = BenchmarkExporter.writeShareFile(context, rows, format)
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    val send = Intent(Intent.ACTION_SEND).apply {
      type = format.mime
      putExtra(Intent.EXTRA_STREAM, uri)
      addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    return Intent.createChooser(send, "Експорт бенчмарку (${format.ext.uppercase()})")
  }

  private fun currentMemoryMb(): Double {
    val rt = Runtime.getRuntime()
    val jvmBytes = rt.totalMemory() - rt.freeMemory()
    val nativeBytes = Debug.getNativeHeapAllocatedSize()
    return (jvmBytes + nativeBytes) / (1024.0 * 1024.0)
  }

  override fun onCleared() {
    super.onCleared()
    cancelRequested.set(true)
    executor.shutdown()
  }

  companion object {
    private const val TAG = "BenchmarkViewModel"
  }
}
