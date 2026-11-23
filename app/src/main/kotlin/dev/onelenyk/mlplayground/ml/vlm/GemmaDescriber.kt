package dev.onelenyk.mlplayground.ml.vlm

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ThinkingConfig
import dev.onelenyk.mlplayground.ml.core.DelegateType
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

data class VlmMetrics(
  val ttftSec: Double,
  val totalSec: Double,
  val prefillTokensPerSec: Double,
  val decodeTokensPerSec: Double,
  val prefillTokens: Int,
  val decodeTokens: Int,
)

class GemmaDescriber(private val context: Context) : AutoCloseable {
  private var engine: Engine? = null
  @Volatile private var activeConversation: Conversation? = null

  var activeDelegate: DelegateType = DelegateType.GPU
    private set
  var loadedModelFile: File? = null
    private set

  val isInitialized: Boolean
    get() = engine?.isInitialized() == true

  fun initialize(modelFile: File, requestedDelegate: DelegateType): Double {
    close()
    val t0 = System.nanoTime()
    val created = try {
      createAndInit(modelFile, requestedDelegate).also { activeDelegate = requestedDelegate }
    } catch (t: Throwable) {
      if (requestedDelegate != DelegateType.GPU) throw t
      Log.w(TAG, "LiteRT-LM GPU init failed, falling back to CPU", t)
      createAndInit(modelFile, DelegateType.CPU).also { activeDelegate = DelegateType.CPU }
    }
    engine = created
    loadedModelFile = modelFile
    return (System.nanoTime() - t0) / 1_000_000_000.0
  }

  private fun createAndInit(modelFile: File, delegate: DelegateType): Engine {
    val mainBackend: Backend
    val visionBackend: Backend
    when (delegate) {
      DelegateType.GPU -> {
        mainBackend = Backend.GPU()
        visionBackend = Backend.GPU()
      }
      DelegateType.CPU -> {
        mainBackend = Backend.CPU(threadCount = 4)
        visionBackend = Backend.CPU(threadCount = 4)
      }
    }
    val config = EngineConfig(
      modelPath = modelFile.absolutePath,
      backend = mainBackend,
      visionBackend = visionBackend,
      maxNumTokens = 2048,
      cacheDir = context.cacheDir.absolutePath,
    )
    return Engine(config).apply { initialize() }
  }

  @OptIn(ExperimentalApi::class)
  suspend fun describe(
    bitmap: Bitmap,
    prompt: String,
    enableThinking: Boolean,
    onDelta: (answerDelta: String, thoughtDelta: String) -> Unit,
  ): VlmMetrics {
    val eng = engine ?: error("Модель Gemma не завантажена")
    val jpegBytes = bitmapToJpegBytes(bitmap, maxDim = 768)
    val convConfig = ConversationConfig(
      thinkingConfig = ThinkingConfig(enableThinking = enableThinking),
      maxOutputToken = 512,
    )
    val tStart = System.nanoTime()
    var firstTokenNs = 0L

    eng.createConversation(convConfig).use { conv ->
      activeConversation = conv
      try {
        val contents = Contents.of(
          Content.ImageBytes(jpegBytes),
          Content.Text(prompt),
        )
        conv.sendMessageAsync(contents).collect { message ->
          val answerChunk = message.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString("") { it.text }
          val thoughtChunk = message.channels["thought"] ?: ""
          if ((answerChunk.isNotEmpty() || thoughtChunk.isNotEmpty()) && firstTokenNs == 0L) {
            firstTokenNs = System.nanoTime()
          }
          if (answerChunk.isNotEmpty() || thoughtChunk.isNotEmpty()) {
            onDelta(answerChunk, thoughtChunk)
          }
        }
        val totalSec = (System.nanoTime() - tStart) / 1_000_000_000.0
        val wallTtft = if (firstTokenNs > 0L) (firstTokenNs - tStart) / 1_000_000_000.0 else totalSec
        val bench = runCatching { conv.getBenchmarkInfo() }.getOrNull()
        return VlmMetrics(
          ttftSec = bench?.timeToFirstTokenInSecond?.takeIf { it > 0.0 } ?: wallTtft,
          totalSec = totalSec,
          prefillTokensPerSec = bench?.lastPrefillTokensPerSecond ?: 0.0,
          decodeTokensPerSec = bench?.lastDecodeTokensPerSecond ?: 0.0,
          prefillTokens = bench?.lastPrefillTokenCount ?: 0,
          decodeTokens = bench?.lastDecodeTokenCount ?: 0,
        )
      } finally {
        activeConversation = null
      }
    }
  }

  fun cancel() {
    runCatching { activeConversation?.cancelProcess() }
  }

  override fun close() {
    cancel()
    runCatching { engine?.close() }
    engine = null
    loadedModelFile = null
  }

  private fun bitmapToJpegBytes(src: Bitmap, maxDim: Int): ByteArray {
    val w = src.width
    val h = src.height
    val scaled = if (max(w, h) > maxDim) {
      val ratio = maxDim.toFloat() / max(w, h)
      Bitmap.createScaledBitmap(src, (w * ratio).roundToInt(), (h * ratio).roundToInt(), true)
    } else {
      src
    }
    return ByteArrayOutputStream().use { out ->
      scaled.compress(Bitmap.CompressFormat.JPEG, 85, out)
      out.toByteArray()
    }
  }

  companion object {
    private const val TAG = "GemmaDescriber"
    const val MODEL_FILENAME = "gemma-4-E2B-it.litertlm"

    fun expectedExternalPath(context: Context): File =
      File(context.getExternalFilesDir("models"), MODEL_FILENAME)

    fun findModelFile(context: Context): File? {
      val extDir = context.getExternalFilesDir("models")
      val primary = if (extDir != null) File(extDir, MODEL_FILENAME) else null
      if (primary != null && primary.isFile && primary.length() > 0L) return primary

      val tmp = File("/data/local/tmp", MODEL_FILENAME)
      if (tmp.isFile && tmp.length() > 0L) return tmp

      return extDir?.listFiles()
        ?.filter { it.isFile && it.name.endsWith(".litertlm") && it.length() > 0L }
        ?.sortedBy { it.name }
        ?.firstOrNull()
    }
  }
}
