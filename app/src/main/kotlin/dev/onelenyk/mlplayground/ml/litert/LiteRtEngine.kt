package dev.onelenyk.mlplayground.ml.litert

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.DetectionResult
import dev.onelenyk.mlplayground.ml.core.EngineType
import dev.onelenyk.mlplayground.ml.core.ObjectDetectorEngine
import dev.onelenyk.mlplayground.ml.core.nowMs
import dev.onelenyk.mlplayground.ml.yolo.TensorLayout
import dev.onelenyk.mlplayground.ml.yolo.YoloModels
import dev.onelenyk.mlplayground.ml.yolo.YoloPostprocessor
import dev.onelenyk.mlplayground.ml.yolo.YoloPreprocessor

/**
 * Option 2: LiteRT (ex-TensorFlow Lite) via the new CompiledModel API.
 * We own the whole pipeline: letterbox -> float tensor -> run -> decode -> NMS.
 */
class LiteRtEngine(
  private val context: Context,
  private val inputSize: Int = 640,
  private val coordsNormalized: Boolean = true,
) : ObjectDetectorEngine {
  override val type = EngineType.LITERT
  override var activeDelegate = DelegateType.CPU
    private set
  override var modelInfo = "-"
    private set

  private var model: CompiledModel? = null
  private var inputs: List<TensorBuffer> = emptyList()
  private var outputs: List<TensorBuffer> = emptyList()
  private lateinit var pre: YoloPreprocessor
  private var post: YoloPostprocessor? = null

  override fun initialize(delegate: DelegateType) {
    val asset = YoloModels.assetName(context, "tflite", inputSize)
      ?: error("Missing yoloe_pf_$inputSize.tflite in assets")
    val labels = YoloModels.labels(context)
    pre = YoloPreprocessor(inputSize, TensorLayout.NHWC)

    model = try {
      create(asset, delegate).also { activeDelegate = delegate }
    } catch (t: Throwable) {
      if (delegate != DelegateType.GPU) throw t
      Log.w(TAG, "GPU compile failed, falling back to CPU", t)
      create(asset, DelegateType.CPU).also { activeDelegate = DelegateType.CPU }
    }
    val m = model!!
    inputs = m.createInputBuffers()
    outputs = m.createOutputBuffers()

    // Warm-up run also tells us the output size -> layout.
    inputs[0].writeFloat(pre.input)
    m.run(inputs, outputs)
    val outSize = outputs[0].readFloat().size
    val spec = YoloModels.inferSpec(outSize, inputSize, labels.size, coordsNormalized)
    post = YoloPostprocessor(spec, labels, inputSize)
    modelInfo = "$asset · ${labels.size} cls · ${spec::class.simpleName}"
    Log.i(TAG, "Loaded $modelInfo on $activeDelegate, outSize=$outSize")
  }

  private fun create(asset: String, delegate: DelegateType): CompiledModel {
    val options = when (delegate) {
      DelegateType.GPU -> CompiledModel.Options(Accelerator.GPU).disableAutoFallbackToCpu()
      DelegateType.CPU -> CompiledModel.Options(Accelerator.CPU).apply {
        cpuOptions = CompiledModel.CpuOptions(4, null, null)
      }
    }
    return CompiledModel.create(context.assets, asset, options)
  }

  override fun detect(bitmap: Bitmap, threshold: Float): DetectionResult {
    val m = model ?: error("Not initialized")
    val p = post ?: error("Not initialized")
    val t0 = nowMs()
    val lb = pre.process(bitmap)
    inputs[0].writeFloat(pre.input)
    val t1 = nowMs()
    m.run(inputs, outputs)
    val out = outputs[0].readFloat()
    val t2 = nowMs()
    val dets = p.process(out, threshold, lb)
    val t3 = nowMs()
    return DetectionResult(dets, t1 - t0, t2 - t1, t3 - t2, bitmap.width, bitmap.height)
  }

  override fun close() {
    inputs.forEach { runCatching { it.close() } }
    outputs.forEach { runCatching { it.close() } }
    runCatching { model?.close() }
    inputs = emptyList(); outputs = emptyList(); model = null
  }

  companion object { private const val TAG = "LiteRtEngine" }
}
