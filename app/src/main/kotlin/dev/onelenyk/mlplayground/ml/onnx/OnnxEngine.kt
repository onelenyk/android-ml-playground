package dev.onelenyk.mlplayground.ml.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import dev.onelenyk.mlplayground.ml.core.Assets
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.DetectionResult
import dev.onelenyk.mlplayground.ml.core.EngineType
import dev.onelenyk.mlplayground.ml.core.ObjectDetectorEngine
import dev.onelenyk.mlplayground.ml.core.nowMs
import dev.onelenyk.mlplayground.ml.yolo.TensorLayout
import dev.onelenyk.mlplayground.ml.yolo.YoloModels
import dev.onelenyk.mlplayground.ml.yolo.YoloPostprocessor
import dev.onelenyk.mlplayground.ml.yolo.YoloPreprocessor
import java.nio.FloatBuffer

/**
 * Option 3: ONNX Runtime Mobile. Same YOLO model + same pre/post code as LiteRT,
 * so the difference is purely the runtime.
 *
 * "GPU" here maps to the NNAPI execution provider (ORT has no generic Android GPU EP);
 * CPU uses the XNNPACK execution provider.
 */
class OnnxEngine(
  private val context: Context,
  private val inputSize: Int = 640,
  private val coordsNormalized: Boolean = true,
) : ObjectDetectorEngine {
  override val type = EngineType.ONNX
  override var activeDelegate = DelegateType.CPU
    private set
  override var modelInfo = "-"
    private set

  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  private var session: OrtSession? = null
  private var inputName = "images"
  private lateinit var pre: YoloPreprocessor
  private var post: YoloPostprocessor? = null
  private val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())

  override fun initialize(delegate: DelegateType) {
    val asset = YoloModels.assetName(context, "onnx", inputSize)
      ?: error("Missing yoloe_pf_$inputSize.onnx in assets")
    val labels = YoloModels.labels(context)
    pre = YoloPreprocessor(inputSize, TensorLayout.NCHW)
    val bytes = Assets.readBytes(context, asset)

    session = try {
      create(bytes, delegate).also { activeDelegate = delegate }
    } catch (t: Throwable) {
      if (delegate != DelegateType.GPU) throw t
      Log.w(TAG, "NNAPI EP failed, falling back to CPU", t)
      create(bytes, DelegateType.CPU).also { activeDelegate = DelegateType.CPU }
    }
    val s = session!!
    inputName = s.inputNames.first()

    val out = runRaw()
    val spec = YoloModels.inferSpec(out.size, inputSize, labels.size, coordsNormalized)
    post = YoloPostprocessor(spec, labels, inputSize)
    modelInfo = "$asset · ${labels.size} cls · ${spec::class.simpleName}"
    Log.i(TAG, "Loaded $modelInfo on $activeDelegate, outSize=${out.size}")
  }

  private fun create(bytes: ByteArray, delegate: DelegateType): OrtSession {
    val opts = OrtSession.SessionOptions().apply {
      setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
      when (delegate) {
        DelegateType.GPU -> addNnapi()
        DelegateType.CPU -> {
          setIntraOpNumThreads(1) // XNNPACK manages its own thread pool
          addXnnpack(mapOf("intra_op_num_threads" to "4"))
        }
      }
    }
    return env.createSession(bytes, opts)
  }

  private fun runRaw(): FloatArray {
    val s = session ?: error("Not initialized")
    OnnxTensor.createTensor(env, FloatBuffer.wrap(pre.input), shape).use { tensor ->
      s.run(mapOf(inputName to tensor)).use { result ->
        val t = result.get(0) as OnnxTensor
        val fb = t.floatBuffer
        return FloatArray(fb.remaining()).also { fb.get(it) }
      }
    }
  }

  override fun detect(bitmap: Bitmap, threshold: Float): DetectionResult {
    val p = post ?: error("Not initialized")
    val t0 = nowMs()
    val lb = pre.process(bitmap)
    val t1 = nowMs()
    val out = runRaw()
    val t2 = nowMs()
    val dets = p.process(out, threshold, lb)
    val t3 = nowMs()
    return DetectionResult(dets, t1 - t0, t2 - t1, t3 - t2, bitmap.width, bitmap.height)
  }

  override fun close() {
    runCatching { session?.close() }
    session = null
  }

  companion object { private const val TAG = "OnnxEngine" }
}
