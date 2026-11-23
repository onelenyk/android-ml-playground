package dev.onelenyk.mlplayground.ml.core

import android.content.Context
import dev.onelenyk.mlplayground.ml.litert.LiteRtEngine
import dev.onelenyk.mlplayground.ml.mediapipe.MediaPipeEngine
import dev.onelenyk.mlplayground.ml.onnx.OnnxEngine

object EngineFactory {
  fun create(context: Context, type: EngineType, inputSize: Int): ObjectDetectorEngine = when (type) {
    EngineType.MEDIAPIPE -> MediaPipeEngine(context)
    EngineType.LITERT -> LiteRtEngine(context, inputSize)
    EngineType.ONNX -> OnnxEngine(context, inputSize)
  }
}
