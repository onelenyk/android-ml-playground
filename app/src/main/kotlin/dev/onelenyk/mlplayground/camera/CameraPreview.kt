package dev.onelenyk.mlplayground.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executors

/** Converts a CameraX RGBA frame to an upright ARGB_8888 bitmap. */
fun ImageProxy.toUprightBitmap(): Bitmap {
  val raw = toBitmap()
  val rotation = imageInfo.rotationDegrees
  if (rotation == 0) return raw
  val m = Matrix().apply { postRotate(rotation.toFloat()) }
  return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true).also { raw.recycle() }
}

/**
 * Camera preview + ImageAnalysis. [onFrame] is invoked on a background thread with an upright
 * bitmap; it may block (e.g. while running inference) — older frames are dropped
 * (STRATEGY_KEEP_ONLY_LATEST), so there is no queue build-up.
 */
@Composable
fun CameraPreview(
  modifier: Modifier = Modifier,
  analysisEnabled: Boolean = true,
  onFrame: (Bitmap) -> Unit,
) {
  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current
  val currentOnFrame = rememberUpdatedState(onFrame)
  val currentEnabled = rememberUpdatedState(analysisEnabled)
  val previewView = remember {
    PreviewView(context).apply {
      scaleType = PreviewView.ScaleType.FILL_CENTER
      implementationMode = PreviewView.ImplementationMode.COMPATIBLE
    }
  }
  val executor = remember { Executors.newSingleThreadExecutor() }

  DisposableEffect(lifecycleOwner) {
    val providerFuture = ProcessCameraProvider.getInstance(context)
    var provider: ProcessCameraProvider? = null
    providerFuture.addListener({
      val p = providerFuture.get().also { provider = it }
      val aspect = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
      val preview = Preview.Builder().setResolutionSelector(aspect.build()).build()
        .also { it.surfaceProvider = previewView.surfaceProvider }
      val analysis = ImageAnalysis.Builder()
        .setResolutionSelector(
          aspect.setResolutionStrategy(
            ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
          ).build()
        )
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .build()
      analysis.setAnalyzer(executor) { proxy ->
        proxy.use {
          if (currentEnabled.value) currentOnFrame.value(it.toUprightBitmap())
        }
      }
      p.unbindAll()
      p.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
    }, ContextCompat.getMainExecutor(context))

    onDispose { provider?.unbindAll() }
  }
  DisposableEffect(Unit) { onDispose { executor.shutdown() } }

  AndroidView(factory = { previewView }, modifier = modifier)
}
