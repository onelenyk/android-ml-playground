package dev.onelenyk.mlplayground.ml.yolo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloPostprocessorTest {

  @Test
  fun compactPostprocessorDecodesUnmapsLetterboxAndAppliesClassAgnosticNms() {
    val inputSize = 640
    // Original image 480x640 (portrait): scale = min(640/480, 640/640) = 1.0, padX = 80, padY = 0
    val lb = Letterbox(scale = 1.0f, padX = 80f, padY = 0f, srcW = 480, srcH = 640)
    val labels = listOf("laptop", "notebook computer", "cat")
    val spec = YoloOutputSpec.Compact(numAnchors = 4, coordsNormalized = true)
    val post = YoloPostprocessor(spec, labels, inputSize, iouThreshold = 0.5f, classAgnosticNms = true)

    // Anchor 0: high-score laptop at center of original image:
    // orig box [120..360, 160..480] -> letterbox [200..440, 160..480] -> cx=320/640=0.5, cy=320/640=0.5, w=240/640=0.375, h=320/640=0.5
    // Anchor 1: overlapping synonym "notebook computer" with lower score (should be suppressed by class-agnostic NMS)
    // Anchor 2: below threshold (should be filtered out)
    // Anchor 3: non-overlapping cat at top-left [0..120, 0..160] -> letterbox [80..200, 0..160] -> cx=140/640, cy=80/640, w=120/640, h=160/640
    val raw = floatArrayOf(
      0.5f, 0.5f, 0.375f, 0.5f, 0.90f, 0f,
      0.51f, 0.50f, 0.375f, 0.5f, 0.70f, 1f,
      0.2f, 0.2f, 0.1f, 0.1f, 0.10f, 2f,
      140f / 640f, 80f / 640f, 120f / 640f, 160f / 640f, 0.80f, 2f,
    )

    val detections = post.process(raw, threshold = 0.25f, lb = lb)
    assertEquals(2, detections.size)

    assertEquals("laptop", detections[0].label)
    assertEquals(0.90f, detections[0].score, 1e-4f)
    assertEquals(120f / 480f, detections[0].box.left, 1e-3f)
    assertEquals(160f / 640f, detections[0].box.top, 1e-3f)
    assertEquals(360f / 480f, detections[0].box.right, 1e-3f)
    assertEquals(480f / 640f, detections[0].box.bottom, 1e-3f)

    assertEquals("cat", detections[1].label)
    assertTrue(detections[1].box.right <= 0.26f)
  }
}
