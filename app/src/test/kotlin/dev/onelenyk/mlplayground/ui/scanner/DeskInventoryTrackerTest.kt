package dev.onelenyk.mlplayground.ui.scanner

import dev.onelenyk.mlplayground.ml.core.Detection
import dev.onelenyk.mlplayground.ml.core.NormBox
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DeskInventoryTrackerTest {

  private fun det(label: String, score: Float = 0.8f) =
    Detection(NormBox(0.1f, 0.1f, 0.3f, 0.3f), label, score)

  @Test
  fun smoothsSingleFrameFlickerAndComputesMedianCount() {
    val tracker = DeskInventoryTracker(windowSize = 6)

    // Frame 1: laptop x1, watch x1
    var items = tracker.update(listOf(det("laptop"), det("watch")))
    assertTrue(items.isEmpty()) // needs >= 2 appearances

    // Frame 2: laptop x1, watch x1, stray false-positive "apple"
    items = tracker.update(listOf(det("laptop"), det("watch"), det("apple")))
    assertEquals(listOf("laptop", "watch"), items.map { it.label })

    // Frames 3..4: laptop x1, bottle x2, watch x1 (apple never reappears)
    tracker.update(listOf(det("laptop"), det("bottle"), det("bottle"), det("watch")))
    items = tracker.update(listOf(det("laptop"), det("bottle"), det("bottle"), det("watch")))

    val byLabel = items.associateBy { it.label }
    assertTrue("apple" !in byLabel)
    assertEquals(1, byLabel["laptop"]?.count)
    assertEquals(1, byLabel["watch"]?.count)

    // Frame 5: bottle x2 -> median count should be 2
    items = tracker.update(listOf(det("laptop"), det("bottle"), det("bottle")))
    assertEquals(2, items.first { it.label == "bottle" }.count)

    tracker.reset()
    assertTrue(tracker.update(listOf(det("laptop"))).isEmpty())
  }
}
