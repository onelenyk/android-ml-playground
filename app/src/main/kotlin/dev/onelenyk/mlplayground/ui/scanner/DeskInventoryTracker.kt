package dev.onelenyk.mlplayground.ui.scanner

import dev.onelenyk.mlplayground.ml.core.Detection

data class InventoryItem(val label: String, val count: Int, val maxScore: Float)

/**
 * Smooths detections across a sliding window of recent frames so the live "On the desk"
 * inventory list does not flicker from single-frame drops.
 */
class DeskInventoryTracker(private val windowSize: Int = 8) {
  private data class FrameSnapshot(val counts: Map<String, Int>, val maxScores: Map<String, Float>)

  private val window = ArrayDeque<FrameSnapshot>(windowSize)

  fun update(detections: List<Detection>): List<InventoryItem> {
    val counts = HashMap<String, Int>()
    val scores = HashMap<String, Float>()
    for (d in detections) {
      counts[d.label] = (counts[d.label] ?: 0) + 1
      val prev = scores[d.label] ?: 0f
      if (d.score > prev) scores[d.label] = d.score
    }
    if (window.size >= windowSize) window.removeFirst()
    window.addLast(FrameSnapshot(counts, scores))

    val minAppearances = if (window.size >= 4) 3 else 2
    val allLabels = window.flatMapTo(LinkedHashSet()) { it.counts.keys }
    val items = ArrayList<InventoryItem>()
    for (label in allLabels) {
      val observedCounts = window.mapNotNull { it.counts[label] }.sorted()
      if (observedCounts.size < minAppearances) continue
      val medianCount = observedCounts[observedCounts.size / 2]
      val maxScore = window.maxOfOrNull { it.maxScores[label] ?: 0f } ?: 0f
      items += InventoryItem(label, medianCount, maxScore)
    }
    return items.sortedWith(
      compareByDescending<InventoryItem> { it.count }
        .thenByDescending { it.maxScore }
        .thenBy { it.label }
    )
  }

  fun reset() {
    window.clear()
  }
}
