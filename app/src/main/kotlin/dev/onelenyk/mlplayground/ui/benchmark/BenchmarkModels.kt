package dev.onelenyk.mlplayground.ui.benchmark

import android.content.Context
import android.os.Build
import dev.onelenyk.mlplayground.ml.core.DelegateType
import dev.onelenyk.mlplayground.ml.core.DetectionResult
import dev.onelenyk.mlplayground.ml.core.EngineType
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class SameFrameEntry(
  val engineType: EngineType,
  val requestedDelegate: DelegateType,
  val activeDelegate: DelegateType,
  val modelInfo: String,
  val result: DetectionResult?,
  val error: String?,
)

data class BenchmarkRow(
  val engineType: EngineType,
  val requestedDelegate: DelegateType,
  val activeDelegate: DelegateType,
  val inputSize: Int,
  val modelInfo: String,
  val initMs: Double,
  val warmupRuns: Int,
  val measuredRuns: Int,
  val meanPreMs: Double,
  val meanInfMs: Double,
  val meanPostMs: Double,
  val meanTotalMs: Double,
  val p50TotalMs: Double,
  val p95TotalMs: Double,
  val fps: Double,
  val memoryMb: Double,
  val perRunInfMs: List<Double>,
  val error: String?,
) {
  val isFallback: Boolean
    get() = requestedDelegate == DelegateType.GPU && activeDelegate == DelegateType.CPU
}

enum class ExportFormat(val ext: String, val mime: String) {
  CSV("csv", "text/csv"),
  JSON("json", "application/json"),
}

object BenchmarkExporter {
  fun toCsv(rows: List<BenchmarkRow>, timestamp: String = nowIso()): String = buildString {
    appendLine(
      "timestamp,device,android_version,sdk,engine,requested_delegate,active_delegate,fallback,input_size,model_info," +
        "init_ms,warmup_runs,measured_runs,mean_pre_ms,mean_inf_ms,mean_post_ms,mean_total_ms,p50_total_ms,p95_total_ms,fps,memory_mb,error"
    )
    for (r in rows) {
      val info = r.modelInfo.replace("\"", "\"\"")
      val err = (r.error ?: "").replace("\"", "\"\"")
      appendLine(
        listOf(
          timestamp,
          Build.MODEL,
          Build.VERSION.RELEASE,
          Build.VERSION.SDK_INT.toString(),
          r.engineType.name,
          r.requestedDelegate.name,
          r.activeDelegate.name,
          r.isFallback.toString(),
          r.inputSize.toString(),
          "\"$info\"",
          fmt(r.initMs),
          r.warmupRuns.toString(),
          r.measuredRuns.toString(),
          fmt(r.meanPreMs),
          fmt(r.meanInfMs),
          fmt(r.meanPostMs),
          fmt(r.meanTotalMs),
          fmt(r.p50TotalMs),
          fmt(r.p95TotalMs),
          fmt(r.fps),
          fmt(r.memoryMb),
          "\"$err\"",
        ).joinToString(",")
      )
    }
  }

  fun toJson(rows: List<BenchmarkRow>, timestamp: String = nowIso()): String {
    val root = JSONObject()
    root.put("timestamp", timestamp)
    root.put("device", Build.MODEL)
    root.put("hardware", Build.HARDWARE)
    root.put("androidVersion", Build.VERSION.RELEASE)
    root.put("sdkInt", Build.VERSION.SDK_INT)

    val arr = JSONArray()
    for (r in rows) {
      val obj = JSONObject()
      obj.put("engine", r.engineType.name)
      obj.put("requestedDelegate", r.requestedDelegate.name)
      obj.put("activeDelegate", r.activeDelegate.name)
      obj.put("fallback", r.isFallback)
      obj.put("inputSize", r.inputSize)
      obj.put("modelInfo", r.modelInfo)
      obj.put("initMs", round2(r.initMs))
      obj.put("warmupRuns", r.warmupRuns)
      obj.put("measuredRuns", r.measuredRuns)
      obj.put("meanPreMs", round2(r.meanPreMs))
      obj.put("meanInfMs", round2(r.meanInfMs))
      obj.put("meanPostMs", round2(r.meanPostMs))
      obj.put("meanTotalMs", round2(r.meanTotalMs))
      obj.put("p50TotalMs", round2(r.p50TotalMs))
      obj.put("p95TotalMs", round2(r.p95TotalMs))
      obj.put("fps", round2(r.fps))
      obj.put("memoryMb", round2(r.memoryMb))
      obj.put("perRunInfMs", JSONArray(r.perRunInfMs.map { round2(it) }))
      if (r.error != null) obj.put("error", r.error)
      arr.put(obj)
    }
    root.put("results", arr)
    return root.toString(2)
  }

  fun saveToExternal(context: Context, rows: List<BenchmarkRow>): Pair<File, File>? {
    val dir = context.getExternalFilesDir("benchmarks") ?: return null
    dir.mkdirs()
    val ts = nowIso()
    val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val csv = toCsv(rows, ts)
    val json = toJson(rows, ts)
    val csvFile = File(dir, "benchmark_$stamp.csv").apply { writeText(csv) }
    val jsonFile = File(dir, "benchmark_$stamp.json").apply { writeText(json) }
    File(dir, "latest.csv").writeText(csv)
    File(dir, "latest.json").writeText(json)
    return csvFile to jsonFile
  }

  fun writeShareFile(context: Context, rows: List<BenchmarkRow>, format: ExportFormat): File {
    val dir = File(context.cacheDir, "exports").apply { mkdirs() }
    val file = File(dir, "ml_playground_benchmark.${format.ext}")
    val content = when (format) {
      ExportFormat.CSV -> toCsv(rows)
      ExportFormat.JSON -> toJson(rows)
    }
    file.writeText(content)
    return file
  }

  private fun nowIso(): String =
    SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(Date())

  private fun fmt(v: Double): String = String.format(Locale.US, "%.2f", v)
  private fun round2(v: Double): Double = Math.round(v * 100.0) / 100.0
}
