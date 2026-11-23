package dev.onelenyk.mlplayground.ml.core

import android.content.Context
import android.content.res.AssetFileDescriptor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

object Assets {
  fun exists(context: Context, name: String): Boolean =
    runCatching { context.assets.open(name).close(); true }.getOrDefault(false)

  /** Memory-maps an uncompressed asset (see noCompress in build.gradle.kts). */
  fun mapAsset(context: Context, name: String): ByteBuffer {
    val fd: AssetFileDescriptor = context.assets.openFd(name)
    FileInputStream(fd.fileDescriptor).use { input ->
      return input.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
  }

  fun readBytes(context: Context, name: String): ByteArray = context.assets.open(name).use { it.readBytes() }

  fun readLines(context: Context, name: String): List<String> =
    context.assets.open(name).bufferedReader().useLines { seq -> seq.map { it.trim() }.toList() }
}
