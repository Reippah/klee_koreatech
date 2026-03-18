package com.example.app_01

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector.ObjectDetectorOptions
import java.util.concurrent.atomic.AtomicLong

/**
 * 사물 촬영(OBJECT) 모드에서만 사용:
 * - MediaPipe ObjectDetector로 프레임 내 사물 bounding box 추출
 * - 중앙 1000x1000 가상 사각형을 벗어났는지 판단
 */
object ObjectOutOfFrameWarning {

    private const val MODEL_ASSET_PATH = "models/efficientdet_lite0.tflite"

    // 과도한 연산 방지용 스로틀
    private val lastRunAtMs = AtomicLong(0L)

    @Volatile
    private var detector: ObjectDetector? = null

    fun close() {
        try {
            detector?.close()
        } catch (_: Exception) {
        } finally {
            detector = null
        }
    }

    private fun getDetector(context: Context): ObjectDetector? {
        detector?.let { return it }
        return try {
            val base = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET_PATH)
                .build()

            val options = ObjectDetectorOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.IMAGE)
                .setMaxResults(5)
                .build()

            ObjectDetector.createFromOptions(context, options).also { detector = it }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * @param minIntervalMs 너무 잦은 분석으로 UI/발열이 커지는 것을 방지하기 위한 최소 간격
     */
    fun shouldWarn(
        context: Context,
        bitmapUpright: Bitmap,
        focusPointNorm: Pair<Float, Float>?,
        squareSizePx: Int = 1000,
        minIntervalMs: Long = 350L
    ): Boolean? {
        val now = SystemClock.elapsedRealtime()
        val last = lastRunAtMs.get()
        // 스로틀 구간에서는 "상태 업데이트를 하지 않음" (경고 깜빡임 방지)
        if (now - last < minIntervalMs) return null
        lastRunAtMs.set(now)

        val det = getDetector(context) ?: return false

        val mpImage = BitmapImageBuilder(bitmapUpright).build()
        val result = try {
            det.detect(mpImage)
        } catch (e: Exception) {
            e.printStackTrace()
            return false
        }

        val detections = result.detections()
        if (detections.isNullOrEmpty()) return false

        val targetBox = pickTargetBox(detections.mapNotNull { it.boundingBox() }, bitmapUpright, focusPointNorm)
            ?: return false

        val w = bitmapUpright.width
        val h = bitmapUpright.height
        val size = squareSizePx.coerceAtMost(minOf(w, h)).coerceAtLeast(200)
        val left = (w - size) / 2f
        val top = (h - size) / 2f
        val square = RectF(left, top, left + size, top + size)

        // 사물 박스가 사각형을 조금이라도 벗어나면 경고
        return (targetBox.left < square.left ||
            targetBox.top < square.top ||
            targetBox.right > square.right ||
            targetBox.bottom > square.bottom)
    }

    private fun pickTargetBox(
        boxes: List<RectF>,
        bitmapUpright: Bitmap,
        focusPointNorm: Pair<Float, Float>?
    ): RectF? {
        if (boxes.isEmpty()) return null

        if (focusPointNorm != null) {
            val fx = (focusPointNorm.first.coerceIn(0f, 1f) * bitmapUpright.width)
            val fy = (focusPointNorm.second.coerceIn(0f, 1f) * bitmapUpright.height)
            val hit = boxes.firstOrNull { it.contains(fx, fy) }
            if (hit != null) return hit
        }

        // fallback: 가장 큰 박스(면적 기준)
        return boxes.maxByOrNull { (it.width() * it.height()) }
    }
}

/**
 * ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888 용 변환기.
 * - rowStride padding을 고려하여 Bitmap(ARGB_8888)으로 변환
 */
fun imageProxyRgbaToBitmap(image: ImageProxy): Bitmap? {
    return try {
        val plane = image.planes[0]
        val buffer = plane.buffer
        buffer.rewind()

        val width = image.width
        val height = image.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride // RGBA = 4

        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)

        val pixels = IntArray(width * height)
        var out = 0
        for (y in 0 until height) {
            val rowStart = y * rowStride
            for (x in 0 until width) {
                val i = rowStart + x * pixelStride
                if (i + 3 >= bytes.size) continue
                val r = bytes[i].toInt() and 0xFF
                val g = bytes[i + 1].toInt() and 0xFF
                val b = bytes[i + 2].toInt() and 0xFF
                val a = bytes[i + 3].toInt() and 0xFF
                pixels[out++] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        bmp
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

fun rotateBitmapIfNeeded(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
    if (rotationDegrees % 360 == 0) return bitmap
    val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
}

