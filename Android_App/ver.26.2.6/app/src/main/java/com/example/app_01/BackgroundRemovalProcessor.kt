package com.example.app_01

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.net.Uri
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector.ObjectDetectorOptions
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 배경 제거 프로세서 (MediaPipe Image Segmentation 기반, 온디바이스)
 *
 * [포인트 기반 - 삼성 Object Cut Out 방식]
 *   removeBackgroundWithPoint(): MediaPipe InteractiveSegmenter (magic_touch.tflite)
 *   - 터치한 위치의 피사체를 온디바이스 AI로 분리, 임의 객체 지원
 *
 * [텍스트 프롬프트 기반 - 1차 배경제거]
 *   removeBackground():
 *     1순위: U²-Net (u2netp.onnx, ONNX Runtime) — 카테고리 제한 없음, 어떤 사물이든 전경/배경 분리
 *     2순위: EfficientDet + InteractiveSegmenter (magic_touch.tflite)
 *     3순위: DeepLab v3 (deeplab_v3.tflite, VOC 21cls) 최후 폴백
 */
object BackgroundRemovalProcessor {

    private const val TAG = "BgRemoval"
    private const val DETECTOR_MODEL = "models/efficientdet_lite0.tflite"
    // MediaPipe 공식 DeepLab v3 (257×257 입력, Pascal VOC 21클래스)
    // 다운로드 출처: https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite
    private const val SEGMENTER_MODEL = "models/segmentation/deeplab_v3.tflite"

    // removeBackground() 진행 단계 수
    private const val PIPELINE_STEP_COUNT = 10

    // U²-Net (u2netp): 카테고리 제한 없는 두드러진 객체 배경 제거
    // 다운로드: https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
    // 입력: NCHW float32 [1,3,320,320], ImageNet 정규화
    // 출력: [1,1,320,320] 전경 확률 (0~1)
    private const val U2NET_MODEL = "models/u2net/u2netp.onnx"
    private const val U2NET_INPUT_SIZE = 320
    private val U2NET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val U2NET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    // 배경 제거 시 최대 이미지 변 길이 (OOM/크래시 방지)
    private const val MAX_IMAGE_DIMENSION = 1024

    // InteractiveSegmenter 모델 경로
    private const val INTERACTIVE_SEGMENTER_MODEL = "models/interactive_segmenter/magic_touch.tflite"

    // 포인트 기반 분리 최대 입력 크기 (2GB 메모리 한도 내 안전 처리)
    private const val ML_KIT_INPUT_MAX = 512

    // 메모리 여유량 임계값 (MB)
    private const val MEM_ABORT_THRESHOLD_MB = 200L   // 이 미만이면 처리 중단
    private const val MEM_LOW_THRESHOLD_MB = 400L     // 이 미만이면 192px 입력
    private const val MEM_MID_THRESHOLD_MB = 800L     // 이 미만이면 256px 입력
    private const val MEM_HIGH_THRESHOLD_MB = 1500L   // 이 미만이면 384px 입력, 이상이면 512px

    // Pascal VOC 21 클래스 (DeepLab v3) - 0=background, 1~20=객체
    private val VOC_LABELS = listOf(
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
        "potted plant", "sheep", "sofa", "train", "tv"
    )

    // 한/영 → VOC 인덱스 매핑 (DeepLab v3 폴백용)
    private val TEXT_TO_INDEX = mapOf(
        "사람" to 15, "person" to 15, "인물" to 15,
        "병" to 5, "bottle" to 5, "캔" to 5, "can" to 5,
        "자동차" to 7, "car" to 7, "차" to 7,
        "의자" to 9, "chair" to 9,
        "개" to 12, "dog" to 12,
        "고양이" to 8, "cat" to 8,
        "컵" to 5, "cup" to 5,
        "식물" to 16, "potted plant" to 16, "화분" to 16,
        "주요" to 15, "물건" to 15, "객체" to 15, "object" to 15,
        "" to 15
    )

    /**
     * 텍스트 프롬프트 → EfficientDet(COCO 80클래스) 레이블 매핑.
     * EfficientDet이 COCO 레이블을 반환하므로 VOC가 아닌 COCO 이름으로 매핑해야 한다.
     */
    private val TEXT_TO_COCO_LABEL = mapOf(
        // 사람
        "person" to "person", "people" to "person", "human" to "person",
        "man" to "person", "woman" to "person", "child" to "person",
        // 전자기기
        "laptop" to "laptop", "notebook" to "laptop", "computer" to "laptop",
        "mouse" to "mouse",
        "keyboard" to "keyboard",
        "tv" to "tv", "monitor" to "tv", "television" to "tv", "screen" to "tv",
        "phone" to "cell phone", "cellphone" to "cell phone", "smartphone" to "cell phone",
        "remote" to "remote", "remote control" to "remote",
        "clock" to "clock",
        "book" to "book",
        // 용기 / 식기
        "bottle" to "bottle", "can" to "bottle",
        "cup" to "cup", "mug" to "cup",
        "bowl" to "bowl",
        "glass" to "wine glass", "wine glass" to "wine glass",
        "fork" to "fork", "knife" to "knife", "spoon" to "spoon",
        "vase" to "vase",
        // 가구
        "chair" to "chair",
        "couch" to "couch", "sofa" to "couch",
        "table" to "dining table", "desk" to "dining table", "dining table" to "dining table",
        "bed" to "bed",
        "toilet" to "toilet",
        // 동물
        "cat" to "cat", "kitten" to "cat",
        "dog" to "dog", "puppy" to "dog",
        "bird" to "bird",
        "horse" to "horse",
        "cow" to "cow",
        "elephant" to "elephant",
        "bear" to "bear",
        "zebra" to "zebra",
        "giraffe" to "giraffe",
        "sheep" to "sheep",
        // 차량
        "car" to "car", "vehicle" to "car",
        "truck" to "truck",
        "bus" to "bus",
        "bicycle" to "bicycle", "bike" to "bicycle",
        "motorcycle" to "motorcycle",
        "airplane" to "airplane", "plane" to "airplane",
        "boat" to "boat", "ship" to "boat",
        // 가방 / 악세서리
        "backpack" to "backpack", "bag" to "backpack",
        "handbag" to "handbag", "purse" to "handbag",
        "umbrella" to "umbrella",
        "tie" to "tie", "suitcase" to "suitcase",
        // 스포츠 / 기타 도구
        "ball" to "sports ball",
        "skateboard" to "skateboard",
        "surfboard" to "surfboard",
        "skis" to "skis",
        "kite" to "kite",
        // 음식
        "pizza" to "pizza",
        "cake" to "cake",
        "banana" to "banana",
        "apple" to "apple",
        "sandwich" to "sandwich",
        "orange" to "orange",
        "donut" to "donut",
        // 식물
        "plant" to "potted plant", "flower" to "potted plant", "potted plant" to "potted plant",
        // 주방가전
        "sink" to "sink",
        "oven" to "oven",
        "microwave" to "microwave",
        "refrigerator" to "refrigerator", "fridge" to "refrigerator",
        "toaster" to "toaster",
        // 기타
        "scissors" to "scissors",
        "toothbrush" to "toothbrush",
        "teddy bear" to "teddy bear", "teddy" to "teddy bear",
        "bench" to "bench",
        "traffic light" to "traffic light",
        "fire hydrant" to "fire hydrant",
        "stop sign" to "stop sign",
    )

    @Volatile private var detector: ObjectDetector? = null
    @Volatile private var segmenter: ImageSegmenter? = null
    @Volatile private var interactiveSegmenter: InteractiveSegmenter? = null
    @Volatile private var ortSession: OrtSession? = null

    fun close() {
        try {
            detector?.close()
            segmenter?.close()
            interactiveSegmenter?.close()
            ortSession?.close()
        } catch (_: Exception) {
        } finally {
            detector = null
            segmenter = null
            interactiveSegmenter = null
            ortSession = null
        }
    }

    // ── 메모리 관리 ───────────────────────────────────────────────────────────

    /** 현재 사용 가능한 시스템 메모리(MB) 반환 */
    private fun getAvailableMemoryMb(context: Context): Long {
        val mi = ActivityManager.MemoryInfo()
        (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
        val availMb = mi.availMem / 1_048_576L
        val totalMb = mi.totalMem / 1_048_576L
        Log.d(TAG, "메모리 여유=${availMb}MB / 전체=${totalMb}MB")
        return availMb
    }

    /**
     * 메모리 여유량에 따른 처리 입력 크기 반환. null이면 메모리 부족으로 처리 불가.
     * 2GB 메모리 한도 관리: 여유 메모리가 적을수록 입력 해상도를 낮춰 처리.
     */
    private fun checkMemoryAndGetInputSize(context: Context): Int? {
        val availMb = getAvailableMemoryMb(context)
        return when {
            availMb < MEM_ABORT_THRESHOLD_MB -> null   // 처리 불가
            availMb < MEM_LOW_THRESHOLD_MB   -> 256    // 최소 256px (192px는 모델 정확도 저하)
            availMb < MEM_MID_THRESHOLD_MB   -> 320
            availMb < MEM_HIGH_THRESHOLD_MB  -> 416
            else                             -> ML_KIT_INPUT_MAX  // 512
        }
    }

    // ── InteractiveSegmenter (magic_touch) ────────────────────────────────────

    private fun hasInteractiveSegmenterModel(context: Context): Boolean {
        return try { context.assets.open(INTERACTIVE_SEGMENTER_MODEL).use { true } }
        catch (_: Exception) { false }
    }

    private fun getInteractiveSegmenter(context: Context): InteractiveSegmenter? {
        interactiveSegmenter?.let { return it }
        if (!hasInteractiveSegmenterModel(context)) return null
        return try {
            // magic_touch.tflite는 float32 신뢰도 마스크(0~1)를 출력
            val opts = InteractiveSegmenter.InteractiveSegmenterOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(INTERACTIVE_SEGMENTER_MODEL).build())
                .setOutputCategoryMask(false)
                .setOutputConfidenceMasks(true)
                .build()
            InteractiveSegmenter.createFromOptions(context, opts).also { interactiveSegmenter = it }
        } catch (e: Exception) {
            Log.e(TAG, "InteractiveSegmenter 초기화 실패", e)
            null
        }
    }

    /**
     * magic_touch 결과에서 전경 신뢰도 마스크(float32)를 FloatArray로 추출.
     * masks[0]=배경, masks[1]=전경(객체). 전경 마스크(index 1)를 우선 사용.
     * ByteBufferExtractor 공식 API로 추출 후 FloatBuffer 폴백.
     */
    private fun extractConfidenceMask(
        result: ImageSegmenterResult,
        outW: Int,
        outH: Int
    ): FloatArray? {
        val masksOpt = result.confidenceMasks()
        if (!masksOpt.isPresent || masksOpt.get().isEmpty()) {
            Log.w(TAG, "confidenceMasks 없음 (categoryMask로 폴백 시도)")
            val catOpt = result.categoryMask()
            if (!catOpt.isPresent) return null
            val catImage = catOpt.get()
            val maskW = catImage.width; val maskH = catImage.height
            if (maskW <= 0 || maskH <= 0) return null
            return try {
                val buf = ByteBufferExtractor.extract(catImage)
                buf.rewind()
                val raw = ByteArray(buf.remaining().coerceAtMost(maskW * maskH))
                buf.get(raw)
                resizeFloatMask(
                    FloatArray(raw.size) { if ((raw[it].toInt() and 0xFF) > 0) 1f else 0f },
                    maskW, maskH, outW, outH
                )
            } catch (e: Exception) {
                Log.e(TAG, "categoryMask 추출 실패", e)
                null
            }
        }

        val masks = masksOpt.get()
        // masks[1] = 전경(객체) 확률, masks[0] = 배경 확률
        // 전경 마스크가 있으면 index 1 사용, 없으면 index 0 사용
        val fgIdx = if (masks.size > 1) 1 else 0
        val confImage = masks[fgIdx]
        val maskW = confImage.width
        val maskH = confImage.height
        Log.d(TAG, "confidence mask[${fgIdx}] size: ${maskW}x${maskH}, total masks=${masks.size}")
        if (maskW <= 0 || maskH <= 0) return null

        return extractFloatMaskFromMPImage(confImage, maskW, maskH, outW, outH, fgIdx)
    }

    /**
     * MPImage에서 float32 마스크를 추출.
     * 1차: ByteBufferExtractor (공식 API), 2차: FloatBuffer reflection 폴백.
     */
    private fun extractFloatMaskFromMPImage(
        mpImage: com.google.mediapipe.framework.image.MPImage,
        maskW: Int, maskH: Int,
        outW: Int, outH: Int,
        maskIdx: Int = 0
    ): FloatArray? {
        // 1차: ByteBufferExtractor (공식 MediaPipe API)
        return try {
            val buf = ByteBufferExtractor.extract(mpImage)
            buf.rewind()
            buf.order(ByteOrder.nativeOrder())
            val floats = FloatArray(maskW * maskH)
            when {
                buf.remaining() >= maskW * maskH * 4 -> {
                    // float32 레이아웃 (4바이트/픽셀)
                    buf.asFloatBuffer().get(floats)
                }
                buf.remaining() == maskW * maskH -> {
                    // uint8 레이아웃 (1바이트/픽셀) → 정규화
                    val raw = ByteArray(maskW * maskH)
                    buf.get(raw)
                    for (i in raw.indices) floats[i] = (raw[i].toInt() and 0xFF) / 255f
                }
                else -> {
                    Log.w(TAG, "버퍼 크기 불일치 remaining=${buf.remaining()} expected=${maskW*maskH*4}")
                    return extractFloatMaskViaReflection(mpImage, maskW, maskH, outW, outH)
                }
            }
            val maxVal = floats.maxOrNull() ?: 0f
            Log.d(TAG, "mask[$maskIdx] max=$maxVal (via ByteBufferExtractor)")
            resizeFloatMask(floats, maskW, maskH, outW, outH)
        } catch (e: Exception) {
            Log.w(TAG, "ByteBufferExtractor 실패, FloatBuffer 폴백: ${e.message}")
            extractFloatMaskViaReflection(mpImage, maskW, maskH, outW, outH)
        }
    }

    /** FloatBuffer reflection 폴백 — ByteBufferExtractor가 실패할 때만 사용 */
    private fun extractFloatMaskViaReflection(
        mpImage: com.google.mediapipe.framework.image.MPImage,
        maskW: Int, maskH: Int,
        outW: Int, outH: Int
    ): FloatArray? {
        return try {
            val allFields = mpImage.javaClass.declaredFields +
                (mpImage.javaClass.superclass?.declaredFields ?: emptyArray())
            for (field in allFields) {
                if (java.nio.FloatBuffer::class.java.isAssignableFrom(field.type)) {
                    field.isAccessible = true
                    val fb = field.get(mpImage) as? java.nio.FloatBuffer ?: continue
                    fb.rewind()
                    val floats = FloatArray(fb.remaining())
                    fb.get(floats)
                    Log.d(TAG, "FloatBuffer reflection max=${floats.maxOrNull()}")
                    return resizeFloatMask(floats, maskW, maskH, outW, outH)
                }
            }
            Log.e(TAG, "FloatBuffer reflection 실패 - 마스크 추출 불가")
            null
        } catch (e: Exception) {
            Log.e(TAG, "FloatBuffer reflection 예외", e)
            null
        }
    }

    /** Float 신뢰도 마스크를 outW×outH 로 바이리니어 보간 리사이즈 (최근접보간보다 경계가 부드러움) */
    private fun resizeFloatMask(src: FloatArray, srcW: Int, srcH: Int, outW: Int, outH: Int): FloatArray {
        if (srcW == outW && srcH == outH) return src
        val out = FloatArray(outW * outH)
        val scaleX = srcW.toFloat() / outW
        val scaleY = srcH.toFloat() / outH
        for (y in 0 until outH) {
            val sy = (y + 0.5f) * scaleY - 0.5f
            val sy0 = sy.toInt().coerceIn(0, srcH - 1)
            val sy1 = (sy0 + 1).coerceIn(0, srcH - 1)
            val fy = (sy - sy0).coerceIn(0f, 1f)
            for (x in 0 until outW) {
                val sx = (x + 0.5f) * scaleX - 0.5f
                val sx0 = sx.toInt().coerceIn(0, srcW - 1)
                val sx1 = (sx0 + 1).coerceIn(0, srcW - 1)
                val fx = (sx - sx0).coerceIn(0f, 1f)
                val v00 = src[sy0 * srcW + sx0]
                val v10 = src[sy0 * srcW + sx1]
                val v01 = src[sy1 * srcW + sx0]
                val v11 = src[sy1 * srcW + sx1]
                out[y * outW + x] = v00 * (1 - fx) * (1 - fy) +
                    v10 * fx * (1 - fy) +
                    v01 * (1 - fx) * fy +
                    v11 * fx * fy
            }
        }
        return out
    }

    /**
     * BFS 기반 홀 채우기: 마스크 경계 외부에서 flood fill 후,
     * 배경으로 연결되지 않은 내부 구멍을 전경(1.0)으로 채운다.
     */
    private fun fillMaskHoles(
        confidence: FloatArray,
        width: Int,
        height: Int,
        threshold: Float = 0.3f
    ): FloatArray {
        val result = confidence.copyOf()
        val isFg = BooleanArray(width * height) { confidence[it] >= threshold }
        val visited = BooleanArray(width * height) { isFg[it] }  // 전경은 이미 처리됨
        val queue = ArrayDeque<Int>()

        // 가장자리의 배경 픽셀을 BFS 시작점으로 추가
        for (x in 0 until width) {
            val topIdx = x
            val botIdx = (height - 1) * width + x
            if (!visited[topIdx]) { visited[topIdx] = true; queue.add(topIdx) }
            if (!visited[botIdx]) { visited[botIdx] = true; queue.add(botIdx) }
        }
        for (y in 0 until height) {
            val leftIdx = y * width
            val rightIdx = y * width + (width - 1)
            if (!visited[leftIdx]) { visited[leftIdx] = true; queue.add(leftIdx) }
            if (!visited[rightIdx]) { visited[rightIdx] = true; queue.add(rightIdx) }
        }

        // BFS: 가장자리 배경과 연결된 모든 배경 픽셀을 visited로 표시
        val dx = intArrayOf(-1, 1, 0, 0)
        val dy = intArrayOf(0, 0, -1, 1)
        while (queue.isNotEmpty()) {
            val idx = queue.removeFirst()
            val cx = idx % width
            val cy = idx / width
            for (d in 0..3) {
                val nx = cx + dx[d]
                val ny = cy + dy[d]
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue
                val nIdx = ny * width + nx
                if (!visited[nIdx]) {
                    visited[nIdx] = true
                    queue.add(nIdx)
                }
            }
        }

        // BFS로 방문하지 못한 픽셀 = 외부 배경과 단절된 내부 구멍 → 전경으로 채움
        for (i in result.indices) {
            if (!visited[i]) result[i] = 1f
        }
        return result
    }

    /**
     * Float 신뢰도 마스크를 알파 채널로 적용해 PNG 생성.
     * 시그모이드 샤프닝 이후 호출되므로 대부분 픽셀이 0 또는 1에 가까움.
     * threshold(0.45) 이하 → 완전 투명, upperBound(0.75) 이상 → 완전 불투명, 중간 → 부드러운 경계.
     */
    private fun applyConfidenceMaskToBitmap(
        bitmap: Bitmap,
        confidence: FloatArray,
        threshold: Float = 0.45f
    ): Bitmap {
        val upperBound = 0.75f
        val range = (upperBound - threshold).coerceAtLeast(0.01f)
        val w = bitmap.width; val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        for (i in pixels.indices) {
            val conf = if (i < confidence.size) confidence[i] else 0f
            val alpha = when {
                conf < threshold   -> 0
                conf >= upperBound -> 255
                else -> ((conf - threshold) / range * 255f).toInt().coerceIn(0, 255)
            }
            pixels[i] = (alpha shl 24) or (pixels[i] and 0x00FFFFFF)
        }
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
            it.setPixels(pixels, 0, w, 0, 0, w, h)
        }
    }

    /**
     * 클릭 지점 인근 영역의 평균 신뢰도 반환.
     * 단일 픽셀보다 경계 노이즈에 강인한 반전 판단에 사용.
     */
    private fun getMaskNeighborhoodAvg(
        confidence: FloatArray,
        width: Int, height: Int,
        cx: Int, cy: Int,
        radius: Int
    ): Float {
        var sum = 0f; var count = 0
        for (dy in -radius..radius) {
            for (dx in -radius..radius) {
                val nx = (cx + dx).coerceIn(0, width - 1)
                val ny = (cy + dy).coerceIn(0, height - 1)
                sum += confidence[ny * width + nx]
                count++
            }
        }
        return if (count > 0) sum / count else 0f
    }

    /**
     * 시그모이드 기반 마스크 샤프닝.
     * 경계 영역의 중간 신뢰도 값(0.3~0.7)을 0 또는 1 쪽으로 강하게 당김.
     * steepness=10 기준: 0.35→0.12, 0.45→0.31, 0.55→0.69, 0.65→0.88
     */
    private fun sharpenMask(confidence: FloatArray, steepness: Float = 10f): FloatArray {
        return FloatArray(confidence.size) { i ->
            val centered = (confidence[i] - 0.5f) * steepness
            (1.0 / (1.0 + Math.exp(-centered.toDouble()))).toFloat()
        }
    }

    /**
     * 클릭 지점과 4방향으로 연결된 전경 픽셀(threshold 이상)만 선택.
     * 클릭한 객체와 무관한 배경 픽셀이나 다른 객체를 자동으로 제거.
     * 클릭 지점이 전경이 아닌 경우 인근 전경 픽셀을 탐색해 시작점으로 사용.
     */
    private fun selectConnectedComponent(
        confidence: FloatArray,
        width: Int, height: Int,
        startX: Int, startY: Int,
        threshold: Float = 0.45f
    ): FloatArray {
        val result = FloatArray(width * height)
        val startIdx = startY * width + startX

        // 시작 픽셀이 전경이 아니면 인근에서 탐색
        val actualStart = if (confidence.getOrElse(startIdx) { 0f } >= threshold) {
            startIdx
        } else {
            findNearestForeground(confidence, width, height, startX, startY, threshold, maxRadius = width / 5)
        }
        if (actualStart < 0) {
            Log.w(TAG, "연결 컴포넌트: 전경 픽셀 없음, 원본 마스크 반환")
            // 전경을 찾지 못하면 원본 confidence 중 threshold 이상을 그대로 반환
            return FloatArray(confidence.size) { if (confidence[it] >= threshold) confidence[it] else 0f }
        }

        // BFS: 연결된 전경 픽셀만 result에 복사
        val visited = BooleanArray(width * height)
        val queue = ArrayDeque<Int>()
        visited[actualStart] = true
        result[actualStart] = confidence[actualStart]
        queue.add(actualStart)

        val dx = intArrayOf(-1, 1, 0, 0)
        val dy = intArrayOf(0, 0, -1, 1)
        while (queue.isNotEmpty()) {
            val idx = queue.removeFirst()
            val cx = idx % width; val cy = idx / width
            for (d in 0..3) {
                val nx = cx + dx[d]; val ny = cy + dy[d]
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue
                val nIdx = ny * width + nx
                if (!visited[nIdx] && confidence[nIdx] >= threshold) {
                    visited[nIdx] = true
                    result[nIdx] = confidence[nIdx]
                    queue.add(nIdx)
                }
            }
        }
        val fgPixels = result.count { it > 0 }
        Log.d(TAG, "연결 컴포넌트 선택: $fgPixels / ${result.size} 픽셀 (${fgPixels * 100 / result.size}%)")
        return result
    }

    /** selectConnectedComponent 의 시작점 탐색 헬퍼: 클릭 위치 인근에서 가장 가까운 전경 픽셀 인덱스 반환 */
    private fun findNearestForeground(
        confidence: FloatArray,
        width: Int, height: Int,
        cx: Int, cy: Int,
        threshold: Float,
        maxRadius: Int
    ): Int {
        for (r in 1..maxRadius) {
            for (dy in -r..r) {
                for (dx in -r..r) {
                    if (Math.abs(dx) != r && Math.abs(dy) != r) continue
                    val nx = (cx + dx).coerceIn(0, width - 1)
                    val ny = (cy + dy).coerceIn(0, height - 1)
                    if (confidence[ny * width + nx] >= threshold) return ny * width + nx
                }
            }
        }
        return -1
    }

    // ── ObjectDetector ─────────────────────────────────────────────────────────

    private fun getDetector(context: Context): ObjectDetector? {
        detector?.let { return it }
        return try {
            val base = BaseOptions.builder().setModelAssetPath(DETECTOR_MODEL).build()
            val opts = ObjectDetectorOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.IMAGE)
                .setMaxResults(10)
                .build()
            ObjectDetector.createFromOptions(context, opts).also { detector = it }
        } catch (e: Exception) {
            Log.e(TAG, "ObjectDetector 초기화 실패", e)
            null
        }
    }

    private fun hasSegmenterModel(context: Context): Boolean {
        return try {
            context.assets.open(SEGMENTER_MODEL).use { true }
        } catch (_: Exception) {
            false
        }
    }

    private fun getSegmenter(context: Context): ImageSegmenter? {
        segmenter?.let { return it }
        if (!hasSegmenterModel(context)) return null
        return try {
            val base = BaseOptions.builder().setModelAssetPath(SEGMENTER_MODEL).build()
            val opts = ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.IMAGE)
                .setOutputConfidenceMasks(false)
                .setOutputCategoryMask(true)
                .build()
            ImageSegmenter.createFromOptions(context, opts).also { segmenter = it }
        } catch (e: Exception) {
            Log.e(TAG, "ImageSegmenter 초기화 실패", e)
            null
        }
    }

    // ── U²-Net (ONNX Runtime) ─────────────────────────────────────────────────

    private fun hasU2NetModel(context: Context): Boolean =
        try { context.assets.open(U2NET_MODEL).use { true } } catch (_: Exception) { false }

    /**
     * ONNX Runtime 세션 초기화 (최초 1회 생성 후 캐시).
     * u2netp.onnx 는 ~4.5MB 로 assets 에서 byte 배열로 로드.
     */
    private fun getOrtSession(context: Context): OrtSession? {
        ortSession?.let { return it }
        if (!hasU2NetModel(context)) return null
        return try {
            val bytes = context.assets.open(U2NET_MODEL).use { it.readBytes() }
            val opts = OrtSession.SessionOptions()
            OrtEnvironment.getEnvironment()
                .createSession(bytes, opts)
                .also { ortSession = it }
        } catch (e: Exception) {
            Log.e(TAG, "U²-Net ONNX 세션 초기화 실패", e)
            null
        }
    }

    /**
     * U²-Net(u2netp) 추론.
     * - 입력: bitmap → 320×320 → ImageNet 정규화 → NCHW Float32 [1,3,320,320]
     * - 입력 이름: "input.1" (u2netp ONNX 모델 내부 이름)
     * - 출력: d1 [1,1,320,320] 전경 확률 → FloatArray(320×320), 민-맥스 정규화 후 반환
     * - 카테고리 제한 없이 어떤 사물이든 배경 분리 가능.
     *
     * @return 320×320 확률 배열(0~1), 실패 시 null
     */
    private fun runU2Net(context: Context, bitmap: Bitmap): FloatArray? {
        val session = getOrtSession(context) ?: return null
        val env = OrtEnvironment.getEnvironment()
        val sz = U2NET_INPUT_SIZE
        val pixCount = sz * sz

        val scaled = scaleBitmapTo(bitmap, sz)
        val needRecycle = scaled !== bitmap

        return try {
            // ── ImageNet 정규화 후 NCHW Float32 배열 생성 ────────────────────
            val data = FloatArray(3 * pixCount)
            for (y in 0 until sz) {
                for (x in 0 until sz) {
                    val px = scaled.getPixel(x, y)
                    val i  = y * sz + x
                    data[i]                = ((px shr 16 and 0xFF) / 255f - U2NET_MEAN[0]) / U2NET_STD[0]
                    data[pixCount + i]     = ((px shr 8  and 0xFF) / 255f - U2NET_MEAN[1]) / U2NET_STD[1]
                    data[2 * pixCount + i] = ((px and 0xFF)         / 255f - U2NET_MEAN[2]) / U2NET_STD[2]
                }
            }
            if (needRecycle) scaled.recycle()

            // u2netp 모델의 입력 이름은 "input.1" (레퍼런스 코드 기준)
            val inputName = if (session.inputNames.contains("input.1")) "input.1"
                            else session.inputNames.iterator().next()
            val inputShape = longArrayOf(1L, 3L, sz.toLong(), sz.toLong())
            Log.d(TAG, "U²-Net inputName=$inputName, outputs=${session.outputNames}")

            val probFlat = FloatArray(pixCount)
            OnnxTensor.createTensor(env, FloatBuffer.wrap(data), inputShape).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { result ->
                    try {
                        // 방법 1: OnnxTensor.floatBuffer 직접 읽기 (효율적)
                        val outTensor = result.get(0) as OnnxTensor
                        val buf = outTensor.floatBuffer
                        buf.get(probFlat, 0, buf.remaining().coerceAtMost(pixCount))
                    } catch (_: Exception) {
                        // 방법 2: 4D Array 파싱 (레퍼런스 코드 방식)
                        @Suppress("UNCHECKED_CAST")
                        val mask4D = result.get(0).value as Array<Array<Array<FloatArray>>>
                        val row2D = mask4D[0][0]
                        for (y in 0 until sz) for (x in 0 until sz) probFlat[y * sz + x] = row2D[y][x]
                    }
                }
            }

            // 민-맥스 정규화 → [0, 1] (rembg 와 동일한 후처리)
            val minV = probFlat.min() ?: 0f
            val maxV = probFlat.max() ?: 1f
            val range = maxV - minV
            if (range > 1e-6f) FloatArray(pixCount) { (probFlat[it] - minV) / range } else probFlat
        } catch (e: Exception) {
            Log.e(TAG, "U²-Net 추론 실패", e)
            if (needRecycle) try { scaled.recycle() } catch (_: Exception) {}
            null
        }
    }

    private fun resolveCategoryIndex(userText: String, detections: List<Pair<String, RectF>>): Int {
        val trimmed = userText.trim().lowercase()
        TEXT_TO_INDEX[trimmed]?.let { return it }
        TEXT_TO_INDEX[userText.trim()]?.let { return it }
        val match = VOC_LABELS.indexOfFirst { it.equals(trimmed, true) }
        if (match >= 0) return match
        val firstLabel = detections.firstOrNull()?.first?.lowercase()
        if (firstLabel != null) {
            val idx = VOC_LABELS.indexOfFirst { it.equals(firstLabel, true) }
            if (idx >= 0) return idx
        }
        return 15
    }

    /**
     * 텍스트 프롬프트와 EfficientDet 감지 결과를 대조해 가장 적합한 BBox를 반환.
     * COCO 레이블 직접 비교 → 부분 포함 → 면적 최대 박스 순으로 폴백.
     */
    private fun findBestDetectionBox(
        userPrompt: String,
        detections: List<Pair<String, RectF>>
    ): RectF? {
        if (detections.isEmpty()) return null
        val p = userPrompt.trim().lowercase()
        val cocoTarget = TEXT_TO_COCO_LABEL[p] ?: p

        // 1) 정확히 일치하는 COCO 레이블
        detections.firstOrNull { (lbl, _) -> lbl.lowercase() == cocoTarget }
            ?.let { return it.second }
        // 2) 부분 포함 (예: "sports ball" ↔ "ball")
        detections.firstOrNull { (lbl, _) ->
            lbl.lowercase().contains(cocoTarget) || cocoTarget.contains(lbl.lowercase())
        }?.let { return it.second }
        // 3) 사용자 입력이 감지 레이블에 포함
        detections.firstOrNull { (lbl, _) -> lbl.lowercase().contains(p) }
            ?.let { return it.second }
        // 4) 어떤 것도 매칭 안 되면 가장 큰 박스(주요 피사체)
        return detections.maxByOrNull { (_, box) ->
            (box.right - box.left) * (box.bottom - box.top)
        }?.second
    }

    /** ObjectDetector 결과에서 사용자 프롬프트에 맞는 bbox 중심점(normalized 0~1) 반환 */
    private fun getForegroundPoint(
        userPrompt: String,
        detections: List<Pair<String, RectF>>
    ): Pair<Float, Float> {
        val box = findBestDetectionBox(userPrompt, detections)
        return if (box != null) {
            ((box.left + box.right) / 2f) to ((box.top + box.bottom) / 2f)
        } else {
            0.5f to 0.5f
        }
    }

    /**
     * 마스크가 의미 없는 trivial 결과인지 검사.
     * 전경 비율이 [minRatio, maxRatio] 범위를 벗어나면 trivial로 판단.
     * - 비율 < 1%  → 배경만 남음 (객체 미검출)
     * - 비율 > 95% → 배경이 거의 제거 안 됨 (입력과 동일)
     */
    private fun isMaskTrivial(
        mask: ByteArray,
        minRatio: Float = 0.01f,
        maxRatio: Float = 0.95f
    ): Boolean {
        if (mask.isEmpty()) return true
        val fg = mask.count { it.toInt() != 0 }
        val ratio = fg.toFloat() / mask.size
        return ratio < minRatio || ratio > maxRatio
    }

    /** FloatArray 신뢰도 마스크의 전경 비율 계산 (threshold 기준) */
    private fun maskForegroundRatio(confidence: FloatArray, threshold: Float = 0.45f): Float {
        if (confidence.isEmpty()) return 0f
        return confidence.count { it >= threshold }.toFloat() / confidence.size
    }

    /**
     * 신뢰도 마스크를 분리 가능한 박스 커널(horizontal → vertical)로 팽창(Dilation).
     * O(W×H) 성능. 얇은 관절/무기 연결부를 두껍게 만들어 BFS 연결성 확보에 사용.
     *
     * @param radius  팽창 반경(px). 이미지 단변의 ~2.5% 권장.
     * @param threshold 소스 마스크를 이진화할 기준값.
     * @return 팽창된 이진 마스크 (0f 또는 1f)
     */
    private fun dilateBinaryMask(
        confidence: FloatArray,
        width: Int,
        height: Int,
        radius: Int,
        threshold: Float = 0.32f
    ): FloatArray {
        if (radius <= 0 || confidence.isEmpty()) {
            return FloatArray(confidence.size) { if (confidence[it] >= threshold) 1f else 0f }
        }
        val src = BooleanArray(width * height) { confidence[it] >= threshold }
        val tmp = BooleanArray(width * height)
        val dst = BooleanArray(width * height)

        // ── 1단계: 수평 팽창 (sliding window count) ─────────────────────────────
        for (y in 0 until height) {
            var count = 0
            // 초기 윈도우 채우기: x=0 기준 [-radius, radius] 범위
            for (dx in 0..radius) {
                if (dx < width && src[y * width + dx]) count++
            }
            for (x in 0 until width) {
                tmp[y * width + x] = count > 0
                // 다음 픽셀로 이동: 오른쪽 새 픽셀 추가, 왼쪽 오래된 픽셀 제거
                val addX = x + radius + 1
                if (addX < width && src[y * width + addX]) count++
                val remX = x - radius
                if (remX >= 0 && src[y * width + remX]) count--
            }
        }

        // ── 2단계: 수직 팽창 ────────────────────────────────────────────────────
        for (x in 0 until width) {
            var count = 0
            for (dy in 0..radius) {
                if (dy < height && tmp[dy * width + x]) count++
            }
            for (y in 0 until height) {
                dst[y * width + x] = count > 0
                val addY = y + radius + 1
                if (addY < height && tmp[addY * width + x]) count++
                val remY = y - radius
                if (remY >= 0 && tmp[remY * width + x]) count--
            }
        }

        return FloatArray(width * height) { if (dst[it]) 1f else 0f }
    }

    sealed class Result {
        data class Success(val bitmap: Bitmap, val savedUri: Uri?) : Result()
        data class Error(val message: String) : Result()
    }

    /**
     * 삼성 Object Cut Out 방식 구현:
     *   MediaPipe InteractiveSegmenter (magic_touch.tflite) 사용.
     *   - 네이티브 크래시 없는 안정적인 MediaPipe 기반 처리
     *   - 메모리 여유량에 따라 입력 해상도 동적 조정 (192~512px)
     *   - 2GB 메모리 한도: 여유 메모리 200MB 미만 시 처리 중단
     */
    fun removeBackgroundWithPoint(
        context: Context,
        sourceBitmap: Bitmap,
        normX: Float,
        normY: Float,
        outputDir: File? = null,
        onProgress: (iteration: Int, totalIterations: Int) -> Unit = { _, _ -> }
    ): Result {
        return try {
            // 메모리 여유량 확인 및 입력 크기 결정 (2GB 한도 관리)
            val inputSize = checkMemoryAndGetInputSize(context)
                ?: return Result.Error(
                    "시스템 메모리 부족 (여유 메모리 ${MEM_ABORT_THRESHOLD_MB}MB 미만).\n" +
                    "다른 앱을 종료 후 다시 시도해 주세요."
                )

            // 소스 비트맵을 메모리에 맞는 크기로 축소
            val raw = scaleDownIfNeeded(sourceBitmap)
            val bitmap = scaleBitmapTo(raw, inputSize)
            val needRecycleRaw = raw !== sourceBitmap
            val needRecycleBitmap = bitmap !== raw

            onProgress(1, 3)

            // InteractiveSegmenter로 터치 위치 피사체 분리
            val iSeg = getInteractiveSegmenter(context)
                ?: return run {
                    if (needRecycleBitmap) bitmap.recycle()
                    if (needRecycleRaw) raw.recycle()
                    Result.Error(
                        "InteractiveSegmenter 모델을 로드할 수 없습니다.\n" +
                        "assets/models/interactive_segmenter/magic_touch.tflite 확인 필요"
                    )
                }

            val mpImage = BitmapImageBuilder(bitmap).build()
            val roi = InteractiveSegmenter.RegionOfInterest.create(NormalizedKeypoint.create(normX, normY))

            onProgress(2, 3)

            val segResult = iSeg.segment(mpImage, roi)

            // magic_touch: float32 confidence mask(0.0~1.0) 추출
            val rawConfidence = extractConfidenceMask(segResult, bitmap.width, bitmap.height)
            Log.d(TAG, "confidence mask: ${rawConfidence?.size}, max=${rawConfidence?.maxOrNull()}")

            onProgress(3, 3)

            if (rawConfidence == null || (rawConfidence.maxOrNull() ?: 0f) < 0.1f) {
                if (needRecycleBitmap) bitmap.recycle()
                if (needRecycleRaw) raw.recycle()
                return Result.Error("이 위치에서 피사체를 찾을 수 없습니다.\n객체 위를 길게 눌러 보세요.")
            }

            // ── 클릭 좌표 및 인근 평균 신뢰도 계산 ─────────────────────────────
            val clickPixX = (normX * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
            val clickPixY = (normY * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
            val neighborRadius = (minOf(bitmap.width, bitmap.height) * 0.04f).toInt().coerceAtLeast(2)
            val clickConf = getMaskNeighborhoodAvg(
                rawConfidence, bitmap.width, bitmap.height, clickPixX, clickPixY, neighborRadius
            )
            // 마스크 전체 평균 계산 (상대 비교로 절대값 0.5 의존 탈피)
            val maskMean = rawConfidence.average().toFloat()
            Log.d(TAG, "클릭($clickPixX,$clickPixY) 인근평균=$clickConf 마스크평균=$maskMean")

            // ── 적응형 마스크 반전 ──────────────────────────────────────────────
            // 클릭 위치 신뢰도가 마스크 전체 평균보다 낮으면 반전.
            // 절대값(0.5) 비교 대신 상대 비교를 사용해 모델 출력 분포가 달라도 안정적 동작.
            val afterInversion = if (clickConf < maskMean) {
                Log.d(TAG, "마스크 반전 (클릭=$clickConf < 평균=$maskMean)")
                FloatArray(rawConfidence.size) { 1f - rawConfidence[it] }
            } else {
                rawConfidence
            }

            // ── 형태학적 팽창 → BFS 연결 컴포넌트 → 원본 신뢰도 복원 ─────────
            // 팽창으로 얇은 관절·무기 연결부를 두껍게 만들어 BFS 연결성 확보.
            // BFS 후 원본 신뢰도 값을 복원해 경계선 품질은 그대로 유지.
            val dilRadius = (minOf(bitmap.width, bitmap.height) * 0.025f).toInt().coerceIn(4, 14)
            val dilated = dilateBinaryMask(afterInversion, bitmap.width, bitmap.height, dilRadius)
            val componentDilated = selectConnectedComponent(
                dilated, bitmap.width, bitmap.height, clickPixX, clickPixY, threshold = 0.5f
            )
            // 팽창된 컴포넌트 마스크 범위 안에서 원본 신뢰도 복원 (경계선 선명도 유지)
            val recovered = FloatArray(afterInversion.size) { i ->
                if (componentDilated[i] >= 0.5f) afterInversion[i] else 0f
            }

            // ── 시그모이드 샤프닝: 복원된 신뢰도를 0/1 로 수렴 ───────────────
            val sharpened = sharpenMask(recovered, steepness = 10f)

            // ── BFS 홀 채우기: 연결 컴포넌트 내부 구멍 제거 ──────────────────
            val filledConfidence = fillMaskHoles(sharpened, bitmap.width, bitmap.height, threshold = 0.5f)

            // ── 고해상도 출력: 원본(raw) 크기로 마스크 업스케일 후 적용 ──────
            val outputBitmap = if (raw !== bitmap) {
                val upscaled = resizeFloatMask(
                    filledConfidence, bitmap.width, bitmap.height, raw.width, raw.height
                )
                applyConfidenceMaskToBitmap(raw, upscaled)
            } else {
                applyConfidenceMaskToBitmap(bitmap, filledConfidence)
            }

            if (needRecycleBitmap) bitmap.recycle()
            if (needRecycleRaw) raw.recycle()

            var savedUri: Uri? = null
            outputDir?.let { dir ->
                dir.mkdirs()
                val file = File(dir, "object_cutout_${System.currentTimeMillis()}.png")
                FileOutputStream(file).use { fos ->
                    outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
                }
                savedUri = Uri.fromFile(file)
            }
            Result.Success(outputBitmap, savedUri)
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM during object extraction", e)
            Result.Error("메모리 부족으로 처리할 수 없습니다.\n다른 앱을 종료 후 다시 시도해 주세요.")
        } catch (e: Exception) {
            Log.e(TAG, "객체 분리 실패", e)
            Result.Error("처리 중 오류: ${e.message ?: "알 수 없는 오류"}")
        }
    }

    /** InteractiveSegmenter 모델 사전 로드 (화면 진입 시 호출) */
    fun warmUpSubjectSegmentation(context: Context) {
        try {
            getInteractiveSegmenter(context)
            Log.d(TAG, "InteractiveSegmenter warm-up 완료")
        } catch (e: Exception) {
            Log.d(TAG, "InteractiveSegmenter warm-up: ${e.message}")
        }
    }


    fun removeBackground(
        context: Context,
        sourceBitmap: Bitmap,
        userPrompt: String,
        outputDir: File? = null,
        onProgress: (iteration: Int, totalIterations: Int) -> Unit = { _, _ -> }
    ): Result {
        // OOM 방지: 큰 이미지는 축소
        val bitmap = scaleDownIfNeeded(sourceBitmap)
        val needRecycleSource = bitmap != sourceBitmap

        // ── 0단계: U²-Net (u2netp) ─ 최우선순위 ──────────────────────────────
        // 카테고리 제한 없이 어떤 사물이든 전경/배경 이진 분리.
        // DeepLab(VOC 21클래스)·InteractiveSegmenter(단일 포인트 한계) 문제 해결.
        onProgress(1, PIPELINE_STEP_COUNT)
        try {
            val u2netProb = runU2Net(context, bitmap)
            if (u2netProb != null) {
                onProgress(5, PIPELINE_STEP_COUNT)
                val u2netMask = resizeFloatMask(
                    u2netProb, U2NET_INPUT_SIZE, U2NET_INPUT_SIZE, bitmap.width, bitmap.height
                )
                val fgRatio = maskForegroundRatio(u2netMask, threshold = 0.45f)
                Log.d(TAG, "U²-Net 전경 비율: $fgRatio")

                if (fgRatio in 0.01f..0.97f) {
                    val outputBitmap = applyConfidenceMaskToBitmap(bitmap, u2netMask)
                    var savedUri: Uri? = null
                    outputDir?.let { dir ->
                        dir.mkdirs()
                        val file = File(dir, "bg_removed_${System.currentTimeMillis()}.png")
                        FileOutputStream(file).use { fos ->
                            outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
                        }
                        savedUri = Uri.fromFile(file)
                    }
                    if (needRecycleSource) bitmap.recycle()
                    onProgress(PIPELINE_STEP_COUNT, PIPELINE_STEP_COUNT)
                    return Result.Success(outputBitmap, savedUri)
                } else {
                    Log.w(TAG, "U²-Net trivial 마스크(비율=$fgRatio), 폴백 파이프라인 시도")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "U²-Net 실패, 폴백 파이프라인으로 전환", e)
        }

        // ── 1단계: EfficientDet으로 객체 탐지 (COCO 80클래스) ─────────────────
        val detections = mutableListOf<Pair<String, RectF>>()
        try {
            getDetector(context)?.let { det ->
                val result = det.detect(BitmapImageBuilder(bitmap).build())
                result.detections()?.forEach { d ->
                    d.categories().firstOrNull()?.let { cat ->
                        d.boundingBox()?.let { box ->
                            detections.add(cat.categoryName() to box)
                        }
                    }
                }
                Log.d(TAG, "탐지 결과 ${detections.size}개: ${detections.map { it.first }}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "ObjectDetector 실행 실패", e)
        }

        // ── 2단계: 텍스트 → 목표 포인트(normalized 0~1) 결정 ──────────────────
        val (normX, normY) = getForegroundPoint(userPrompt, detections)
        val detectedBox = findBestDetectionBox(userPrompt, detections)
        Log.d(TAG, "목표 포인트: ($normX, $normY) | 박스: $detectedBox | 프롬프트: '$userPrompt'")

        // ── 3단계: MediaPipe InteractiveSegmenter (magic_touch.tflite) ─ 2순위 ─
        // EfficientDet이 찾은 포인트를 InteractiveSegmenter에 전달.
        // 점 하나만으로 피사체 경계를 정밀 추정하며 임의 객체에 동작.
        val inputSize = checkMemoryAndGetInputSize(context)
        if (inputSize != null) {
            try {
                val iSeg = getInteractiveSegmenter(context)
                if (iSeg != null) {
                    val segBitmap = scaleBitmapTo(bitmap, inputSize)
                    val needRecycleSeg = segBitmap !== bitmap

                    val mpImage = BitmapImageBuilder(segBitmap).build()
                    val roi = InteractiveSegmenter.RegionOfInterest.create(
                        NormalizedKeypoint.create(normX, normY)
                    )
                    val segResult = iSeg.segment(mpImage, roi)
                    val rawConf = extractConfidenceMask(segResult, segBitmap.width, segBitmap.height)

                    onProgress(PIPELINE_STEP_COUNT / 3, PIPELINE_STEP_COUNT)

                    if (rawConf != null && (rawConf.maxOrNull() ?: 0f) >= 0.1f) {
                        val clickX = (normX * segBitmap.width).toInt().coerceIn(0, segBitmap.width - 1)
                        val clickY = (normY * segBitmap.height).toInt().coerceIn(0, segBitmap.height - 1)
                        val radius = (minOf(segBitmap.width, segBitmap.height) * 0.04f).toInt().coerceAtLeast(2)

                        val clickConf = getMaskNeighborhoodAvg(rawConf, segBitmap.width, segBitmap.height, clickX, clickY, radius)
                        val maskMean = rawConf.average().toFloat()
                        val oriented = if (clickConf < maskMean) {
                            Log.d(TAG, "마스크 반전 (click=$clickConf < mean=$maskMean)")
                            FloatArray(rawConf.size) { 1f - rawConf[it] }
                        } else rawConf

                        // 팽창 → BFS → 원본 신뢰도 복원 (얇은 연결부 보존)
                        val dilRadius = (minOf(segBitmap.width, segBitmap.height) * 0.025f).toInt().coerceIn(4, 14)
                        val dilated = dilateBinaryMask(oriented, segBitmap.width, segBitmap.height, dilRadius)
                        val componentDilated = selectConnectedComponent(
                            dilated, segBitmap.width, segBitmap.height, clickX, clickY, threshold = 0.5f
                        )
                        val recovered = FloatArray(oriented.size) { i ->
                            if (componentDilated[i] >= 0.5f) oriented[i] else 0f
                        }
                        val sharpened = sharpenMask(recovered, steepness = 10f)
                        val fgRatio = maskForegroundRatio(sharpened)
                        Log.d(TAG, "InteractiveSegmenter 전경 비율: $fgRatio (dilRadius=$dilRadius)")

                        if (fgRatio in 0.005f..0.97f) {
                            val filled = fillMaskHoles(sharpened, segBitmap.width, segBitmap.height)

                            val outputBitmap = if (needRecycleSeg) {
                                val upscaled = resizeFloatMask(filled, segBitmap.width, segBitmap.height, bitmap.width, bitmap.height)
                                applyConfidenceMaskToBitmap(bitmap, upscaled)
                            } else {
                                applyConfidenceMaskToBitmap(bitmap, filled)
                            }

                            if (needRecycleSeg) segBitmap.recycle()
                            onProgress(PIPELINE_STEP_COUNT * 2 / 3, PIPELINE_STEP_COUNT)

                            var savedUri: Uri? = null
                            outputDir?.let { dir ->
                                dir.mkdirs()
                                val file = File(dir, "bg_removed_${System.currentTimeMillis()}.png")
                                FileOutputStream(file).use { fos ->
                                    outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
                                }
                                savedUri = Uri.fromFile(file)
                            }
                            if (needRecycleSource) bitmap.recycle()
                            onProgress(PIPELINE_STEP_COUNT, PIPELINE_STEP_COUNT)
                            return Result.Success(outputBitmap, savedUri)
                        } else {
                            Log.w(TAG, "InteractiveSegmenter trivial 마스크(비율=$fgRatio), DeepLab v3 시도")
                        }
                    } else {
                        Log.w(TAG, "InteractiveSegmenter 마스크 없음, DeepLab v3 시도")
                    }
                    if (needRecycleSeg) segBitmap.recycle()
                }
            } catch (e: Exception) {
                Log.w(TAG, "InteractiveSegmenter 실패, DeepLab v3 시도", e)
            }
        }

        // ── 4단계: MediaPipe DeepLab v3 (ImageSegmenter) ─ 최후 폴백 ──────────
        // Pascal VOC 21클래스 카테고리 마스크.
        // U²-Net·InteractiveSegmenter 모두 실패했을 때만 사용.
        onProgress(PIPELINE_STEP_COUNT, PIPELINE_STEP_COUNT)
        val seg = getSegmenter(context) ?: run {
            if (needRecycleSource) bitmap.recycle()
            return Result.Error(
                "MediaPipe DeepLab v3 모델을 로드할 수 없습니다.\n" +
                "assets/models/segmentation/deeplab_v3.tflite 파일을 확인해 주세요."
            )
        }

        val targetIndex = resolveCategoryIndex(userPrompt, detections)
        Log.d(TAG, "DeepLab v3 대상 클래스: $targetIndex (${VOC_LABELS.getOrNull(targetIndex)})")
        return try {
            val segResult = seg.segment(BitmapImageBuilder(bitmap).build())
            val mask = extractCategoryMaskFromResult(segResult, targetIndex, bitmap.width, bitmap.height)
                ?: run {
                    if (needRecycleSource) bitmap.recycle()
                    return Result.Error(
                        "DeepLab v3 마스크 생성에 실패했습니다.\n" +
                        "해당 객체가 이미지에서 인식되지 않았습니다."
                    )
                }

            if (isMaskTrivial(mask)) {
                if (needRecycleSource) bitmap.recycle()
                return Result.Error(
                    "객체를 찾지 못했습니다.\n" +
                    "DeepLab v3는 VOC 21클래스(사람, 동물, 차량, 가구 등)만 지원합니다.\n" +
                    "프롬프트 예: person, cat, dog, car, bottle, chair"
                )
            }

            val outputBitmap = applyMaskToBitmap(bitmap, mask)
            var savedUri: Uri? = null
            outputDir?.let { dir ->
                dir.mkdirs()
                val file = File(dir, "bg_removed_${System.currentTimeMillis()}.png")
                FileOutputStream(file).use { fos ->
                    outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
                }
                savedUri = Uri.fromFile(file)
            }
            if (needRecycleSource) bitmap.recycle()
            Result.Success(outputBitmap, savedUri)
        } catch (e: Exception) {
            Log.e(TAG, "DeepLab 배경 제거 실패", e)
            if (needRecycleSource) try { bitmap.recycle() } catch (_: Exception) { }
            Result.Error("배경 제거 중 오류: ${e.message}")
        }
    }

    /** OOM 방지: 최대 변 길이 초과 시 축소 */
    private fun scaleDownIfNeeded(bitmap: Bitmap): Bitmap {
        return scaleBitmapTo(bitmap, MAX_IMAGE_DIMENSION)
    }

    /** 최대 변 길이가 maxDim 이하가 되도록 축소. 이미 작으면 동일 인스턴스 반환 */
    private fun scaleBitmapTo(bitmap: Bitmap, maxDim: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w <= maxDim && h <= maxDim) return bitmap
        val scale = maxDim.toFloat() / maxOf(w, h)
        val newW = (w * scale).toInt().coerceAtLeast(1)
        val newH = (h * scale).toInt().coerceAtLeast(1)
        Log.i(TAG, "이미지 축소: ${w}x${h} -> ${newW}x${newH}")
        return Bitmap.createScaledBitmap(bitmap, newW, newH, true)
    }

    private fun createResizedMask(
        rawMask: ByteArray,
        maskW: Int,
        maskH: Int,
        outW: Int,
        outH: Int,
        targetCategory: Int = -1,
        threshold: Int = 128
    ): ByteArray {
        val out = ByteArray(outW * outH)
        val useThreshold = targetCategory < 0
        for (y in 0 until outH) {
            for (x in 0 until outW) {
                val mx = (x * maskW) / outW
                val my = (y * maskH) / outH
                val idx = (my * maskW + mx).coerceIn(0, rawMask.size - 1)
                val value = rawMask[idx].toInt() and 0xFF
                val hit = if (useThreshold) value >= threshold else value == targetCategory
                out[y * outW + x] = if (hit) 1 else 0
            }
        }
        return out
    }

    private fun extractCategoryMaskFromResult(
        result: ImageSegmenterResult,
        targetCategory: Int,
        width: Int,
        height: Int
    ): ByteArray? {
        val maskOpt = result.categoryMask()
        if (!maskOpt.isPresent) return null
        val maskImage = maskOpt.get()
        val maskW = maskImage.width
        val maskH = maskImage.height
        if (maskW <= 0 || maskH <= 0) return null

        val rawMask = ByteArray(maskW * maskH)
        val buffer = try {
            ByteBufferExtractor.extract(maskImage)
        } catch (e: Exception) {
            getByteBufferFromMPImage(maskImage)
        } ?: return null
        try {
            buffer.rewind()
            buffer.get(rawMask)
        } catch (e: Exception) {
            Log.e(TAG, "마스크 버퍼 읽기 실패", e)
            return null
        }
        return createResizedMask(rawMask, maskW, maskH, width, height, targetCategory = targetCategory, threshold = -1)
    }

    private fun getByteBufferFromMPImage(mpImage: com.google.mediapipe.framework.image.MPImage): ByteBuffer? {
        return try {
            for (field in mpImage.javaClass.declaredFields) {
                if (ByteBuffer::class.java.isAssignableFrom(field.type)) {
                    field.isAccessible = true
                    (field.get(mpImage) as? ByteBuffer)?.let { return it }
                }
            }
            mpImage.javaClass.superclass?.let { superClass ->
                for (field in superClass.declaredFields) {
                    if (ByteBuffer::class.java.isAssignableFrom(field.type)) {
                        field.isAccessible = true
                        (field.get(mpImage) as? ByteBuffer)?.let { return it }
                    }
                }
            }
            null
        } catch (e: Exception) {
            Log.e(TAG, "MPImage ByteBuffer 추출 실패", e)
            null
        }
    }

    private fun applyMaskToBitmap(bitmap: Bitmap, mask: ByteArray): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        for (i in pixels.indices) {
            val a = if (i < mask.size && mask[i].toInt() != 0) 0xFF else 0x00
            pixels[i] = (a shl 24) or (pixels[i] and 0x00FFFFFF)
        }
        out.setPixels(pixels, 0, w, 0, 0, w, h)
        return out
    }
}
