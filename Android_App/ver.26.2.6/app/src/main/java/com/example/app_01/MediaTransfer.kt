package com.example.app_01

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream

data class MediaTransferResult(
    val successCount: Int,
    val failCount: Int,
    val message: String
)

private fun guessExtFromName(name: String?): String {
    val ext = name?.substringAfterLast('.', missingDelimiterValue = "")?.lowercase() ?: ""
    return when (ext) {
        "jpg", "jpeg", "png", "webp", "heic", "heif" -> ext
        else -> "jpg"
    }
}

private fun guessMimeFromExt(ext: String): String {
    return when (ext.lowercase()) {
        "png" -> "image/png"
        "webp" -> "image/webp"
        "heic" -> "image/heic"
        "heif" -> "image/heif"
        "jpeg", "jpg" -> "image/jpeg"
        else -> "image/jpeg"
    }
}

private fun resolveDisplayNameCompat(context: Context, uri: Uri): String? {
    return try {
        if (uri.scheme == "content") {
            context.contentResolver.query(
                uri,
                arrayOf(OpenableColumns.DISPLAY_NAME),
                null,
                null,
                null
            )?.use { cursor ->
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0 && cursor.moveToFirst()) cursor.getString(idx) else null
            }
        } else {
            uri.lastPathSegment?.substringAfterLast('/')
        }
    } catch (_: Exception) {
        null
    }
}

private fun isVideoUriCompat(context: Context, uri: Uri): Boolean {
    return try {
        if (uri.scheme == "content") {
            val type = context.contentResolver.getType(uri)
            type?.startsWith("video/") == true
        } else {
            uri.toString().endsWith(".mp4", ignoreCase = true)
        }
    } catch (_: Exception) {
        false
    }
}

/**
 * 앱 내부(외부파일 디렉토리)로 사진 가져오기.
 * - 저장 위치: externalFilesDir/imported/
 * - loadCapturedMedia()가 스캔하는 범위에 포함되도록 datasets 하위는 사용하지 않음
 */
suspend fun importImagesToAppLibrary(context: Context, uris: List<Uri>): MediaTransferResult {
    if (uris.isEmpty()) return MediaTransferResult(0, 0, "가져올 사진이 없습니다.")
    val resolver = context.contentResolver
    val outDir = File(context.getExternalFilesDir(null), "imported").apply { mkdirs() }

    var ok = 0
    var fail = 0
    for (u in uris) {
        try {
            val displayName = resolveDisplayNameCompat(context, u)
            val ext = guessExtFromName(displayName)
            val base = (displayName?.substringBeforeLast('.', missingDelimiterValue = "import") ?: "import")
                .replace(Regex("[^a-zA-Z0-9_\\-\\uAC00-\\uD7A3]"), "_")
                .take(40)
                .ifBlank { "import" }
            val fileName = "${base}_${System.currentTimeMillis()}.$ext"
            val outFile = File(outDir, fileName)

            val inStream = resolver.openInputStream(u)
            if (inStream == null) {
                fail++
                continue
            }
            inStream.use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            ok++
        } catch (_: Exception) {
            fail++
        }
    }

    val msg = when {
        ok > 0 && fail > 0 -> "가져오기 완료: ${ok}장 (실패 ${fail}장)"
        ok > 0 -> "가져오기 완료: ${ok}장"
        else -> "가져오기에 실패했습니다."
    }
    return MediaTransferResult(ok, fail, msg)
}

/**
 * 선택된 사진을 휴대폰 갤러리(MediaStore)에 내보내기.
 * - API 29+ : RELATIVE_PATH로 Pictures/App_01 에 저장
 */
suspend fun exportImagesToSystemGallery(context: Context, uris: List<Uri>): MediaTransferResult {
    if (uris.isEmpty()) return MediaTransferResult(0, 0, "내보낼 사진이 없습니다.")
    val resolver = context.contentResolver

    var ok = 0
    var fail = 0

    for (u in uris) {
        try {
            // 비디오는 이번 요청 범위(사진)에서 제외
            if (isVideoUriCompat(context, u)) {
                fail++
                continue
            }

            val displayName = resolveDisplayNameCompat(context, u)
            val ext = guessExtFromName(displayName)
            val mime = resolver.getType(u) ?: guessMimeFromExt(ext)

            val outName = (displayName?.substringBeforeLast('.', missingDelimiterValue = "export") ?: "export")
                .replace(Regex("[^a-zA-Z0-9_\\-\\uAC00-\\uD7A3]"), "_")
                .take(40)
                .ifBlank { "export" } + "_${System.currentTimeMillis()}.$ext"

            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, outName)
                put(MediaStore.Images.Media.MIME_TYPE, mime)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/App_01")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val collection =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
                } else {
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                }

            val outUri = resolver.insert(collection, values)
            if (outUri == null) {
                fail++
                continue
            }

            val inStream = resolver.openInputStream(u)
            val outStream = resolver.openOutputStream(outUri, "w")
            if (inStream == null || outStream == null) {
                try { resolver.delete(outUri, null, null) } catch (_: Exception) {}
                fail++
                continue
            }

            inStream.use { input ->
                outStream.use { output ->
                    input.copyTo(output)
                }
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val done = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
                resolver.update(outUri, done, null, null)
            }

            ok++
        } catch (_: SecurityException) {
            fail++
        } catch (_: Exception) {
            fail++
        }
    }

    val msg = when {
        ok > 0 && fail > 0 -> "내보내기 완료: ${ok}장 (실패 ${fail}장)"
        ok > 0 -> "내보내기 완료: ${ok}장"
        else -> "내보내기에 실패했습니다. (권한 또는 파일 문제)"
    }
    return MediaTransferResult(ok, fail, msg)
}

/**
 * 비트맵을 시스템 갤러리(MediaStore)에 저장.
 * Pictures/App_01 에 저장되어 시스템 사진 앱 등에서 확인 가능.
 */
fun saveBitmapToSystemGallery(context: Context, bitmap: Bitmap, prefix: String = "bg_removed"): Uri? {
    return try {
        val resolver = context.contentResolver
        val outName = "${prefix}_${System.currentTimeMillis()}.png"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, outName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/App_01")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }
        val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        } else {
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        }
        val outUri = resolver.insert(collection, values) ?: return null
        resolver.openOutputStream(outUri, "w")?.use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val done = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
            resolver.update(outUri, done, null, null)
        }
        outUri
    } catch (e: Exception) {
        null
    }
}

/**
 * 비트맵을 앱 갤러리(imported/)에 저장.
 * loadCapturedMedia()가 스캔하는 범위에 포함됨.
 * @return 저장된 파일의 Uri, 실패 시 null
 */
fun saveBitmapToAppGallery(context: Context, bitmap: Bitmap): Uri? {
    return try {
        val outDir = File(context.getExternalFilesDir(null), "imported").apply { mkdirs() }
        val fileName = "gemini_${System.currentTimeMillis()}.jpg"
        val outFile = File(outDir, fileName)
        FileOutputStream(outFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }
        Uri.fromFile(outFile)
    } catch (e: Exception) {
        null
    }
}

/**
 * 배경 제거된 비트맵을 앱 갤러리(cutouts/)에 PNG로 저장.
 * 투명 배경을 보존하며 loadCapturedMedia()가 스캔하는 범위에 포함됨.
 * @return 저장된 파일의 Uri, 실패 시 null
 */
fun saveCutoutToAppGallery(context: Context, bitmap: Bitmap): Uri? {
    return try {
        val outDir = File(context.getExternalFilesDir(null), "cutouts").apply { mkdirs() }
        val fileName = "cutout_${System.currentTimeMillis()}.png"
        val outFile = File(outDir, fileName)
        FileOutputStream(outFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        Uri.fromFile(outFile)
    } catch (e: Exception) {
        null
    }
}
