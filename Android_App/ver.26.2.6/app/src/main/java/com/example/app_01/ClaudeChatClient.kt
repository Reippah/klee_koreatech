package com.example.app_01

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Claude API를 통한 채팅 클라이언트.
 * 텍스트와 이미지를 입력받아 Claude 응답(텍스트)을 반환.
 * 참고: Claude는 이미지 출력을 지원하지 않음 (텍스트 응답만)
 */
object ClaudeChatClient {
    private const val TAG = "ClaudeChat"
    private const val MODEL = "claude-sonnet-4-6"
    private const val BASE_URL = "https://api.anthropic.com/v1/messages"
    private const val TIMEOUT_SEC = 60L

    private val client = OkHttpClient.Builder()
        .connectTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .readTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .writeTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .build()

    private fun getApiKey(): String {
        return try {
            BuildConfig.CLAUDE_API_KEY?.takeIf { it.isNotBlank() } ?: ""
        } catch (e: Throwable) {
            ""
        }
    }

    fun isAvailable(): Boolean = getApiKey().isNotBlank()

    sealed class ChatResult {
        data class Success(val text: String) : ChatResult()
        data class Error(val message: String) : ChatResult()
    }

    /**
     * 텍스트와 이미지(들)를 입력받아 Claude 응답 반환.
     */
    suspend fun sendMessage(
        text: String,
        imageBase64List: List<String> = emptyList()
    ): ChatResult = withContext(Dispatchers.IO) {
        val apiKey = getApiKey()
        if (apiKey.isBlank()) {
            Log.w(TAG, "Claude API 키가 설정되지 않았습니다.")
            return@withContext ChatResult.Error("API 키가 설정되지 않았습니다. local.properties에 claude_api_key를 추가하세요.")
        }

        val contentBlocks = JSONArray()
        for (base64 in imageBase64List) {
            contentBlocks.put(JSONObject().apply {
                put("type", "image")
                put("source", JSONObject().apply {
                    put("type", "base64")
                    put("media_type", "image/jpeg")
                    put("data", base64)
                })
            })
        }
        contentBlocks.put(JSONObject().apply {
            put("type", "text")
            put("text", text.ifBlank { if (imageBase64List.isNotEmpty()) "이 이미지들에 대해 설명해 주세요." else "무엇을 도와드릴까요?" })
        })

        val messages = JSONArray().put(JSONObject().apply {
            put("role", "user")
            put("content", contentBlocks)
        })

        val body = JSONObject().apply {
            put("model", MODEL)
            put("max_tokens", 4096)
            put("messages", messages)
        }.toString()

        val request = Request.Builder()
            .url(BASE_URL)
            .addHeader("x-api-key", apiKey)
            .addHeader("anthropic-version", "2023-06-01")
            .addHeader("content-type", "application/json")
            .post(body.toRequestBody("application/json".toMediaType()))
            .build()

        try {
            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                if (!response.isSuccessful) {
                    Log.e(TAG, "API 오류: ${response.code} $responseBody")
                    val errMsg = try {
                        JSONObject(responseBody).optJSONObject("error")?.optString("message", responseBody)
                            ?: "HTTP ${response.code}: $responseBody"
                    } catch (_: Exception) {
                        "HTTP ${response.code}: ${response.message}"
                    }
                    return@withContext ChatResult.Error(errMsg)
                }
                parseTextFromResponse(responseBody)?.let { ChatResult.Success(it) }
                    ?: ChatResult.Error("응답 파싱 실패")
            }
        } catch (e: Exception) {
            Log.e(TAG, "API 호출 실패", e)
            ChatResult.Error("네트워크 오류: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    fun bitmapToBase64(bitmap: Bitmap): String {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 85, stream)
        return Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)
    }

    private fun parseTextFromResponse(jsonBody: String): String? {
        return try {
            val root = JSONObject(jsonBody)
            val content = root.optJSONArray("content") ?: return null
            val sb = StringBuilder()
            for (i in 0 until content.length()) {
                val block = content.getJSONObject(i)
                if (block.optString("type") == "text") {
                    sb.append(block.optString("text", ""))
                }
            }
            sb.toString().takeIf { it.isNotBlank() }
        } catch (e: Exception) {
            Log.e(TAG, "응답 파싱 실패", e)
            null
        }
    }
}
