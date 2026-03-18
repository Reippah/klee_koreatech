package com.example.app_01

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

/**
 * MediaActionSound는 개별 볼륨 조절이 불가능해서,
 * 셔터 클릭음을 간단히 합성하여 원하는 볼륨으로 재생한다.
 *
 * - 시스템 볼륨을 건드리지 않음
 * - 짧은 "클릭" 톤(약 25ms)
 */
object SoftShutterSound {
    private const val SAMPLE_RATE = 44100
    private const val DURATION_MS = 25
    private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_OUT_MONO
    private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

    @Volatile private var audioTrack: AudioTrack? = null
    @Volatile private var pcm: ShortArray? = null
    private val lock = Any()

    fun play(volume: Float = 0.3f) {
        val v = volume.coerceIn(0f, 1f)
        val track = ensureTrack() ?: return
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                track.setVolume(v)
            } else {
                @Suppress("DEPRECATION")
                track.setStereoVolume(v, v)
            }
            // MODE_STATIC 재생을 위해 상태 리셋
            try {
                track.stop()
                @Suppress("DEPRECATION")
                track.reloadStaticData()
            } catch (_: Throwable) {
                // ignore
            }
            track.play()
        } catch (_: Throwable) {
            // ignore: 소리 재생 실패는 기능 핵심이 아님(촬영은 계속)
        }
    }

    private fun ensureTrack(): AudioTrack? {
        audioTrack?.let { return it }
        synchronized(lock) {
            audioTrack?.let { return it }
            val samples = pcm ?: synthesizeClickPcm().also { pcm = it }
            val byteCount = samples.size * 2

            val attrs = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build()

            val format = AudioFormat.Builder()
                .setSampleRate(SAMPLE_RATE)
                .setEncoding(ENCODING)
                .setChannelMask(CHANNEL_CONFIG)
                .build()

            val track = try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    AudioTrack.Builder()
                        .setAudioAttributes(attrs)
                        .setAudioFormat(format)
                        .setTransferMode(AudioTrack.MODE_STATIC)
                        .setBufferSizeInBytes(byteCount)
                        .build()
                } else {
                    @Suppress("DEPRECATION")
                    AudioTrack(
                        AudioManager.STREAM_MUSIC,
                        SAMPLE_RATE,
                        CHANNEL_CONFIG,
                        ENCODING,
                        byteCount,
                        AudioTrack.MODE_STATIC
                    )
                }
            } catch (_: Throwable) {
                return null
            }

            return try {
                val written = track.write(samples, 0, samples.size)
                if (written <= 0) {
                    track.release()
                    null
                } else {
                    // 한 번만 재생
                    track.setLoopPoints(0, samples.size, 0)
                    audioTrack = track
                    track
                }
            } catch (_: Throwable) {
                try { track.release() } catch (_: Throwable) {}
                null
            }
        }
    }

    private fun synthesizeClickPcm(): ShortArray {
        val n = (SAMPLE_RATE * (DURATION_MS / 1000f)).toInt().coerceAtLeast(64)
        val out = ShortArray(n)
        val rand = Random(7)

        // 짧은 고주파+노이즈 + 지수 감쇠 엔벨로프
        val f1 = 2200.0
        val f2 = 4800.0
        for (i in 0 until n) {
            val t = i.toDouble() / SAMPLE_RATE.toDouble()
            val env = exp(-t / 0.006) // 빠르게 감쇠
            val tone = 0.65 * sin(2.0 * PI * f1 * t) + 0.35 * sin(2.0 * PI * f2 * t)
            val noise = (rand.nextDouble() * 2.0 - 1.0) * 0.25
            val s = (tone + noise) * env
            val pcm16 = (s * 0.9 * Short.MAX_VALUE.toDouble()).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            out[i] = pcm16.toShort()
        }
        return out
    }
}

