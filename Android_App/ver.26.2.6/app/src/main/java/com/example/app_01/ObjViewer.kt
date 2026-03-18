package com.example.app_01

import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix as GLMatrix
import android.view.MotionEvent
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.max

/**
 * PLY(포인트클라우드) -> OBJ 변환 (캐시)
 * - 현재 앱의 .ply는 faces 없이 vertex(x,y,z)만 있는 형태를 기준으로 처리합니다.
 * - OBJ는 "v"로 vertex를 기록하고, "p"로 포인트 프리미티브도 함께 기록합니다(라인 길이 제한 고려해 chunk).
 */
data class ObjConversionResult(
    val file: File? = null,
    val error: String? = null
)

fun convertPlyToObjCached(context: Context, plyFile: File): ObjConversionResult {
    if (!plyFile.exists()) return ObjConversionResult(error = "원본 PLY 파일이 존재하지 않습니다.")

    return try {
        val parsed = parsePlyPointsAny(plyFile)
        val points = parsed.points
        val count = parsed.count
        if (count <= 0) return ObjConversionResult(error = "PLY에서 vertex(점)를 0개로 인식했습니다.")

        val outDir = File(context.getExternalFilesDir(null), "models_obj").apply { mkdirs() }
        val key = "${plyFile.nameWithoutExtension}_${plyFile.lastModified()}"
        val outFile = File(outDir, "${key}.obj")

        // 캐시 hit
        if (outFile.exists() && outFile.length() > 0L) return ObjConversionResult(file = outFile)

        // 기존 캐시(같은 nameWithoutExtension) 정리(너무 많이 쌓이는 것 방지)
        outDir.listFiles { f ->
            f.isFile &&
                f.name.startsWith("${plyFile.nameWithoutExtension}_") &&
                f.name.endsWith(".obj", ignoreCase = true) &&
                f.name != outFile.name
        }?.forEach { f ->
            try { f.delete() } catch (_: Exception) {}
        }

        FileOutputStream(outFile).bufferedWriter().use { w ->
            w.appendLine("# Generated from ${plyFile.name}")
            w.appendLine("# vertices: $count")
            // vertex
            for (i in 0 until count) {
                val x = points[i * 3]
                val y = points[i * 3 + 1]
                val z = points[i * 3 + 2]
                w.append("v ")
                w.append(x.toString()); w.append(' ')
                w.append(y.toString()); w.append(' ')
                w.append(z.toString())
                w.appendLine()
            }

            // point primitive (선택적으로 기록)
            // - OBJ index는 1-based
            val chunk = 128
            var idx = 1
            while (idx <= count) {
                val end = minOf(count, idx + chunk - 1)
                w.append("p")
                for (j in idx..end) {
                    w.append(' ')
                    w.append(j.toString())
                }
                w.appendLine()
                idx = end + 1
            }
        }

        if (!outFile.exists() || outFile.length() <= 0L) {
            ObjConversionResult(error = "OBJ 파일 생성에 실패했습니다(파일이 생성되지 않았거나 크기가 0입니다).")
        } else {
            ObjConversionResult(file = outFile)
        }
    } catch (e: Exception) {
        e.printStackTrace()
        ObjConversionResult(error = "OBJ 변환 중 예외 발생: ${e.message ?: e.javaClass.simpleName}")
    }
}

private data class PlyProperty(val type: String, val name: String)

private data class PlyHeader(
    val format: String,
    val littleEndian: Boolean,
    val vertexCount: Int,
    val vertexProperties: List<PlyProperty>,
    val dataStartOffset: Long
)

private fun parsePlyHeader(raf: RandomAccessFile): PlyHeader {
    var format = ""
    var vertexCount = 0
    var inVertex = false
    val props = ArrayList<PlyProperty>(16)

    // 첫 줄은 보통 "ply"
    val first = raf.readLine() ?: throw IllegalArgumentException("PLY 헤더를 읽을 수 없습니다.")
    if (!first.trim().equals("ply", ignoreCase = true)) {
        throw IllegalArgumentException("PLY 파일이 아닙니다(첫 줄: $first)")
    }

    while (true) {
        val line = raf.readLine() ?: throw IllegalArgumentException("PLY 헤더가 끝나기 전에 파일이 종료되었습니다.")
        val t = line.trim()
        if (t.isEmpty()) continue
        if (t.startsWith("comment")) continue

        if (t.startsWith("format")) {
            format = t
            continue
        }
        if (t.startsWith("element vertex")) {
            val parts = t.split(Regex("\\s+"))
            vertexCount = parts.getOrNull(2)?.toIntOrNull() ?: 0
            inVertex = true
            continue
        }
        if (t.startsWith("element")) {
            // vertex 섹션 종료
            inVertex = false
            continue
        }
        if (t.startsWith("property") && inVertex) {
            val parts = t.split(Regex("\\s+"))
            // list property는 미지원 (point cloud 목적)
            if (parts.size >= 5 && parts[1] == "list") {
                throw IllegalArgumentException("PLY vertex에 list property가 포함되어 있어 지원하지 않습니다: $t")
            }
            val type = parts.getOrNull(1) ?: ""
            val name = parts.getOrNull(2) ?: ""
            if (type.isNotBlank() && name.isNotBlank()) props.add(PlyProperty(type, name))
            continue
        }
        if (t == "end_header") break
    }

    val f = format.lowercase()
    val little = when {
        f.contains("binary_little_endian") -> true
        f.contains("binary_big_endian") -> false
        else -> true // ascii면 의미 없음
    }
    if (format.isBlank()) {
        throw IllegalArgumentException("PLY format 라인이 없습니다.")
    }
    if (vertexCount <= 0) {
        throw IllegalArgumentException("PLY vertex count가 0입니다.")
    }
    if (props.isEmpty()) {
        throw IllegalArgumentException("PLY vertex properties를 찾지 못했습니다.")
    }

    val dataOffset = raf.filePointer
    return PlyHeader(
        format = format,
        littleEndian = little,
        vertexCount = vertexCount,
        vertexProperties = props,
        dataStartOffset = dataOffset
    )
}

private fun typeSizeBytes(type: String): Int {
    return when (type.lowercase()) {
        "char", "int8", "uchar", "uint8" -> 1
        "short", "int16", "ushort", "uint16" -> 2
        "int", "int32", "uint", "uint32", "float", "float32" -> 4
        "double", "float64" -> 8
        else -> throw IllegalArgumentException("지원하지 않는 PLY property 타입: $type")
    }
}

private fun readScalarAsFloat(buf: ByteBuffer, type: String): Float {
    return when (type.lowercase()) {
        "char", "int8" -> buf.get().toInt().toFloat()
        "uchar", "uint8" -> (buf.get().toInt() and 0xFF).toFloat()
        "short", "int16" -> buf.short.toInt().toFloat()
        "ushort", "uint16" -> (buf.short.toInt() and 0xFFFF).toFloat()
        "int", "int32" -> buf.int.toFloat()
        "uint", "uint32" -> (buf.int.toLong() and 0xFFFF_FFFFL).toFloat()
        "float", "float32" -> buf.float
        "double", "float64" -> buf.double.toFloat()
        else -> throw IllegalArgumentException("지원하지 않는 PLY property 타입: $type")
    }
}

/**
 * ASCII / binary_little_endian / binary_big_endian PLY 지원
 * - vertex 섹션의 property 목록을 기준으로 x,y,z를 추출합니다.
 */
private fun parsePlyPointsAny(file: File): PlyParseResult {
    RandomAccessFile(file, "r").use { raf ->
        val header = parsePlyHeader(raf)
        val fmtLower = header.format.lowercase()

        val props = header.vertexProperties
        val xIdx = props.indexOfFirst { it.name == "x" }
        val yIdx = props.indexOfFirst { it.name == "y" }
        val zIdx = props.indexOfFirst { it.name == "z" }
        if (xIdx < 0 || yIdx < 0 || zIdx < 0) {
            throw IllegalArgumentException("PLY vertex properties에 x/y/z가 없습니다. (format: ${header.format})")
        }

        val rawPoints = ArrayList<Float>(header.vertexCount * 3)

        if (fmtLower.contains("ascii")) {
            // dataStartOffset 이후를 라인으로 읽는다
            raf.seek(header.dataStartOffset)
            var readCount = 0
            while (readCount < header.vertexCount) {
                val line = raf.readLine() ?: break
                val t = line.trim()
                if (t.isEmpty()) continue
                val parts = t.split(Regex("\\s+"))
                if (parts.size < props.size) continue
                val x = parts.getOrNull(xIdx)?.toFloatOrNull()
                val y = parts.getOrNull(yIdx)?.toFloatOrNull()
                val z = parts.getOrNull(zIdx)?.toFloatOrNull()
                if (x != null && y != null && z != null) {
                    rawPoints.add(x); rawPoints.add(y); rawPoints.add(z)
                    readCount++
                }
            }
        } else if (fmtLower.contains("binary_little_endian") || fmtLower.contains("binary_big_endian")) {
            raf.seek(header.dataStartOffset)
            val order = if (header.littleEndian) ByteOrder.LITTLE_ENDIAN else ByteOrder.BIG_ENDIAN

            // vertex stride 계산
            val stride = props.sumOf { typeSizeBytes(it.type) }
            val row = ByteArray(stride)

            for (i in 0 until header.vertexCount) {
                raf.readFully(row)
                val bb = ByteBuffer.wrap(row).order(order)
                var x = 0f
                var y = 0f
                var z = 0f
                for ((pi, p) in props.withIndex()) {
                    val v = readScalarAsFloat(bb, p.type)
                    when (pi) {
                        xIdx -> x = v
                        yIdx -> y = v
                        zIdx -> z = v
                    }
                }
                rawPoints.add(x); rawPoints.add(y); rawPoints.add(z)
            }
        } else {
            throw IllegalArgumentException("지원하지 않는 PLY format: ${header.format}")
        }

        val count = rawPoints.size / 3
        if (count <= 0) {
            throw IllegalArgumentException("PLY에서 vertex를 읽지 못했습니다. (format: ${header.format})")
        }

        // bounds -> normalize (기존 parsePlyPoints와 동일)
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var minZ = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        var maxZ = -Float.MAX_VALUE

        for (i in 0 until count) {
            val x = rawPoints[i * 3]
            val y = rawPoints[i * 3 + 1]
            val z = rawPoints[i * 3 + 2]
            if (x < minX) minX = x
            if (y < minY) minY = y
            if (z < minZ) minZ = z
            if (x > maxX) maxX = x
            if (y > maxY) maxY = y
            if (z > maxZ) maxZ = z
        }

        val cx = (minX + maxX) / 2f
        val cy = (minY + maxY) / 2f
        val cz = (minZ + maxZ) / 2f
        val maxDim = max(maxX - minX, max(maxY - minY, maxZ - minZ))
        val half = if (maxDim > 0f) maxDim / 2f else 1f

        val points = FloatArray(count * 3)
        for (i in 0 until count) {
            val x = rawPoints[i * 3]
            val y = rawPoints[i * 3 + 1]
            val z = rawPoints[i * 3 + 2]
            points[i * 3] = (x - cx) / half
            points[i * 3 + 1] = (y - cy) / half
            points[i * 3 + 2] = (z - cz) / half
        }

        return PlyParseResult(points, count)
    }
}

data class ObjParseResult(
    val points: FloatArray,
    val count: Int
)

fun parseObjVertices(file: File): ObjParseResult? {
    if (!file.exists()) return null
    return try {
        val raw = ArrayList<Float>(1024)
        file.bufferedReader().useLines { lines ->
            lines.forEach { line ->
                val t = line.trim()
                if (t.startsWith("v ")) {
                    val parts = t.split(Regex("\\s+"))
                    if (parts.size >= 4) {
                        raw.add(parts[1].toFloatOrNull() ?: 0f)
                        raw.add(parts[2].toFloatOrNull() ?: 0f)
                        raw.add(parts[3].toFloatOrNull() ?: 0f)
                    }
                }
            }
        }
        val count = raw.size / 3
        if (count <= 0) return null
        val points = FloatArray(count * 3)
        for (i in 0 until count * 3) points[i] = raw[i]
        ObjParseResult(points, count)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

private class ObjRenderer : GLSurfaceView.Renderer {
    private var program = 0
    private var aPos = 0
    private var uMvp = 0
    private var pointBuffer: FloatBuffer? = null
    private var vertexCount = 0
    private var aspectRatio = 1f

    private var rotationX = 0f
    private var rotationY = 0f

    fun setPoints(points: FloatArray, count: Int) {
        val bb = ByteBuffer.allocateDirect(points.size * 4).order(ByteOrder.nativeOrder())
        val fb = bb.asFloatBuffer()
        fb.put(points)
        fb.position(0)
        pointBuffer = fb
        vertexCount = count
    }

    fun addRotation(dx: Float, dy: Float) {
        rotationY += dx
        rotationX += dy
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        aPos = GLES20.glGetAttribLocation(program, "aPos")
        uMvp = GLES20.glGetUniformLocation(program, "uMvp")
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        aspectRatio = if (height == 0) 1f else width.toFloat() / height.toFloat()
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val buffer = pointBuffer ?: return
        if (vertexCount <= 0) return

        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glUseProgram(program)

        val proj = FloatArray(16)
        val view = FloatArray(16)
        val model = FloatArray(16)
        val mvp = FloatArray(16)

        GLMatrix.perspectiveM(proj, 0, 45f, aspectRatio, 0.1f, 100f)
        GLMatrix.setLookAtM(view, 0, 0f, 0f, 3f, 0f, 0f, 0f, 0f, 1f, 0f)
        GLMatrix.setIdentityM(model, 0)
        GLMatrix.rotateM(model, 0, rotationX, 1f, 0f, 0f)
        GLMatrix.rotateM(model, 0, rotationY, 0f, 1f, 0f)
        GLMatrix.multiplyMM(mvp, 0, view, 0, model, 0)
        GLMatrix.multiplyMM(mvp, 0, proj, 0, mvp, 0)

        GLES20.glUniformMatrix4fv(uMvp, 1, false, mvp, 0)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 3, GLES20.GL_FLOAT, false, 0, buffer)
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, vertexCount)
        GLES20.glDisableVertexAttribArray(aPos)
    }

    private fun loadShader(type: Int, code: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, code)
        GLES20.glCompileShader(shader)
        return shader
    }

    private fun createProgram(vs: String, fs: String): Int {
        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vs)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fs)
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
        return program
    }

    companion object {
        private const val VERTEX_SHADER = """
            uniform mat4 uMvp;
            attribute vec3 aPos;
            void main() {
                gl_Position = uMvp * vec4(aPos, 1.0);
                gl_PointSize = 2.5;
            }
        """

        private const val FRAGMENT_SHADER = """
            precision mediump float;
            void main() {
                gl_FragColor = vec4(0.49, 0.83, 0.2, 1.0);
            }
        """
    }
}

class ObjSurfaceView(context: Context) : GLSurfaceView(context) {
    private val renderer = ObjRenderer()
    private var lastX = 0f
    private var lastY = 0f

    init {
        setEGLContextClientVersion(2)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun loadModel(file: File) {
        Thread {
            val parsed = parseObjVertices(file)
            if (parsed != null) {
                queueEvent {
                    renderer.setPoints(parsed.points, parsed.count)
                }
            }
        }.start()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                lastX = event.x
                lastY = event.y
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = event.x - lastX
                val dy = event.y - lastY
                renderer.addRotation(dx * 0.5f, dy * 0.5f)
                lastX = event.x
                lastY = event.y
            }
        }
        return true
    }
}

