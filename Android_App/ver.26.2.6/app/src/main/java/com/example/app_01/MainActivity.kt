package com.example.app_01

import android.Manifest
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.MediaActionSound
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Surface
import org.json.JSONObject
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.SystemClock
import androidx.exifinterface.media.ExifInterface
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix as GLMatrix
import android.view.MotionEvent
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import androidx.compose.foundation.Canvas
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.AspectRatio
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import android.util.Size
import android.os.Handler
import android.os.Looper
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.CameraFront
import androidx.compose.material.icons.filled.CameraRear
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Public
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.TextButton
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import androidx.compose.material3.MaterialTheme
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.toMutableStateList
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import android.view.WindowManager
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.zIndex
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.Thermostat
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.SettingsInputAntenna
import androidx.compose.material.icons.filled.Compress
import androidx.compose.material.icons.filled.Brush
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.FileUpload
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.LockOpen
import androidx.compose.material.icons.filled.RotateLeft
import androidx.compose.material.icons.filled.Flip
import androidx.compose.material.icons.filled.Crop
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.material3.HorizontalDivider
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import kotlin.math.roundToInt
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.ui.input.pointer.PointerInputChange
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import coil.compose.rememberAsyncImagePainter
import android.widget.VideoView
import android.widget.MediaController
import android.media.MediaMetadataRetriever
import android.graphics.Bitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.example.app_01.ui.theme.App_01Theme
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.delay
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager
import javax.net.ssl.HostnameVerifier
import java.security.cert.X509Certificate

// 서버 설정
private const val UPLOAD_ENDPOINT = "/upload"
private const val STATUS_ENDPOINT = "/status"
private const val DOWNLOAD_ENDPOINT = "/download"
private const val DEFAULT_SERVER_ADDRESS = "192.168.0.88"
private const val DEFAULT_SERVER_PORT = 8000
private const val DEFAULT_USE_HTTPS = false // HTTP 기본값

// SAM3 배경제거 서버 (sam3_server.py, 기본 포트 8001)
private const val SAM3_BG_REMOVE_ENDPOINT = "/bg-remove"
private const val SAM3_DEFAULT_PORT = 8001


// SharedPreferences 키
private const val PREF_SERVER_ADDRESS = "server_address"
private const val PREF_SERVER_PORT = "server_port"
private const val PREF_USE_HTTPS = "use_https"

// SharedPreferences에서 서버 주소 가져오기
private fun getServerAddress(context: Context): String {
    val prefs = context.getSharedPreferences("app_settings", Context.MODE_PRIVATE)
    return prefs.getString(PREF_SERVER_ADDRESS, DEFAULT_SERVER_ADDRESS) ?: DEFAULT_SERVER_ADDRESS
}

// SharedPreferences에서 서버 포트 가져오기
private fun getServerPort(context: Context): Int {
    val prefs = context.getSharedPreferences("app_settings", Context.MODE_PRIVATE)
    return prefs.getInt(PREF_SERVER_PORT, DEFAULT_SERVER_PORT)
}

// SharedPreferences에서 HTTPS 사용 여부 가져오기
private fun getUseHttps(context: Context): Boolean {
    val prefs = context.getSharedPreferences("app_settings", Context.MODE_PRIVATE)
    return prefs.getBoolean(PREF_USE_HTTPS, DEFAULT_USE_HTTPS)
}

/** 이미지 URI의 가로·세로 픽셀 크기 반환 (inJustDecodeBounds) */
private fun getImageDimensions(context: Context, uri: Uri): Pair<Int, Int>? {
    return try {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
        val w = opts.outWidth
        val h = opts.outHeight
        if (w <= 0 || h <= 0) return null

        // EXIF 회전 확인: 90/270도 회전된 이미지는 가로/세로를 교환해야
        // Coil이 EXIF를 반영해 표시하므로, 터치 좌표 계산도 표시 기준으로 맞춰야 함
        val orientation = context.contentResolver.openInputStream(uri)?.use { stream ->
            ExifInterface(stream).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        } ?: ExifInterface.ORIENTATION_NORMAL

        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90,
            ExifInterface.ORIENTATION_ROTATE_270,
            ExifInterface.ORIENTATION_TRANSVERSE,
            ExifInterface.ORIENTATION_TRANSPOSE -> h to w  // 가로/세로 교환
            else -> w to h
        }
    } catch (_: Exception) {
        null
    }
}

/** OOM 방지: 최대 변 길이 이하로 디코딩. EXIF 회전을 적용해 표시 방향과 일치시킴 */
private fun decodeBitmapWithMaxDimension(context: Context, uri: Uri, maxDim: Int): Bitmap? {
    return try {
        // 원본 크기 확인
        val sizeOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, sizeOpts) }
        val rawW = sizeOpts.outWidth
        val rawH = sizeOpts.outHeight
        if (rawW <= 0 || rawH <= 0) return null

        // EXIF 회전 정보 읽기
        val orientation = context.contentResolver.openInputStream(uri)?.use { stream ->
            ExifInterface(stream).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        } ?: ExifInterface.ORIENTATION_NORMAL

        val rotates90or270 = orientation == ExifInterface.ORIENTATION_ROTATE_90 ||
            orientation == ExifInterface.ORIENTATION_ROTATE_270 ||
            orientation == ExifInterface.ORIENTATION_TRANSVERSE ||
            orientation == ExifInterface.ORIENTATION_TRANSPOSE

        // 회전 후 실효 크기 기준으로 sampleSize 계산
        val effectiveW = if (rotates90or270) rawH else rawW
        val effectiveH = if (rotates90or270) rawW else rawH
        var sampleSize = 1
        if (effectiveW > maxDim || effectiveH > maxDim) {
            val halfW = effectiveW / 2
            val halfH = effectiveH / 2
            while (halfW / sampleSize >= maxDim && halfH / sampleSize >= maxDim) {
                sampleSize *= 2
            }
        }

        val decodeOpts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inJustDecodeBounds = false
        }
        val raw = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, decodeOpts)
        } ?: return null

        // 회전 행렬 적용 (Coil과 동일한 방향으로 보정)
        val rotationDeg = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90,
            ExifInterface.ORIENTATION_TRANSVERSE  -> 90f
            ExifInterface.ORIENTATION_ROTATE_180,
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> 180f
            ExifInterface.ORIENTATION_ROTATE_270,
            ExifInterface.ORIENTATION_TRANSPOSE   -> 270f
            else -> 0f
        }
        val needsMirror = orientation == ExifInterface.ORIENTATION_FLIP_HORIZONTAL ||
            orientation == ExifInterface.ORIENTATION_FLIP_VERTICAL ||
            orientation == ExifInterface.ORIENTATION_TRANSVERSE ||
            orientation == ExifInterface.ORIENTATION_TRANSPOSE

        if (rotationDeg == 0f && !needsMirror) return raw

        val matrix = Matrix().apply {
            if (needsMirror) postScale(-1f, 1f, raw.width / 2f, raw.height / 2f)
            if (rotationDeg != 0f) postRotate(rotationDeg, raw.width / 2f, raw.height / 2f)
        }
        Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
            .also { if (it !== raw) raw.recycle() }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

// SharedPreferences에 서버 설정 저장하기
private fun saveServerSettings(context: Context, address: String, port: Int, useHttps: Boolean) {
    val prefs = context.getSharedPreferences("app_settings", Context.MODE_PRIVATE)
    prefs.edit()
        .putString(PREF_SERVER_ADDRESS, address)
        .putInt(PREF_SERVER_PORT, port)
        .putBoolean(PREF_USE_HTTPS, useHttps)
        .apply()
}

enum class CaptureMode {
    PHOTO, VIDEO
}

enum class ResolutionPreset(val width: Int, val height: Int) {
    RESOLUTION_1024x1024(1024, 1024)   // 1024x1024 정사각형 (단일 해상도)
}

enum class MainTab {
    LIBRARY, CLAUDE, CAMERA, CREATE, PROFILE
}

enum class LibraryTab {
    GALLERY, DATASET, MODEL_3D
}

enum class LibraryDetailScreen {
    NONE, DATASET_FOLDER, MODEL_VIEWER, OBJ_VIEWER
}

enum class CameraEntryMode {
    OBJECT, SPACE_2D, SPACE_3D, MOBILE_SPACE
}

private fun CameraEntryMode.isSpaceMode(): Boolean = this != CameraEntryMode.OBJECT
private fun CameraEntryMode.isObjectMode(): Boolean = this == CameraEntryMode.OBJECT

private val AppBackgroundColor = Color.Black
private val BottomBarBackgroundColor = Color(0xFF9CD83B)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // (삭제) 광택/반사 제거 옵션 제거에 따라 OpenCV 초기화 제거

        setContent {
            App_01Theme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    CameraApp(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun CameraApp(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    var hasStoragePermission by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.READ_MEDIA_IMAGES
                ) == PackageManager.PERMISSION_GRANTED
            } else {
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                ) == PackageManager.PERMISSION_GRANTED
            }
        )
    }
    var showServerSettings by remember { mutableStateOf(false) }
    var showSensorCheck by remember { mutableStateOf(false) }
    var selectedTab by remember { mutableStateOf(MainTab.CAMERA) }
    var selectedLibraryTab by remember { mutableStateOf(LibraryTab.GALLERY) }
    var cameraEntryMode by remember { mutableStateOf(CameraEntryMode.OBJECT) }
    var isCameraActive by remember { mutableStateOf(false) }
    var capturedImages by remember { mutableStateOf<List<Uri>>(emptyList()) }
    var lastCapturedImageUri by remember { mutableStateOf<Uri?>(null) }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    val storagePermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasStoragePermission = isGranted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
        if (!hasStoragePermission) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                storagePermissionLauncher.launch(Manifest.permission.READ_MEDIA_IMAGES)
            } else {
                storagePermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }
    }

    LaunchedEffect(Unit) {
        loadCapturedMedia(context) { images ->
            capturedImages = images
            if (images.isNotEmpty()) {
                lastCapturedImageUri = images.first()
            }
        }
    }

    var selectedMediaUri by remember { mutableStateOf<Uri?>(null) }
    var selectedMediaIndex by remember { mutableStateOf(0) }
    var viewingMediaList by remember { mutableStateOf<List<Uri>>(emptyList()) }
    
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(AppBackgroundColor)
    ) {
        if (selectedMediaUri != null && viewingMediaList.isNotEmpty()) {
            MediaDetailScreen(
                mediaList = viewingMediaList,
                initialIndex = selectedMediaIndex,
                onBack = { selectedMediaUri = null },
                onMediaChanged = { index ->
                    if (index in viewingMediaList.indices) {
                        selectedMediaUri = viewingMediaList[index]
                        selectedMediaIndex = index
                    }
                },
                onGalleryUpdated = {
                    loadCapturedMedia(context) { images ->
                        capturedImages = images
                        if (images.isNotEmpty()) lastCapturedImageUri = images.first()
                    }
                }
            )
        } else if (showServerSettings) {
            ServerSettingsScreen(
                onBack = { showServerSettings = false }
            )
        } else if (showSensorCheck) {
            SensorCheckScreen(
                onBack = { showSensorCheck = false }
            )
        } else {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                when (selectedTab) {
                    MainTab.LIBRARY -> {
                        GalleryScreen(
                            images = capturedImages,
                            libraryTab = selectedLibraryTab,
                            onLibraryTabChange = { selectedLibraryTab = it },
                            onMediaSelected = { uri, list ->
                                viewingMediaList = list
                                val index = list.indexOf(uri)
                                if (index >= 0) {
                                    selectedMediaIndex = index
                                    selectedMediaUri = uri
                                }
                            },
                            onImageDeleted = {
                                loadCapturedMedia(context) { images ->
                                    capturedImages = images
                                    if (images.isNotEmpty()) {
                                        lastCapturedImageUri = images.first()
                                    } else {
                                        lastCapturedImageUri = null
                                    }
                                }
                            }
                        )
                    }
                    MainTab.CLAUDE -> {
                        ClaudeChatScreen(
                            galleryImages = capturedImages,
                            onGalleryUpdated = {
                                loadCapturedMedia(context) { images ->
                                    capturedImages = images
                                    if (images.isNotEmpty()) lastCapturedImageUri = images.first()
                                }
                            }
                        )
                    }
                    MainTab.CAMERA -> {
                        if (hasCameraPermission) {
                            if (isCameraActive) {
                                BackHandler {
                                    isCameraActive = false
                                }
                                Box(modifier = Modifier.fillMaxSize()) {
                                    CameraScreen(
                                        cameraEntryMode = cameraEntryMode,
                                        lastCapturedImageUri = lastCapturedImageUri,
                                        onImageCaptured = { uri ->
                                            lastCapturedImageUri = uri
                                            loadCapturedMedia(context) { images ->
                                                capturedImages = images
                                                if (images.isNotEmpty()) {
                                                    lastCapturedImageUri = images.first()
                                                }
                                            }
                                        },
                                        onVideoCaptured = { uri ->
                                            lastCapturedImageUri = uri
                                            loadCapturedMedia(context) { images ->
                                                capturedImages = images
                                                if (images.isNotEmpty()) {
                                                    lastCapturedImageUri = images.first()
                                                }
                                            }
                                        },
                                        onGalleryClick = {
                                            selectedLibraryTab = LibraryTab.GALLERY
                                            selectedTab = MainTab.LIBRARY
                                            isCameraActive = false
                                        }
                                    )
                                }
                            } else {
                                CameraEntryScreen(
                                    selectedMode = cameraEntryMode,
                                    onModeSelected = { mode ->
                                        cameraEntryMode = mode
                                        isCameraActive = true
                                    }
                                )
                            }
                        } else {
                            Box(
                                modifier = Modifier.fillMaxSize(),
                                contentAlignment = Alignment.Center
                            ) {
    Text(
                                    text = "카메라 권한이 필요합니다",
                                    fontSize = 18.sp,
                                    color = Color.White
                                )
                            }
                        }
                    }
                    MainTab.CREATE -> {
                        CreatePlaceholderScreen()
                    }
                    MainTab.PROFILE -> {
                        ProfileScreen(
                            onServerSettingsClick = { showServerSettings = true },
                            onSensorCheckClick = { showSensorCheck = true }
                        )
                    }
                }
            }

            val showBottomBar = !(selectedTab == MainTab.CAMERA && isCameraActive)
            if (showBottomBar) {
                BottomNavigationBar(
                    selectedTab = selectedTab,
                    onTabSelected = { tab ->
                        selectedTab = tab
                        if (tab != MainTab.CAMERA) {
                            isCameraActive = false
                        }
                    }
                )
            }
        }
    }
}

/** 디바이스에서 발견된 개별 카메라 정보 */

@Composable
fun CameraScreen(
    cameraEntryMode: CameraEntryMode,
    lastCapturedImageUri: Uri?,
    onImageCaptured: (Uri) -> Unit,
    onVideoCaptured: (Uri) -> Unit,
    onGalleryClick: () -> Unit
) {
    val context = LocalContext.current
    val mediaActionSound = remember {
        MediaActionSound().apply {
            load(MediaActionSound.START_VIDEO_RECORDING)
            load(MediaActionSound.STOP_VIDEO_RECORDING)
        }
    }
    DisposableEffect(Unit) {
        onDispose { mediaActionSound.release() }
    }
    val lifecycleOwner = LocalLifecycleOwner.current
    var imageCapture: ImageCapture? by remember { mutableStateOf(null) }
    var videoCapture: VideoCapture<androidx.camera.video.Recorder>? by remember { mutableStateOf(null) }
    var lensFacing by remember { mutableStateOf(CameraSelector.LENS_FACING_BACK) }
    var captureMode by remember { mutableStateOf(CaptureMode.PHOTO) }
    var isRecording by remember { mutableStateOf(false) }
    var recordingTime by remember { mutableStateOf(0L) }
    var recording: Recording? by remember { mutableStateOf(null) }
    var previewView: PreviewView? by remember { mutableStateOf(null) }
    var camera: Camera? by remember { mutableStateOf(null) }
    var isCameraReady by remember { mutableStateOf(false) }
    var isFlashOn by remember { mutableStateOf(false) }
    var previewOriginInRoot by remember { mutableStateOf<Offset?>(null) }
    var datasetDir by remember { mutableStateOf<File?>(null) }
    var isDatasetCollectionEnabled by remember { mutableStateOf(true) }
    var selectedResolution by remember { mutableStateOf(ResolutionPreset.RESOLUTION_1024x1024) }
    var azimuthDegrees by remember { mutableStateOf(0f) }

    // ── 후면 카메라 바인딩 정보 ──────────────────────────────────────────────
    // telephoto / wide 각각 "논리 카메라 ID" 또는 "물리 카메라 ID + 부모 논리 ID" 중 하나만 설정됨
    var rearTelephotoLogicalId  by remember { mutableStateOf<String?>(null) }  // 논리 망원
    var rearTelephotoPhysId     by remember { mutableStateOf<String?>(null) }  // 물리 망원
    var rearTelephotoPhysParent by remember { mutableStateOf<String?>(null) }  // 물리 망원의 부모 논리 ID
    var rearWideId              by remember { mutableStateOf<String?>(null) }  // 광각 논리

    // 앱 시작 시 한 번: 2단계 탐색 (논리 → 물리 서브카메라 순)
    LaunchedEffect(Unit) {
        try {
            val cameraProvider = withContext(Dispatchers.IO) {
                ProcessCameraProvider.getInstance(context).get()
            }
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

            // ── 1단계: CameraX 논리 카메라 탐색 ──────────────────────────────
            data class LogicalCam(val id: String, val minFocal: Float)

            @Suppress("UnsafeOptInUsageError")
            val logicalBack: List<LogicalCam> = cameraProvider.availableCameraInfos
                .filter { it.lensFacing == CameraSelector.LENS_FACING_BACK }
                .mapNotNull { info ->
                    val id = runCatching {
                        Camera2CameraInfo.from(info).cameraId
                    }.getOrNull() ?: return@mapNotNull null
                    val focals = runCatching {
                        cameraManager.getCameraCharacteristics(id)
                            .get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    }.getOrNull()?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null
                    LogicalCam(id, focals.min())
                }
                .sortedBy { it.minFocal }

            android.util.Log.d("CameraScreen",
                "논리 후면 카메라: ${logicalBack.map { "${it.id}(${it.minFocal}mm)" }}")

            if (logicalBack.size >= 2) {
                // 논리 카메라가 2개 이상 → 초점거리 최대 = 망원, 3.5mm 근접 = 광각
                rearTelephotoLogicalId = logicalBack.maxByOrNull { it.minFocal }?.id
                rearWideId             = logicalBack.minByOrNull { Math.abs(it.minFocal - 3.5f) }?.id
                android.util.Log.d("CameraScreen",
                    "논리 선택 → 망원: $rearTelephotoLogicalId, 광각: $rearWideId")
            } else {
                // ── 2단계: 논리 카메라가 1개(퓨전 카메라)인 경우 → 물리 서브카메라 탐색 ──
                rearWideId = logicalBack.firstOrNull()?.id

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val logicalIds = logicalBack.map { it.id }
                    var bestPhysId: String? = null
                    var bestParentId: String? = null
                    var bestFocal = 0f

                    for (parentId in logicalIds) {
                        val parentChars = runCatching {
                            cameraManager.getCameraCharacteristics(parentId)
                        }.getOrNull() ?: continue
                        for (physId in parentChars.physicalCameraIds) {
                            val chars = runCatching {
                                cameraManager.getCameraCharacteristics(physId)
                            }.getOrNull() ?: continue
                            val focals = chars
                                .get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                                ?.takeIf { it.isNotEmpty() } ?: continue
                            val maxF = focals.max()
                            if (maxF > bestFocal) {
                                bestFocal = maxF; bestPhysId = physId; bestParentId = parentId
                            }
                        }
                    }

                    // 광각 초점거리보다 1.5배 이상 길어야 망원으로 인정
                    val wideFocal = logicalBack.firstOrNull()?.minFocal ?: 3.5f
                    if (bestPhysId != null && bestFocal > wideFocal * 1.5f) {
                        rearTelephotoPhysId     = bestPhysId
                        rearTelephotoPhysParent = bestParentId
                    }
                    android.util.Log.d("CameraScreen",
                        "물리 선택 → 망원: $rearTelephotoPhysId(${bestFocal}mm, parent=$rearTelephotoPhysParent), 광각: $rearWideId")
                }
            }
        } catch (e: Exception) {
            android.util.Log.w("CameraScreen", "카메라 탐색 실패: ${e.message}")
        }
    }

    // [추가] 촬영 중 화면 켜짐 유지
    val view = LocalView.current
    DisposableEffect(isRecording) {
        if (isRecording) {
            view.keepScreenOn = true
        }
        onDispose {
            view.keepScreenOn = false
        }
    }

    // [추가] 백그라운드 작업 유지를 위한 서비스 제어
    LaunchedEffect(isRecording) {
        val intent = Intent(context, AppForegroundService::class.java)
        if (isRecording) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        } else {
            context.stopService(intent)
        }
    }

    // [추가] 모델링 부적합 경고 표시 상태
    var showModelingWarning by remember { mutableStateOf(false) }

    // [추가] 사물 촬영(OBJECT) 전용: 사물이 중앙 가상 사각형(1000x1000) 밖으로 벗어났는지 경고
    DisposableEffect(Unit) {
        onDispose {
            try {
                ObjectOutOfFrameWarning.close()
            } catch (_: Exception) {
            }
        }
    }

    LaunchedEffect(isFlashOn, camera) {
        try {
            camera?.cameraControl?.enableTorch(isFlashOn)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    var baseAzimuthDegrees by remember { mutableStateOf<Float?>(null) }
    // 수직(세로) 들었을 때 0도를 기준으로 하는 기울기
    var pitchDegrees by remember { mutableStateOf(0f) }
    var basePitchDegrees by remember { mutableStateOf<Float?>(null) }
    var capturedSectors by remember { mutableStateOf<Set<Int>>(emptySet()) }
    val ringSize = 120.dp

    val sectorCount = 30
    val sectorSize = 360f / sectorCount
    // 지면-휴대폰 각도 기준: 90도, 110도, 70도 (공간 스캔 시)
    val pitchTargets = remember(cameraEntryMode) {
        if (cameraEntryMode.isSpaceMode()) listOf(90f, 110f, 70f) else listOf(120f)
    }
    val pitchTolerance = 10f
    var currentPitchIndex by remember { mutableStateOf(0) }
    val currentTargetPitchState = remember {
        derivedStateOf { pitchTargets.getOrNull(currentPitchIndex) }
    }
    val currentTargetPitch by currentTargetPitchState
    val effectivePitchDegreesState = remember {
        // [최적화] pitchDegrees가 이제 중력 기준 절대 각도(수직=90)이므로 그대로 사용
        derivedStateOf { pitchDegrees }
    }
    val effectivePitchDegrees by effectivePitchDegreesState
    val isPitchAlignedState = remember {
        derivedStateOf {
            val target = currentTargetPitchState.value
            target != null && abs(effectivePitchDegrees - target) <= pitchTolerance
        }
    }
    val isPitchAligned by isPitchAlignedState
    val relativeAzimuthDegrees by remember {
        derivedStateOf {
            val base = baseAzimuthDegrees
            if (base == null) {
                azimuthDegrees
            } else {
                var diff = azimuthDegrees - base
                if (diff < 0f) diff += 360f
                diff % 360f
            }
        }
    }
    val displayAzimuthDegrees by remember {
        derivedStateOf { relativeAzimuthDegrees % 360f }
    }

    val currentSectorIndex by remember {
        derivedStateOf {
            ((displayAzimuthDegrees / sectorSize).toInt()).coerceIn(0, sectorCount - 1)
        }
    }

    DisposableEffect(cameraEntryMode) {
        capturedSectors = emptySet()
        currentPitchIndex = 0
        basePitchDegrees = null
        onDispose { }
    }

    DisposableEffect(cameraEntryMode) {
        val shouldUseSensors = cameraEntryMode.isSpaceMode() || cameraEntryMode.isObjectMode()
        if (!shouldUseSensors) {
            onDispose { }
            return@DisposableEffect onDispose { }
        }

        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                val rotationMatrix = FloatArray(9)
                val orientation = FloatArray(3)
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                val adjustedRotationMatrix = FloatArray(9)
                val rotation = previewView?.display?.rotation ?: Surface.ROTATION_0
                val (xAxis, yAxis) = when (rotation) {
                    Surface.ROTATION_90 -> SensorManager.AXIS_Y to SensorManager.AXIS_MINUS_X
                    Surface.ROTATION_180 -> SensorManager.AXIS_MINUS_X to SensorManager.AXIS_MINUS_Y
                    Surface.ROTATION_270 -> SensorManager.AXIS_MINUS_Y to SensorManager.AXIS_X
                    else -> SensorManager.AXIS_X to SensorManager.AXIS_Y
                }
                SensorManager.remapCoordinateSystem(
                    rotationMatrix,
                    xAxis,
                    yAxis,
                    adjustedRotationMatrix
                )
                SensorManager.getOrientation(adjustedRotationMatrix, orientation)
                var rawAzimuth = Math.toDegrees(orientation[0].toDouble()).toFloat()
                if (rawAzimuth < 0f) rawAzimuth += 360f

                // [수정] Low-pass Filter 적용 (노이즈/Jitter 감소)
                val alpha = 0.15f
                val currentAzimuth = azimuthDegrees
                var delta = rawAzimuth - currentAzimuth
                if (delta > 180f) delta -= 360f
                if (delta < -180f) delta += 360f
                var nextAzimuth = currentAzimuth + delta * alpha
                if (nextAzimuth < 0f) nextAzimuth += 360f
                if (nextAzimuth >= 360f) nextAzimuth -= 360f

                azimuthDegrees = nextAzimuth

                val rawPitch = Math.toDegrees(orientation[1].toDouble()).toFloat()
                
                // [개선] 중력 벡터를 이용한 절대 기울기 계산 (수직=90도)
                // rotationMatrix[7]: Y축 성분, rotationMatrix[8]: Z축 성분
                val worldZInPhoneY = rotationMatrix[7]
                val worldZInPhoneZ = rotationMatrix[8]
                val angleDeg = Math.toDegrees(Math.atan2(worldZInPhoneZ.toDouble(), worldZInPhoneY.toDouble())).toFloat()
                
                // 앞으로 숙이면 90도 미만, 뒤로 젖히면 90도 초과가 되도록 설정
                pitchDegrees = 90f + angleDeg
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        if (rotationSensor != null) {
            sensorManager.registerListener(
                listener,
                rotationSensor,
                SensorManager.SENSOR_DELAY_UI
            )
        }

        onDispose {
            sensorManager.unregisterListener(listener)
        }
    }

    // 동영상 촬영 시간 업데이트
    LaunchedEffect(isRecording) {
        while (isRecording) {
            delay(100)
            recordingTime += 100
        }
    }

    // 동영상 촬영 시 데이터셋 수집 제어 (촬영 시작 3초 후 시작)
    LaunchedEffect(isRecording, captureMode) {
        if (captureMode != CaptureMode.VIDEO || !isRecording) {
            isDatasetCollectionEnabled = true
            return@LaunchedEffect
        }
        isDatasetCollectionEnabled = false
        delay(3000)
        if (isRecording && captureMode == CaptureMode.VIDEO) {
            isDatasetCollectionEnabled = true
        }
    }

    // 동영상 촬영 시 구역 체크: 1초 간격으로 현재 구역 기록 (사물/공간 모두)
    LaunchedEffect(isRecording, cameraEntryMode, captureMode, currentPitchIndex, isDatasetCollectionEnabled, datasetDir) {
        // [수정] 이동식 공간 촬영인 경우: 촬영 시작 3초 후(isDatasetCollectionEnabled=true) 1초 간격 자동 촬영
        if (cameraEntryMode == CameraEntryMode.MOBILE_SPACE) {
            // 녹화가 아니거나 데이터셋 수집 중단 시 경고 끄기
            if (!isRecording || !isDatasetCollectionEnabled) {
                showModelingWarning = false
            }

            if (captureMode == CaptureMode.VIDEO && isRecording && isDatasetCollectionEnabled) {
                var captureCount = 0
                val dir = datasetDir
                val capture = imageCapture
                
                // 직전에 저장된 데이터셋 이미지의 축소본 (유사도 비교용)
                var lastSavedBitmapSmall: android.graphics.Bitmap? = null

                if (dir != null && capture != null) {
                    while (isRecording && isDatasetCollectionEnabled) {
                        captureCount++
                        val fileName = "mobile_${captureCount}.jpg"
                        
                        captureDatasetImage(context, 0, 0, dir, capture, fileName) { currentBitmap ->
                            // 1. 모델링 적합성 검사 (경고 표시 업데이트)
                            val isSuitable = checkModelingSuitability(currentBitmap)
                            showModelingWarning = !isSuitable

                            // [수정] 적합하지 않으면(경고 상태) 데이터셋 저장 중단 (파일도 삭제하는 것이 좋으나 captureDatasetImage 내부 구현상 파일은 이미 생성됨. 여기서는 리스트 관리나 후속 처리를 막음)
                            // captureDatasetImage 함수가 파일을 이미 저장했으므로, 엄밀히는 파일을 지워야 하지만
                            // 현재 구조상 콜백에서 파일을 지우기 어려우므로, 다음 비교 대상 갱신 등을 하지 않음으로서 논리적 제외 처리.
                            // 하지만 파일이 남으면 안 되므로, 파일 삭제 로직 추가가 필요할 수 있음.
                            // 일단 요청사항인 "수집이 이루어지지 않도록"을 위해 로직 흐름을 차단.
                            if (!isSuitable) {
                                // 생성된 파일 삭제
                                val createdFile = File(dir, fileName)
                                if (createdFile.exists()) {
                                    createdFile.delete()
                                }
                                return@captureDatasetImage false
                            }

                            // [추가] 유사도 기반 저장 필터링
                            if (lastSavedBitmapSmall == null) {
                                // 첫 번째 데이터셋은 무조건 저장
                                lastSavedBitmapSmall = android.graphics.Bitmap.createScaledBitmap(currentBitmap, 64, 64, true)
                                true
                            } else {
                                // 현재 이미지와 직전 저장된 이미지 간의 구조적 유사도 비교
                                val similarity = calculateImageSimilarity(lastSavedBitmapSmall!!, currentBitmap)
                                
                                // 유사도가 70% ~ 90% 사이일 경우에만 저장
                                if (similarity >= 0.7f && similarity <= 0.9f) {
                                    // 조건 만족: 저장하고 비교 기준 갱신
                                    lastSavedBitmapSmall = android.graphics.Bitmap.createScaledBitmap(currentBitmap, 64, 64, true)
                                    true
                                } else {
                                    // 조건 불만족: 저장하지 않음 (너무 다르거나, 너무 비슷함) -> 파일 삭제
                                    val createdFile = File(dir, fileName)
                                    if (createdFile.exists()) {
                                        createdFile.delete()
                                    }
                                    false
                                }
                            }
                        }
                        delay(1000) // 1초 간격
                    }
                }
            }
            return@LaunchedEffect
        }

        if (captureMode == CaptureMode.VIDEO && isRecording && isDatasetCollectionEnabled) {
            // 사물/공간 촬영: 피치 각도 체크 (90도 등)
            if (currentPitchIndex < pitchTargets.size) {
                while (isRecording) {
                    delay(500) // 딜레이 단축 (반응성 향상)
                    val targetPitch = currentTargetPitchState.value
                    val pitchNow = effectivePitchDegreesState.value
                    val isAligned =
                        targetPitch != null && abs(pitchNow - targetPitch) <= pitchTolerance
                    if (isAligned) {
                        val sectorIndex =
                            ((displayAzimuthDegrees / sectorSize).toInt()).coerceIn(0, sectorCount - 1)
                        if (!capturedSectors.contains(sectorIndex)) {
                            // [수정] 선명도 보정(초점 재조정) 로직 제거됨 - 사용자 요청
                            // Joint Reflection Removal and Depth Estimation 준비 단계
                            
                            capturedSectors = capturedSectors + sectorIndex
                            val dir = datasetDir
                            val capture = imageCapture
                            if (dir != null && capture != null && targetPitch != null) {
                                captureDatasetImage(context, sectorIndex, targetPitch.toInt(), dir, capture)
                            }
                            
                            // 연속 촬영 방지를 위해 잠시 대기
                            delay(300)
                        }
                    }
                }
            }
        }
    }

    // 한 각도에서 360도 촬영 완료 시 다음 각도로 전환
    LaunchedEffect(capturedSectors, isRecording, cameraEntryMode, captureMode, currentPitchIndex) {
        // [추가] 이동식 공간 촬영인 경우 섹터 로직 생략
        if (cameraEntryMode == CameraEntryMode.MOBILE_SPACE) {
            return@LaunchedEffect
        }

        if (captureMode == CaptureMode.VIDEO &&
            isRecording &&
            currentPitchIndex < pitchTargets.size &&
            capturedSectors.size >= sectorCount
        ) {
            currentPitchIndex += 1
            capturedSectors = emptySet()
        }
    }

    // 카메라 바인딩 함수
    fun bindCamera(view: PreviewView) {
        val executor = ContextCompat.getMainExecutor(context)
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                // ── 후면 카메라 선택 우선순위 ────────────────────────────────
                //  1) 망원 논리 카메라  2) 망원 물리 카메라(부모 논리 + setPhysicalCameraId)
                //  3) 광각 논리 카메라  4) 기기 기본 후면 카메라(폴백)
                // 전면: 기기 기본 전면 카메라
                @Suppress("UnsafeOptInUsageError")
                fun selectorByLogicalId(logicalId: String) = CameraSelector.Builder()
                    .addCameraFilter { list ->
                        val matched = list.filter {
                            runCatching { Camera2CameraInfo.from(it).cameraId == logicalId }
                                .getOrDefault(false)
                        }
                        matched.ifEmpty { list.filter { it.lensFacing == CameraSelector.LENS_FACING_BACK } }
                    }
                    .build()

                // 물리 망원 카메라 사용 여부와 그 ID (Preview/ImageCapture 빌더에 적용)
                val physicalIdToApply: String? = when {
                    lensFacing == CameraSelector.LENS_FACING_BACK
                        && rearTelephotoLogicalId == null
                        && rearTelephotoPhysId != null -> rearTelephotoPhysId
                    else -> null
                }

                val cameraSelector = when {
                    lensFacing == CameraSelector.LENS_FACING_FRONT ->
                        CameraSelector.DEFAULT_FRONT_CAMERA
                    rearTelephotoLogicalId != null ->
                        selectorByLogicalId(rearTelephotoLogicalId!!)
                    rearTelephotoPhysParent != null ->
                        selectorByLogicalId(rearTelephotoPhysParent!!)
                    rearWideId != null ->
                        selectorByLogicalId(rearWideId!!)
                    else ->
                        CameraSelector.DEFAULT_BACK_CAMERA
                }

                cameraProvider.unbindAll()

                // 선택된 해상도
                val targetSize = Size(selectedResolution.width, selectedResolution.height)
                
                // ViewPort 설정으로 명시적 비율 지정
                val viewPort = androidx.camera.core.ViewPort.Builder(
                    android.util.Rational(
                        selectedResolution.width,
                        selectedResolution.height
                    ),
                    view.display.rotation
                ).setScaleType(androidx.camera.core.ViewPort.FILL_CENTER).build()
                
                val resolutionSelector = ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            targetSize,
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()

                val previewBuilder = Preview.Builder()
                    .setResolutionSelector(resolutionSelector)
                if (physicalIdToApply != null) {
                    @Suppress("UnsafeOptInUsageError")
                    Camera2Interop.Extender(previewBuilder).setPhysicalCameraId(physicalIdToApply)
                }
                val preview = previewBuilder.build().also {
                    it.setSurfaceProvider(view.surfaceProvider)
                }

                val imageCaptureBuilder = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setResolutionSelector(resolutionSelector)
                if (physicalIdToApply != null) {
                    @Suppress("UnsafeOptInUsageError")
                    Camera2Interop.Extender(imageCaptureBuilder).setPhysicalCameraId(physicalIdToApply)
                }
                val newImageCapture = imageCaptureBuilder.build()

                if (captureMode == CaptureMode.PHOTO) {
                    imageCapture = newImageCapture

                    // UseCaseGroup을 사용하여 ViewPort 적용
                    val useCaseGroup = androidx.camera.core.UseCaseGroup.Builder()
                        .setViewPort(viewPort)
                        .addUseCase(preview)
                        .addUseCase(imageCapture!!)
                        .build()

                    camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        useCaseGroup
                    )
                    videoCapture = null
                    isCameraReady = true
                } else {
                // VideoCapture에도 해상도 설정 적용
                // Recorder는 해상도보다는 Quality를 사용하지만,
                // 선택된 해상도에 따라 최대한 근접한 품질을 선택합니다.
                val recorder = androidx.camera.video.Recorder.Builder()
                    .setQualitySelector(
                        androidx.camera.video.QualitySelector.from(
                            androidx.camera.video.Quality.HIGHEST
                        )
                    )
                    .build()

                    videoCapture = VideoCapture.withOutput(recorder)
                    imageCapture = newImageCapture

                    // UseCaseGroup을 사용하여 ViewPort 적용
                    val useCaseGroup = androidx.camera.core.UseCaseGroup.Builder()
                        .setViewPort(viewPort)
                        .addUseCase(preview)
                        .addUseCase(videoCapture!!)
                        .addUseCase(imageCapture!!)
                        .build()

                    camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        useCaseGroup
                    )
                    isCameraReady = true
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, executor)
    }

    // lensFacing, captureMode, selectedResolution, 카메라ID 변경 시 재바인딩
    LaunchedEffect(lensFacing, captureMode, selectedResolution, previewView,
        rearTelephotoLogicalId, rearTelephotoPhysId, rearWideId) {
        isCameraReady = false
        previewView?.let { bindCamera(it) }
    }

    // 선택된 해상도의 비율 계산 (정사각형: 1024/1024)
    val density = LocalDensity.current

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
        // 해상도 변경 시 PreviewView를 완전히 재생성하기 위해 key 사용
        key(selectedResolution) {
            // 정사각형 해상도: 1:1 미리보기 강제 적용
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RectangleShape)
                    .onGloballyPositioned { coords ->
                        previewOriginInRoot = coords.positionInRoot()
                    },
                contentAlignment = Alignment.Center
            ) {
                AndroidView(
                    factory = { ctx ->
                        PreviewView(ctx).apply {
                            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                            scaleType = PreviewView.ScaleType.FILL_CENTER
                            previewView = this
                        }
                    },
                    modifier = Modifier.fillMaxSize()
                )
            }
        }

        // 상단 바 배경
        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .height(64.dp)
                .background(Color.Black)
        )

        // 동영상 촬영 시간 표시 (상단 알약)
        if (isRecording) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 72.dp)
                    .background(Color.Red, RoundedCornerShape(16.dp))
                    .padding(horizontal = 12.dp, vertical = 4.dp)
            ) {
                Text(
                    text = "동영상 촬영 ${formatTime(recordingTime)}",
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // 통합 경고 메시지 영역 (기울기/모델링 적합성)
        val shouldShowWarningColumn = captureMode == CaptureMode.VIDEO && isRecording
        if (shouldShowWarningColumn) {
            Column(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 110.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                if (captureMode == CaptureMode.VIDEO && isRecording) {
                    // 기울기 경고 (이동식 공간 촬영 제외)
                    if (cameraEntryMode != CameraEntryMode.MOBILE_SPACE) {
                        val isAllPitchCompleted = currentPitchIndex >= pitchTargets.size
                        if (!isAllPitchCompleted && !isPitchAligned) {
                            val targetPitch = currentTargetPitch
                            val warningMsg = if (targetPitch != null) {
                                val delta = effectivePitchDegrees - targetPitch
                                when {
                                    delta < -pitchTolerance -> "기기를 아래쪽으로 기울여주세요."
                                    delta > pitchTolerance -> "기기를 위쪽으로 기울여주세요."
                                    else -> null
                                }
                            } else null

                            if (warningMsg != null) {
                                Text(
                                    text = warningMsg,
                                    color = Color.White,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier
                                        .background(Color.Red, RoundedCornerShape(8.dp))
                                        .padding(horizontal = 12.dp, vertical = 4.dp)
                                )
                            }
                        }
                    }

                    // 모델링 적합성 경고 (깊이 데이터 부족)
                    // 이동식 공간 촬영인 경우에만 표시 (showModelingWarning은 이동식 공간 촬영에서만 갱신됨)
                    if (showModelingWarning) {
                        Text(
                            text = "공간에 대한 깊이 데이터가 부족합니다.",
                            color = Color.White,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier
                                .background(Color.Red, RoundedCornerShape(8.dp))
                                .padding(horizontal = 12.dp, vertical = 4.dp)
                        )
                    }
                }
            }
        }

        // 상단 모드 전환 + 해상도 선택 (한 줄, 알약 형태)
        val topMenuPadding = 8.dp
        Row(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .padding(top = topMenuPadding),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            TopMenuSegmented(
                leftText = "사진",
                rightText = "동영상",
                isLeftSelected = captureMode == CaptureMode.PHOTO,
                onLeftClick = { captureMode = CaptureMode.PHOTO },
                onRightClick = { captureMode = CaptureMode.VIDEO }
            )
            
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = { isFlashOn = !isFlashOn }) {
                    Icon(
                        imageVector = if (isFlashOn) Icons.Filled.FlashOn else Icons.Filled.FlashOff,
                        contentDescription = "Flash",
                        tint = Color.White
                    )
                }
                
            }
        }

        // 동영상 촬영 중 구역 수집 정보 표시 (사물/공간 모두) - 이동식 공간 촬영은 제외
        if (captureMode == CaptureMode.VIDEO && isRecording && cameraEntryMode != CameraEntryMode.MOBILE_SPACE) {
            // 구역 수집 진행률 표시
            Text(
                text = "${capturedSectors.size} / ${sectorCount}장",
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 190.dp)
                    .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                    .padding(horizontal = 10.dp, vertical = 4.dp)
            )
            
            // 동영상 촬영 안내 메시지 (사물/공간 모두 90도 촬영 지원)
            val isAllPitchCompleted = currentPitchIndex >= pitchTargets.size
            val instructionText = when {
                isAllPitchCompleted -> "모든 각도 촬영 완료"
                else -> currentTargetPitch?.let { "${it.toInt()}도 측정입니다" } ?: "촬영을 진행해주세요"
            }
            Text(
                text = instructionText,
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 230.dp)
                    .background(
                        if (isPitchAligned) Color(0xFF4CAF50).copy(alpha = 0.8f)
                        else Color.Black.copy(alpha = 0.6f),
                        RoundedCornerShape(12.dp)
                    )
                    .padding(horizontal = 10.dp, vertical = 4.dp)
            )
            }
        
        // 하단 컨트롤 바
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color.Black)
                .padding(horizontal = 24.dp, vertical = 16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // 마지막 촬영 사진 썸네일 또는 갤러리 버튼
            Box(
                modifier = Modifier
                    .size(56.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .border(2.dp, Color.White, RoundedCornerShape(8.dp))
                    .clickable { onGalleryClick() },
                contentAlignment = Alignment.Center
            ) {
                if (lastCapturedImageUri != null) {
                    val isVideo = isVideoUri(context, lastCapturedImageUri)
                    var videoThumbnail by remember(lastCapturedImageUri) { mutableStateOf<Bitmap?>(null) }
                    
                    // 동영상 썸네일 로드
                    if (isVideo) {
                        LaunchedEffect(lastCapturedImageUri) {
                            videoThumbnail = withContext(Dispatchers.IO) {
                                try {
                                    val retriever = MediaMetadataRetriever()
                                    try {
                                        retriever.setDataSource(context, lastCapturedImageUri)
                                        retriever.getFrameAtTime(0)
                                    } finally {
                                        retriever.release()
                                    }
                                } catch (e: Exception) {
                                    e.printStackTrace()
                                    null
                                }
                            }
                        }
                        
                        // 동영상 썸네일 표시
                        if (videoThumbnail != null) {
                            Image(
                                bitmap = videoThumbnail!!.asImageBitmap(),
                                contentDescription = "마지막 촬영 동영상",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Crop
                            )
                            // 동영상 아이콘 오버레이
                            Icon(
                                imageVector = Icons.Filled.Videocam,
                                contentDescription = "동영상",
                                tint = Color.White,
                                modifier = Modifier
                                    .align(Alignment.BottomEnd)
                                    .size(16.dp)
                                    .padding(4.dp)
                                    .background(Color.Black.copy(alpha = 0.6f), CircleShape)
                                    .padding(2.dp)
                            )
                        } else {
                            // 썸네일 로딩 중
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(Color.Black)
                            )
                            Icon(
                                imageVector = Icons.Filled.Videocam,
                                contentDescription = "동영상",
                                tint = Color.White.copy(alpha = 0.7f),
                                modifier = Modifier.size(24.dp)
                            )
                        }
                    } else {
                        // 이미지 표시
                        Image(
                            painter = rememberAsyncImagePainter(lastCapturedImageUri),
                            contentDescription = "마지막 촬영 사진",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop
                        )
                    }
                } else {
                    Icon(
                        imageVector = Icons.Filled.PhotoLibrary,
                        contentDescription = "갤러리",
                        tint = Color.White,
                        modifier = Modifier.size(32.dp)
                    )
                }
            }

            // 동영상 촬영 중 구역 링 표시 (사물/공간 모두) - 이동식 공간 촬영은 제외
            val showRing = captureMode == CaptureMode.VIDEO && isRecording && cameraEntryMode != CameraEntryMode.MOBILE_SPACE
            val captureButtonSize = if (isRecording) 64.dp else 72.dp

            // 촬영 버튼 + 링 (중심 일치)
            Box(
                modifier = Modifier.size(if (showRing) ringSize else captureButtonSize)
            ) {
                if (showRing) {
                    Canvas(modifier = Modifier.matchParentSize()) {
                        val canvasCenter = Offset(size.width / 2f, size.height / 2f)
                        val radius = size.minDimension / 2f
                        val barEnd = radius * 0.84f
                        val barStart = radius * 0.76f
                        val stroke = 10f

                        for (i in 0 until sectorCount) {
                            val angle = i * (360f / sectorCount)
                            val color = when {
                                i == currentSectorIndex -> Color.White
                                capturedSectors.contains(i) -> Color(0xFF4CAF50)
                                else -> Color.DarkGray
                            }
                            rotate(degrees = angle, pivot = canvasCenter) {
                                drawLine(
                                    color = color,
                                    start = Offset(canvasCenter.x, canvasCenter.y - barStart),
                                    end = Offset(canvasCenter.x, canvasCenter.y - barEnd),
                                    strokeWidth = stroke,
                                    cap = StrokeCap.Round
                                )
                            }
                        }
                    }
                }
                Box(
                    modifier = Modifier
                        .size(captureButtonSize)
                        .align(Alignment.Center)
                        .clip(CircleShape)
                        .background(if (isRecording) Color.Red else Color.White)
                        .clickable(enabled = !isRecording || captureMode == CaptureMode.VIDEO) {
                            if (captureMode == CaptureMode.PHOTO) {
                                imageCapture?.let { capture ->
                                    // 셔터 소리를 약 30% 수준으로 낮춤 (MediaActionSound는 볼륨 조절 불가)
                                    SoftShutterSound.play(volume = 0.3f)
                                    takePhoto(context, capture) { uri ->
                                        onImageCaptured(uri)
                                    }
                                }
                            } else {
                                if (!isRecording) {
                                    // 동영상 촬영 시작 - 카메라가 준비되었는지 확인
                                    if (isCameraReady && videoCapture != null) {
                                        videoCapture?.let { capture ->
                                            mediaActionSound.play(MediaActionSound.START_VIDEO_RECORDING)
                                            startVideoRecording(
                                                context,
                                                capture,
                                                onRecordingStarted = { recordingInstance ->
                                                    recording = recordingInstance
                                                    isRecording = true
                                                    recordingTime = 0L
                                                    val sessionId = SimpleDateFormat(
                                                        "yyyy-MM-dd-HH-mm-ss-SSS",
                                                        Locale.US
                                                    ).format(System.currentTimeMillis())
                                                    val root = File(context.getExternalFilesDir(null), "datasets")
                                                    if (!root.exists()) {
                                                        root.mkdirs()
                                                    }
                                                    datasetDir = File(root, sessionId).apply {
                                                        mkdirs()
                                                    }
                                                    // 동영상 촬영 시작 시 방위각 기준 설정
                                                    baseAzimuthDegrees = azimuthDegrees
                                                    capturedSectors = emptySet()
                                                    basePitchDegrees = pitchDegrees
                                                    currentPitchIndex = 0
                                                },
                                                onVideoSaved = { uri ->
                                                    onVideoCaptured(uri)
                                                }
                                            )
                                        }
                                    }
                                } else {
                                    // 동영상 촬영 중지
                                    mediaActionSound.play(MediaActionSound.STOP_VIDEO_RECORDING)
                                    recording?.stop()
                                    recording = null
                                    isRecording = false
                                    recordingTime = 0L

                                    // [추가] 빈 데이터셋 폴더 정리
                                    // 데이터셋 폴더에 이미지가 하나도 없으면 폴더 자동 삭제
                                    val targetDir = datasetDir
                                    if (targetDir != null && targetDir.exists() && targetDir.isDirectory) {
                                        val files = targetDir.listFiles()
                                        if (files == null || files.isEmpty()) {
                                            targetDir.delete()
                                        }
                                    }

                                    datasetDir = null
                                    // 동영상 촬영 종료 시 상태 초기화
                                    baseAzimuthDegrees = null
                                    capturedSectors = emptySet()
                                    basePitchDegrees = null
                                    currentPitchIndex = 0
                                }
                            }
                        },
                    contentAlignment = Alignment.Center
                ) {
                    if (captureMode == CaptureMode.PHOTO) {
                        Icon(
                            imageVector = Icons.Filled.Camera,
                            contentDescription = "촬영",
                            tint = Color.Black,
                            modifier = Modifier.size(36.dp)
                        )
                    } else {
                        if (isRecording) {
                            Box(
                                modifier = Modifier
                                    .size(24.dp)
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(Color.White)
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Filled.Videocam,
                                contentDescription = "동영상 촬영",
                                tint = Color.Black,
                                modifier = Modifier.size(36.dp)
                            )
                        }
                    }
                }
            }

            // 전면/후면 카메라 전환 버튼
            Icon(
                imageVector = Icons.Filled.Cameraswitch,
                contentDescription = "카메라 전환",
                tint = Color.White,
                modifier = Modifier
                    .size(48.dp)
                    .clickable {
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                            CameraSelector.LENS_FACING_FRONT
                        else
                            CameraSelector.LENS_FACING_BACK
                    }
            )
        }

        // 링은 촬영 버튼 내부로 이동
    }
}

@Composable
fun CameraEntryScreen(
    selectedMode: CameraEntryMode,
    onModeSelected: (CameraEntryMode) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppBackgroundColor)
            .padding(16.dp)
    ) {
        Text(
            text = "카메라",
            color = Color.White,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(72.dp))
        CameraModeButton(
            text = "사물 스캔",
            isSelected = selectedMode == CameraEntryMode.OBJECT,
            onClick = { onModeSelected(CameraEntryMode.OBJECT) }
        )
        Spacer(modifier = Modifier.height(20.dp))
        CameraModeButton(
            text = "2차원 공간 스캔",
            isSelected = selectedMode == CameraEntryMode.SPACE_2D,
            onClick = { onModeSelected(CameraEntryMode.SPACE_2D) }
        )
        Spacer(modifier = Modifier.height(20.dp))
        CameraModeButton(
            text = "3차원 공간 스캔",
            isSelected = selectedMode == CameraEntryMode.SPACE_3D,
            onClick = { onModeSelected(CameraEntryMode.SPACE_3D) }
        )
        Spacer(modifier = Modifier.height(20.dp))
        CameraModeButton(
            text = "이동식 공간 촬영",
            isSelected = selectedMode == CameraEntryMode.MOBILE_SPACE,
            onClick = { onModeSelected(CameraEntryMode.MOBILE_SPACE) }
        )
    }
}

@Composable
fun CameraModeButton(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 48.dp)
            .height(58.dp)
            .clip(RoundedCornerShape(29.dp))
            .background(if (isSelected) Color.White else Color.Transparent)
            .border(1.dp, Color.White, RoundedCornerShape(28.dp))
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = text,
            color = if (isSelected) Color.Black else Color.White,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun TopMenuPill(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(if (isSelected) Color.White else Color.Transparent)
            .border(1.dp, Color.White, RoundedCornerShape(20.dp))
            .clickable { onClick() }
            .padding(horizontal = 12.dp, vertical = 6.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = text,
            color = if (isSelected) Color.Black else Color.White,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun TopMenuSegmented(
    leftText: String,
    rightText: String,
    isLeftSelected: Boolean,
    onLeftClick: () -> Unit,
    onRightClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .height(34.dp)
            .clip(RoundedCornerShape(18.dp))
            .background(Color.Black)
            .border(1.dp, Color.White, RoundedCornerShape(18.dp))
    ) {
        Box(
            modifier = Modifier
                .clip(
                    RoundedCornerShape(
                        topStart = 18.dp,
                        bottomStart = 18.dp,
                        topEnd = 0.dp,
                        bottomEnd = 0.dp
                    )
                )
                .background(if (isLeftSelected) Color.White else Color.Black)
                .clickable { onLeftClick() }
                .padding(horizontal = 14.dp, vertical = 6.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = leftText,
                color = if (isLeftSelected) Color.Black else Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold
            )
        }
        Box(
            modifier = Modifier
                .width(1.dp)
                .fillMaxHeight()
                .background(Color.White)
        )
        Box(
            modifier = Modifier
                .clip(
                    RoundedCornerShape(
                        topStart = 0.dp,
                        bottomStart = 0.dp,
                        topEnd = 18.dp,
                        bottomEnd = 18.dp
                    )
                )
                .background(if (!isLeftSelected) Color.White else Color.Black)
                .clickable { onRightClick() }
                .padding(horizontal = 14.dp, vertical = 6.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = rightText,
                color = if (!isLeftSelected) Color.Black else Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
fun LibraryTabButton(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(if (isSelected) Color.White else Color.Transparent)
            .border(1.dp, Color.White, RoundedCornerShape(20.dp))
            .clickable { onClick() }
            .padding(horizontal = 16.dp, vertical = 7.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = text,
            color = if (isSelected) Color.Black else Color.White,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun ClaudeChatScreen(
    galleryImages: List<Uri>,
    onGalleryUpdated: () -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var messageText by remember { mutableStateOf("") }
    var messages by remember { mutableStateOf<List<ChatMessage>>(emptyList()) }
    var isWaiting by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showImageSelectDialog by remember { mutableStateOf(false) }
    var datasetFolders by remember { mutableStateOf<List<DatasetFolder>>(emptyList()) }
    var attachedImages by remember { mutableStateOf<List<Uri>>(emptyList()) }
    val listState = rememberLazyListState()

    // 갤러리: 이미지만 (동영상 제외)
    val galleryImageUris = remember(galleryImages) {
        galleryImages.filter { uri ->
            val path = uri.path ?: ""
            path.endsWith(".jpg", true) || path.endsWith(".jpeg", true) ||
                path.endsWith(".png", true) || path.endsWith(".webp", true) ||
                path.endsWith(".heic", true) || path.endsWith(".heif", true)
        }
    }

    LaunchedEffect(Unit) {
        loadDatasetFolders(context) { datasetFolders = it }
    }

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppBackgroundColor)
    ) {
        Text(
            text = "Claude",
            color = Color.White,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(16.dp)
        )
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Color.White.copy(alpha = 0.3f))
        )
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            state = listState,
            contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(messages) { msg ->
                ChatBubble(
                    message = msg,
                    isUser = msg.isUser
                )
            }
            if (isWaiting) {
                item {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        horizontalArrangement = Arrangement.Start
                    ) {
                        Box(
                            modifier = Modifier
                                .background(Color(0xFF2A2A2A), RoundedCornerShape(16.dp))
                                .padding(16.dp)
                        ) {
                            Text(
                                text = "답변 생성 중...",
                                color = Color.White.copy(alpha = 0.8f),
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }
        }
        errorMessage?.let { err ->
            Text(
                text = err,
                color = Color(0xFFFF6B6B),
                fontSize = 12.sp,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
            )
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF1A1A1A))
                .padding(12.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            IconButton(
                onClick = { showImageSelectDialog = true }
            ) {
                Icon(
                    imageVector = Icons.Filled.AddPhotoAlternate,
                    contentDescription = "이미지 첨부",
                    tint = if (attachedImages.isNotEmpty()) Color(0xFF9CD83B) else Color.White.copy(alpha = 0.8f)
                )
            }
            OutlinedTextField(
                value = messageText,
                onValueChange = { messageText = it },
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 4.dp),
                placeholder = { Text("메시지 입력...", color = Color.Gray) },
                maxLines = 4,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    focusedBorderColor = Color.White.copy(alpha = 0.5f),
                    unfocusedBorderColor = Color.White.copy(alpha = 0.3f),
                    cursorColor = Color.White
                )
            )
            IconButton(
                onClick = {
                    if (isWaiting) return@IconButton
                    val text = messageText.trim()
                    val images = attachedImages
                    if (text.isEmpty() && images.isEmpty()) return@IconButton

                    val displayText = when {
                        images.size > 1 -> "[${images.size}장의 이미지]"
                        images.size == 1 -> "[이미지 1장]"
                        else -> ""
                    }
                    messages = messages + ChatMessage(
                        text = if (text.isNotEmpty()) text else displayText,
                        isUser = true,
                        imageUris = images
                    )
                    messageText = ""
                    attachedImages = emptyList()
                    isWaiting = true
                    errorMessage = null

                    scope.launch {
                        val imageBase64List = images.mapNotNull { uri ->
                            try {
                                context.contentResolver.openInputStream(uri)?.use { stream ->
                                    val bitmap = BitmapFactory.decodeStream(stream)
                                    bitmap?.let { ClaudeChatClient.bitmapToBase64(it) }
                                }
                            } catch (e: Exception) { null }
                        }
                        val result = ClaudeChatClient.sendMessage(
                            text = text.ifBlank { if (imageBase64List.size > 1) "이 이미지들에 대해 설명해 주세요." else "이 이미지에 대해 설명해 주세요." },
                            imageBase64List = imageBase64List
                        )
                        isWaiting = false
                        when (result) {
                            is ClaudeChatClient.ChatResult.Success ->
                                messages = messages + ChatMessage(text = result.text, isUser = false, imageUris = emptyList())
                            is ClaudeChatClient.ChatResult.Error ->
                                errorMessage = result.message
                        }
                    }
                }
            ) {
                Icon(
                    imageVector = Icons.Filled.Send,
                    contentDescription = "전송",
                    tint = Color(0xFF9CD83B)
                )
            }
        }

        // 이미지 선택 다이얼로그 (갤러리·데이터셋 폴더)
        if (showImageSelectDialog) {
            ClaudeImageSelectDialog(
                galleryImages = galleryImageUris,
                datasetFolders = datasetFolders,
                onGalleryImageSelected = { uri ->
                    attachedImages = listOf(uri)
                    showImageSelectDialog = false
                },
                onDatasetFolderSelected = { folder ->
                    val imageExts = setOf("jpg", "jpeg", "png", "webp")
                    val uris = folder.dir.listFiles { f ->
                        f.isFile && imageExts.contains(f.extension.lowercase())
                    }?.sortedBy { it.nameWithoutExtension.toIntOrNull() ?: Int.MAX_VALUE }
                        ?.map { Uri.fromFile(it) } ?: emptyList()
                    attachedImages = uris
                    showImageSelectDialog = false
                },
                onDismiss = { showImageSelectDialog = false }
            )
        }
    }
}

@Composable
private fun ClaudeImageSelectDialog(
    galleryImages: List<Uri>,
    datasetFolders: List<DatasetFolder>,
    onGalleryImageSelected: (Uri) -> Unit,
    onDatasetFolderSelected: (DatasetFolder) -> Unit,
    onDismiss: () -> Unit
) {
    var selectedTab by remember { mutableStateOf(0) } // 0: 갤러리, 1: 데이터셋(폴더)

    Dialog(onDismissRequest = onDismiss) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.7f)
                .background(Color(0xFF1E1E1E), RoundedCornerShape(16.dp))
                .padding(16.dp)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "이미지 선택 (갤러리·데이터셋 폴더)",
                        color = Color.White,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Icon(
                        imageVector = Icons.Filled.Close,
                        contentDescription = "닫기",
                        tint = Color.White,
                        modifier = Modifier
                            .size(24.dp)
                            .clickable { onDismiss() }
                    )
                }
                Spacer(modifier = Modifier.height(12.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    listOf("갤러리 (${galleryImages.size})" to 0, "데이터셋 폴더 (${datasetFolders.size})" to 1).forEach { (label, index) ->
                        val isSelected = selectedTab == index
                        Text(
                            text = label,
                            color = if (isSelected) Color.Black else Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier
                                .clip(RoundedCornerShape(12.dp))
                                .background(
                                    if (isSelected) Color(0xFF9CD83B) else Color.White.copy(alpha = 0.2f)
                                )
                                .clickable { selectedTab = index }
                                .padding(horizontal = 14.dp, vertical = 8.dp)
                        )
                    }
                }
                Spacer(modifier = Modifier.height(12.dp))
                when (selectedTab) {
                    0 -> {
                        if (galleryImages.isEmpty()) {
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxWidth(),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "갤러리에 이미지가 없습니다.",
                                    color = Color.White.copy(alpha = 0.7f),
                                    fontSize = 14.sp
                                )
                            }
                        } else {
                            LazyVerticalGrid(
                                columns = GridCells.Fixed(3),
                                modifier = Modifier.weight(1f),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                items(galleryImages) { uri ->
                                    Image(
                                        painter = rememberAsyncImagePainter(uri),
                                        contentDescription = null,
                                        modifier = Modifier
                                            .aspectRatio(1f)
                                            .clip(RoundedCornerShape(8.dp))
                                            .clickable { onGalleryImageSelected(uri) }
                                            .border(1.dp, Color.White.copy(alpha = 0.3f), RoundedCornerShape(8.dp)),
                                        contentScale = ContentScale.Crop
                                    )
                                }
                            }
                        }
                    }
                    1 -> {
                        if (datasetFolders.isEmpty()) {
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxWidth(),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "데이터셋 폴더가 없습니다.",
                                    color = Color.White.copy(alpha = 0.7f),
                                    fontSize = 14.sp
                                )
                            }
                        } else {
                            LazyVerticalGrid(
                                columns = GridCells.Fixed(2),
                                modifier = Modifier.weight(1f),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                items(datasetFolders) { folder ->
                                    Column(
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(8.dp))
                                            .border(1.dp, Color.White.copy(alpha = 0.3f), RoundedCornerShape(8.dp))
                                            .clickable { onDatasetFolderSelected(folder) }
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .aspectRatio(1f)
                                        ) {
                                            folder.coverUri?.let { uri ->
                                                Image(
                                                    painter = rememberAsyncImagePainter(uri),
                                                    contentDescription = null,
                                                    modifier = Modifier.fillMaxSize(),
                                                    contentScale = ContentScale.Crop
                                                )
                                            }
                                            Box(
                                                modifier = Modifier
                                                    .align(Alignment.BottomCenter)
                                                    .fillMaxWidth()
                                                    .background(Color.Black.copy(alpha = 0.6f))
                                                    .padding(4.dp)
                                            ) {
                                                Text(
                                                    text = "${folder.name} (${folder.count}장)",
                                                    color = Color.White,
                                                    fontSize = 12.sp,
                                                    modifier = Modifier.fillMaxWidth(),
                                                    textAlign = TextAlign.Center
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private data class ChatMessage(
    val text: String,
    val isUser: Boolean,
    val imageUris: List<Uri> = emptyList()
)

@Composable
private fun ChatBubble(
    message: ChatMessage,
    isUser: Boolean
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 280.dp)
                .background(
                    if (isUser) Color(0xFF9CD83B) else Color(0xFF2A2A2A),
                    RoundedCornerShape(
                        topStart = 16.dp,
                        topEnd = 16.dp,
                        bottomStart = if (isUser) 16.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 16.dp
                    )
                )
                .padding(14.dp)
        ) {
            Column {
                if (message.imageUris.isNotEmpty()) {
                    val showUris = message.imageUris.take(6)
                    Column {
                        showUris.chunked(3).forEach { rowUris ->
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                rowUris.forEach { uri ->
                                    Image(
                                        painter = rememberAsyncImagePainter(uri),
                                        contentDescription = null,
                                        modifier = Modifier
                                            .weight(1f)
                                            .aspectRatio(1f)
                                            .clip(RoundedCornerShape(4.dp)),
                                        contentScale = ContentScale.Crop
                                    )
                                }
                            }
                        }
                        if (message.imageUris.size > 6) {
                            Text(
                                text = "외 ${message.imageUris.size - 6}장",
                                color = if (isUser) Color.Black.copy(alpha = 0.7f) else Color.White.copy(alpha = 0.7f),
                                fontSize = 11.sp
                            )
                        }
                    }
                    if (message.text.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                }
                if (message.text.isNotEmpty()) {
                    Text(
                        text = message.text,
                        color = if (isUser) Color.Black else Color.White,
                        fontSize = 14.sp
                    )
                }
            }
        }
    }
}

@Composable
fun CreatePlaceholderScreen() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(AppBackgroundColor)
    ) {
        Text(
            text = "창작 마당",
            color = Color.White,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
        )
    }
}

@Composable
fun ProfileScreen(
    onServerSettingsClick: () -> Unit,
    onSensorCheckClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppBackgroundColor)
    ) {
        // 타이틀: 프로필
        Text(
            text = "프로필",
            color = Color.White,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(16.dp)
        )
        
        // 구분선
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Color.White)
        )
        
        // 서버 설정 메뉴
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { onServerSettingsClick() }
                .padding(16.dp)
        ) {
            Text(
                text = "서버 설정",
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }
        
        // 구분선
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Color.White)
        )
        
        // 센서 확인 메뉴
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { onSensorCheckClick() }
                .padding(16.dp)
        ) {
            Text(
                text = "센서 확인",
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }
        
        // 마지막 구분선
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Color.White)
        )
    }
}

@Composable
fun SensorCheckScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val sensorManager = remember { context.getSystemService(Context.SENSOR_SERVICE) as SensorManager }
    
    BackHandler {
        onBack()
    }

    // 어플 구동(공간 촬영 및 기울기 측정)에 필수적인 센서만 구성
    val sensorItems = remember {
        listOf(
            // 앱에서는 GAME_ROTATION_VECTOR를 우선 사용하고, 없으면 ROTATION_VECTOR로 대체함
            SensorInfo("회전 벡터(게임/일반)", Sensor.TYPE_ROTATION_VECTOR, Icons.Default.Explore),
            SensorInfo("자이로스코프", Sensor.TYPE_GYROSCOPE, Icons.Default.Public),
            SensorInfo("가속도계", Sensor.TYPE_ACCELEROMETER, Icons.Default.Speed),
            SensorInfo("자기장 센서", Sensor.TYPE_MAGNETIC_FIELD, Icons.Default.Public)
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .padding(16.dp)
    ) {
        // 헤더
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
            horizontalArrangement = Arrangement.Start,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "센서 확인",
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                Box(
                    modifier = Modifier
                        .width(100.dp)
                        .height(2.dp)
                        .background(Color.White)
                )
            }
        }

        // 3열 그리드 레이아웃
        LazyVerticalGrid(
            columns = GridCells.Fixed(3),
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(32.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            items(sensorItems) { item ->
                val isPresent = if (item.type == Sensor.TYPE_ROTATION_VECTOR) {
                    sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR) != null ||
                        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR) != null
                } else {
                    sensorManager.getDefaultSensor(item.type) != null
                }
                SensorGridItem(item.name, item.icon, isPresent)
            }
        }
    }
}

data class SensorInfo(val name: String, val type: Int, val icon: ImageVector)

data class DatasetFolder(
    val name: String,
    val dir: File,
    val coverUri: Uri?,
    val count: Int
)

data class PlyModel(
    val name: String,
    val file: File,
    val lastModified: Long
)

@Composable
fun SensorGridItem(name: String, icon: ImageVector, isPresent: Boolean) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth()
    ) {
        Box(
            modifier = Modifier
                .size(64.dp)
                .padding(8.dp),
            contentAlignment = Alignment.Center
        ) {
            // 메인 센서 아이콘
            Icon(
                imageVector = icon,
                contentDescription = name,
                tint = if (isPresent) Color(0xFF7ED321) else Color.Red,
                modifier = Modifier.size(48.dp)
            )
            
            // 미보유 시 빨간색 금지 표시 (원 + 사선)
            if (!isPresent) {
                Canvas(modifier = Modifier.size(48.dp)) {
                    val strokeWidth = 3.dp.toPx()
                    // 빨간 원
                    drawCircle(
                        color = Color.Red,
                        style = Stroke(width = strokeWidth)
                    )
                    // 사선
                    drawLine(
                        color = Color.Red,
                        start = Offset(size.width * 0.2f, size.height * 0.2f),
                        end = Offset(size.width * 0.8f, size.height * 0.8f),
                        strokeWidth = strokeWidth
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Text(
            text = name,
            color = if (isPresent) Color(0xFF7ED321) else Color.Red,
            fontSize = 11.sp,
            textAlign = TextAlign.Center,
            lineHeight = 14.sp
        )
    }
}

@Composable
fun BottomNavigationBar(
    selectedTab: MainTab,
    onTabSelected: (MainTab) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(BottomBarBackgroundColor)
            .padding(vertical = 6.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        BottomNavItem(
            label = "라이브러리",
            icon = Icons.Filled.PhotoLibrary,
            isSelected = selectedTab == MainTab.LIBRARY,
            onClick = { onTabSelected(MainTab.LIBRARY) }
        )
        BottomNavItem(
            label = "클로드",
            icon = Icons.Filled.AutoAwesome,
            isSelected = selectedTab == MainTab.CLAUDE,
            onClick = { onTabSelected(MainTab.CLAUDE) }
        )
        BottomNavItem(
            label = "카메라",
            icon = Icons.Filled.Camera,
            isSelected = selectedTab == MainTab.CAMERA,
            onClick = { onTabSelected(MainTab.CAMERA) }
        )
        BottomNavItem(
            label = "창작마당",
            icon = Icons.Filled.Public,
            isSelected = selectedTab == MainTab.CREATE,
            onClick = { onTabSelected(MainTab.CREATE) }
        )
        BottomNavItem(
            label = "프로필",
            icon = Icons.Filled.Person,
            isSelected = selectedTab == MainTab.PROFILE,
            onClick = { onTabSelected(MainTab.PROFILE) }
        )
    }
}

@Composable
fun BottomNavItem(
    label: String,
    icon: ImageVector,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val contentColor = if (isSelected) Color.Black else Color.Black.copy(alpha = 0.6f)
    Column(
        modifier = Modifier
            .clickable { onClick() }
            .padding(horizontal = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = contentColor,
            modifier = Modifier.size(24.dp)
        )
        Text(
            text = label,
            color = contentColor,
            fontSize = 12.sp
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun GalleryScreen(
    images: List<Uri>,
    libraryTab: LibraryTab,
    onLibraryTabChange: (LibraryTab) -> Unit,
    onMediaSelected: (Uri, List<Uri>) -> Unit,
    onImageDeleted: () -> Unit
) {
    val context = LocalContext.current
    val noServerResponseMsg = "서버에 대한 응답이 없습니다.\n서버 연결을 확인해주십시오."
    var isEditMode by remember { mutableStateOf(false) }
    var selectedItems by remember { mutableStateOf<Set<Uri>>(emptySet()) }
    var isUploading by remember { mutableStateOf(false) }
    var uploadProgress by remember { mutableStateOf(0 to 0) } // (current, total)
    var uploadMessage by remember { mutableStateOf<String?>(null) }
    var uploadSourceTab by remember { mutableStateOf<LibraryTab?>(null) }
    var showUploadResultPopup by remember { mutableStateOf(false) }
    var uploadResultPopupMessage by remember { mutableStateOf<String?>(null) }
    
    var showDeleteConfirm by remember { mutableStateOf(false) }
    var showDeleteAllConfirm by remember { mutableStateOf(false) }
    var isDatasetEditMode by remember { mutableStateOf(false) }
    var selectedDatasetFolders by remember { mutableStateOf<Set<String>>(emptySet()) }
    var showDatasetDeleteConfirm by remember { mutableStateOf(false) }
    var datasetFolders by remember { mutableStateOf<List<DatasetFolder>>(emptyList()) }
    var datasetImages by remember { mutableStateOf<List<Uri>>(emptyList()) }
    var currentDatasetFolder by remember { mutableStateOf<DatasetFolder?>(null) }
    var libraryDetailScreen by remember { mutableStateOf(LibraryDetailScreen.NONE) }
    var plyModels by remember { mutableStateOf<List<PlyModel>>(emptyList()) }
    var currentPlyModel by remember { mutableStateOf<PlyModel?>(null) }

    // [추가] 내보내기/가져오기 작업 상태
    var isTransferring by remember { mutableStateOf(false) }
    val transferScope = rememberCoroutineScope()

    // [추가] 1차 배경제거 작업 상태
    var isBgRemoving by remember { mutableStateOf(false) }
    var showBgRemoveDialog by remember { mutableStateOf(false) }
    var bgRemovePrompt by remember { mutableStateOf("") }
    var bgRemovePromptError by remember { mutableStateOf(false) }

    // 3D 모델링 전송 팝업 상태
    var show3DModelingDialog by remember { mutableStateOf(false) }
    var modelingPromptText by remember { mutableStateOf("") }
    var modelingPromptError by remember { mutableStateOf(false) }
    var pending3DSourceTab by remember { mutableStateOf<LibraryTab?>(null) }
    var pending3DGalleryUris by remember { mutableStateOf<List<Uri>>(emptyList()) }
    var pending3DDatasetFolders by remember { mutableStateOf<List<String>>(emptyList()) }
    var sam3ProgressPercent by remember { mutableStateOf(0) }
    var sam3ProgressMessage by remember { mutableStateOf("") }
    var sam3ResultMessage by remember { mutableStateOf<String?>(null) }

    // [추가] 가져오기 런처 (시스템 갤러리/파일 선택)
    val importLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        if (uris.isNullOrEmpty()) {
            uploadSourceTab = LibraryTab.GALLERY
            uploadMessage = "가져오기가 취소되었습니다."
            return@rememberLauncherForActivityResult
        }
        if (isUploading || isTransferring) {
            uploadSourceTab = LibraryTab.GALLERY
            uploadMessage = "다른 작업이 진행 중입니다."
            return@rememberLauncherForActivityResult
        }
        isTransferring = true
        uploadSourceTab = LibraryTab.GALLERY
        uploadMessage = "가져오는 중..."
        transferScope.launch {
            val result = withContext(Dispatchers.IO) {
                importImagesToAppLibrary(context, uris)
            }
            uploadMessage = result.message
            isTransferring = false
            if (result.successCount > 0) {
                onImageDeleted() // 목록 갱신
            }
        }
    }

    // 뒤로가기 버튼 처리 (편집 모드에서만)
    BackHandler(enabled = isEditMode || isDatasetEditMode) {
        if (libraryTab == LibraryTab.DATASET) {
            isDatasetEditMode = false
            selectedDatasetFolders = emptySet()
        } else {
            isEditMode = false
            selectedItems = emptySet()
        }
    }

    LaunchedEffect(libraryTab) {
        if (libraryTab != LibraryTab.GALLERY && isEditMode) {
            isEditMode = false
            selectedItems = emptySet()
        }
        if (libraryTab != LibraryTab.DATASET && isDatasetEditMode) {
            isDatasetEditMode = false
            selectedDatasetFolders = emptySet()
        }
        if (libraryTab == LibraryTab.DATASET) {
            loadDatasetFolders(context) { folders ->
                datasetFolders = folders
            }
        }
        if (libraryTab == LibraryTab.MODEL_3D) {
            loadPlyModels(context) { models ->
                plyModels = models
            }
        }
    }

    // [추가] 데이터셋 탭에서 0장(빈) 폴더 주기적 자동 삭제/갱신
    // - 폴더 상세 화면에서는 폴더가 사라지면 UX가 깨질 수 있어 제외
    LaunchedEffect(libraryTab, libraryDetailScreen) {
        if (libraryTab != LibraryTab.DATASET) return@LaunchedEffect
        if (libraryDetailScreen == LibraryDetailScreen.DATASET_FOLDER) return@LaunchedEffect

        while (true) {
            val folders = withContext(Dispatchers.IO) { loadDatasetFoldersSync(context) }
            datasetFolders = folders
            delay(10_000) // 10초마다 정리/갱신
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppBackgroundColor)
    ) {
        // 헤더
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            Text(
                text = "라이브러리",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Spacer(modifier = Modifier.height(12.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                LibraryTabButton(
                    text = "3D 모델",
                    isSelected = libraryTab == LibraryTab.MODEL_3D,
                    onClick = { onLibraryTabChange(LibraryTab.MODEL_3D) }
                )
                LibraryTabButton(
                    text = "데이터셋",
                    isSelected = libraryTab == LibraryTab.DATASET,
                    onClick = { onLibraryTabChange(LibraryTab.DATASET) }
                )
                LibraryTabButton(
                    text = "갤러리",
                    isSelected = libraryTab == LibraryTab.GALLERY,
                    onClick = { onLibraryTabChange(LibraryTab.GALLERY) }
                )
                Spacer(modifier = Modifier.weight(1f))
                if (isEditMode || isDatasetEditMode) {
                    Text(
                        text = "취소",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .clip(RoundedCornerShape(12.dp))
                            .border(1.dp, Color.White, RoundedCornerShape(12.dp))
                            .clickable {
                                if (libraryTab == LibraryTab.DATASET) {
                                    isDatasetEditMode = false
                                    selectedDatasetFolders = emptySet()
                                } else {
                                    isEditMode = false
                                    selectedItems = emptySet()
                                }
                            }
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }
        }

        if (libraryTab == LibraryTab.MODEL_3D) {
            if (libraryDetailScreen == LibraryDetailScreen.OBJ_VIEWER && currentPlyModel != null) {
                BackHandler {
                    libraryDetailScreen = LibraryDetailScreen.NONE
                    currentPlyModel = null
                }
                val plyFile = currentPlyModel!!.file
                var isConverting by remember(plyFile) { mutableStateOf(true) }
                var objFile by remember(plyFile) { mutableStateOf<File?>(null) }
                var convertError by remember(plyFile) { mutableStateOf<String?>(null) }

                LaunchedEffect(plyFile) {
                    isConverting = true
                    convertError = null
                    objFile = null
                    val result = withContext(Dispatchers.IO) {
                        convertPlyToObjCached(context, plyFile)
                    }
                    val out = result.file
                    if (out == null || !out.exists()) {
                        convertError = result.error ?: "OBJ 변환에 실패했습니다."
                        isConverting = false
                    } else {
                        objFile = out
                        isConverting = false
                    }
                }

                Box(modifier = Modifier.fillMaxSize()) {
                    when {
                        isConverting -> {
                            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                Text(
                                    text = "OBJ 변환 중...",
                                    color = Color.White,
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                        convertError != null -> {
                            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                Text(
                                    text = convertError ?: "오류",
                                    color = Color.White,
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Bold,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                        objFile != null -> {
                            AndroidView(
                                factory = { ctx ->
                                    ObjSurfaceView(ctx).apply {
                                        loadModel(objFile!!)
                                    }
                                },
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    }

                    Icon(
                        imageVector = Icons.Filled.Close,
                        contentDescription = "닫기",
                        tint = Color.White,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(16.dp)
                            .size(32.dp)
                            .clickable {
                                libraryDetailScreen = LibraryDetailScreen.NONE
                                currentPlyModel = null
                            }
                    )
                }
            } else {
                if (plyModels.isEmpty()) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "3D 모델이 아직 없습니다",
                            color = Color.White.copy(alpha = 0.7f),
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                } else {
                    LazyVerticalGrid(
                        columns = GridCells.Fixed(3),
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = androidx.compose.foundation.layout.PaddingValues(8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(plyModels) { model ->
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                modifier = Modifier
                                    .clickable {
                                        currentPlyModel = model
                                        libraryDetailScreen = LibraryDetailScreen.OBJ_VIEWER
                                    }
                            ) {
                                Box(
                                    modifier = Modifier
                                        .aspectRatio(1f)
                                        .clip(RoundedCornerShape(8.dp))
                                        .background(Color(0xFF1A1A1A))
                                        .border(1.dp, Color.White.copy(alpha = 0.2f), RoundedCornerShape(8.dp)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Filled.Public,
                                        contentDescription = "3D Model",
                                        tint = Color(0xFF7ED321),
                                        modifier = Modifier.size(48.dp)
                                    )
                                }
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = model.name,
                                    color = Color.White,
                                    fontSize = 12.sp,
                                    maxLines = 1,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                    }
                }
            }
        } else if (libraryTab == LibraryTab.DATASET) {
            if (libraryDetailScreen == LibraryDetailScreen.DATASET_FOLDER && currentDatasetFolder != null) {
                BackHandler {
                    libraryDetailScreen = LibraryDetailScreen.NONE
                    currentDatasetFolder = null
                }
                if (datasetImages.isEmpty()) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "데이터셋이 아직 없습니다",
                            color = Color.White.copy(alpha = 0.7f),
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                } else {
                    LazyVerticalGrid(
                        columns = GridCells.Fixed(3),
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = androidx.compose.foundation.layout.PaddingValues(8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(datasetImages) { uri ->
                            Image(
                                painter = rememberAsyncImagePainter(uri),
                                contentDescription = "데이터셋 이미지",
                                modifier = Modifier
                                    .aspectRatio(1f)
                                    .clip(RoundedCornerShape(8.dp))
                                    .clickable { onMediaSelected(uri, datasetImages) },
                                contentScale = ContentScale.Crop
                            )
                        }
                    }
                }
            } else
            if (datasetFolders.isEmpty()) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "데이터셋이 아직 없습니다",
                        color = Color.White.copy(alpha = 0.7f),
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            } else {
                if (isDatasetEditMode) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "선택 ${selectedDatasetFolders.size}",
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 12.sp
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Icon(
                                imageVector = Icons.Filled.CloudUpload,
                                contentDescription = "업로드",
                                tint = if (selectedDatasetFolders.isNotEmpty()) Color(0xFF7ED321) else Color(0xFF7ED321).copy(alpha = 0.4f),
                                modifier = Modifier
                                    .size(32.dp)
                                    .clickable {
                                        uploadSourceTab = LibraryTab.DATASET
                                        when {
                                            isUploading -> {
                                                uploadMessage = "이미 업로드 중입니다"
                                            }
                                            selectedDatasetFolders.isEmpty() -> {
                                                uploadMessage = "선택된 폴더가 없습니다"
                                            }
                                            else -> {
                                                // 3D 모델링 프롬프트 팝업을 먼저 표시
                                                pending3DSourceTab = LibraryTab.DATASET
                                                pending3DDatasetFolders = selectedDatasetFolders.toList()
                                                modelingPromptText = ""
                                                modelingPromptError = false
                                                show3DModelingDialog = true
                                            }
                                        }
                                    }
                            )
                            Text(
                                text = "삭제",
                                color = if (selectedDatasetFolders.isNotEmpty()) Color.White else Color.White.copy(alpha = 0.4f),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (selectedDatasetFolders.isNotEmpty()) Color.Red else Color.Red.copy(alpha = 0.4f))
                                    .clickable(enabled = selectedDatasetFolders.isNotEmpty()) {
                                        showDatasetDeleteConfirm = true
                                    }
                                    .padding(horizontal = 12.dp, vertical = 6.dp)
                            )
                        }
                    }
                }
                LazyVerticalGrid(
                    columns = GridCells.Fixed(3),
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(8.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(datasetFolders) { folder ->
                        val folderPath = folder.dir.absolutePath
                        val isSelected = selectedDatasetFolders.contains(folderPath)
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Box(
                                modifier = Modifier
                                    .aspectRatio(1f)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(Color(0xFF1A1A1A))
                                    .border(1.dp, Color.White.copy(alpha = 0.6f), RoundedCornerShape(8.dp))
                                    .combinedClickable(
                                        onClick = {
                                            if (isDatasetEditMode) {
                                                selectedDatasetFolders = if (isSelected) {
                                                    selectedDatasetFolders - folderPath
                                                } else {
                                                    selectedDatasetFolders + folderPath
                                                }
                                            } else {
                                                currentDatasetFolder = folder
                                                loadDatasetImages(folder.dir) { images ->
                                                    datasetImages = images
                                                    libraryDetailScreen = LibraryDetailScreen.DATASET_FOLDER
                                                }
                                            }
                                        },
                                        onLongClick = {
                                            if (!isDatasetEditMode) {
                                                isDatasetEditMode = true
                                            }
                                            selectedDatasetFolders = if (isSelected) {
                                                selectedDatasetFolders - folderPath
                                            } else {
                                                selectedDatasetFolders + folderPath
                                            }
                                        }
                                    ),
                                contentAlignment = Alignment.Center
                            ) {
                                folder.coverUri?.let { uri ->
                                    Image(
                                        painter = rememberAsyncImagePainter(uri),
                                        contentDescription = "데이터셋 표지",
                                        modifier = Modifier.fillMaxSize(),
                                        contentScale = ContentScale.Crop
                                    )
                                }
                                if (isSelected) {
                                    Box(
                                        modifier = Modifier
                                            .matchParentSize()
                                            .background(Color.Black.copy(alpha = 0.35f))
                                    )
                                    Icon(
                                        imageVector = Icons.Filled.Check,
                                        contentDescription = "선택됨",
                                        tint = Color.White,
                                        modifier = Modifier.size(28.dp)
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = folder.name,
                                color = Color.White,
                                fontSize = 11.sp,
                                textAlign = TextAlign.Center,
                                maxLines = 2
                            )
                            Text(
                                text = "${folder.count}장",
                                color = Color.White.copy(alpha = 0.7f),
                                fontSize = 10.sp
                            )
                        }
                    }
                }
            }
        } else {
            if (isEditMode) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 8.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // 업로드 버튼
                    Icon(
                        imageVector = Icons.Filled.CloudUpload,
                        contentDescription = "업로드",
                        tint = if (selectedItems.isNotEmpty()) Color(0xFF9CD83B) else Color(0xFF9CD83B).copy(alpha = 0.4f),
                        modifier = Modifier
                            .size(20.dp)
                            .clickable {
                                uploadSourceTab = LibraryTab.GALLERY
                                when {
                                    isUploading || isTransferring -> {
                                        uploadMessage = "이미 업로드 중입니다"
                                    }
                                    selectedItems.isEmpty() -> {
                                        uploadMessage = "선택된 미디어가 없습니다"
                                    }
                                    else -> {
                                        // 3D 모델링 프롬프트 팝업을 먼저 표시
                                        pending3DSourceTab = LibraryTab.GALLERY
                                        pending3DGalleryUris = selectedItems.toList()
                                        modelingPromptText = ""
                                        modelingPromptError = false
                                        show3DModelingDialog = true
                                    }
                                }
                            }
                    )

                    // [추가] 1차 배경제거 버튼
                    Text(
                        text = "1차 배경제거",
                        color = if (selectedItems.isNotEmpty() && !isUploading && !isTransferring && !isBgRemoving) Color.White else Color.White.copy(alpha = 0.4f),
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        softWrap = false,
                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(
                                if (selectedItems.isNotEmpty() && !isUploading && !isTransferring && !isBgRemoving) Color(0xFF1A6B2F)
                                else Color(0xFF1A6B2F).copy(alpha = 0.4f)
                            )
                            .clickable(enabled = selectedItems.isNotEmpty() && !isUploading && !isTransferring && !isBgRemoving) {
                                bgRemovePrompt = ""
                                bgRemovePromptError = false
                                showBgRemoveDialog = true
                            }
                            .padding(horizontal = 6.dp, vertical = 4.dp)
                    )

                    // [추가] 내보내기 버튼 (선택된 사진을 휴대폰 갤러리로)
                    Text(
                        text = "내보내기",
                        color = if (selectedItems.isNotEmpty() && !isUploading && !isTransferring) Color.White else Color.White.copy(alpha = 0.4f),
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        softWrap = false,
                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(
                                if (selectedItems.isNotEmpty() && !isUploading && !isTransferring) Color(0xFF2F2F2F)
                                else Color(0xFF2F2F2F).copy(alpha = 0.4f)
                            )
                            .clickable(enabled = selectedItems.isNotEmpty() && !isUploading && !isTransferring) {
                                isTransferring = true
                                uploadSourceTab = LibraryTab.GALLERY
                                uploadMessage = "내보내는 중..."
                                val items = selectedItems.toList()
                                transferScope.launch {
                                    val result = withContext(Dispatchers.IO) {
                                        exportImagesToSystemGallery(context, items)
                                    }
                                    uploadMessage = result.message
                                    isTransferring = false
                                }
                            }
                            .padding(horizontal = 6.dp, vertical = 4.dp)
                    )

                    // [추가] 가져오기 버튼 (휴대폰 갤러리에서 앱으로)
                    Text(
                        text = "가져오기",
                        color = if (!isUploading && !isTransferring) Color.White else Color.White.copy(alpha = 0.4f),
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        softWrap = false,
                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(
                                if (!isUploading && !isTransferring) Color(0xFF2F2F2F)
                                else Color(0xFF2F2F2F).copy(alpha = 0.4f)
                            )
                            .clickable(enabled = !isUploading && !isTransferring) {
                                importLauncher.launch("image/*")
                            }
                            .padding(horizontal = 6.dp, vertical = 4.dp)
                    )

                    // 선택 삭제 버튼
                    Text(
                        text = "삭제",
                        color = if (selectedItems.isNotEmpty()) Color.White else Color.White.copy(alpha = 0.4f),
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        softWrap = false,
                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (selectedItems.isNotEmpty()) Color.Red else Color.Red.copy(alpha = 0.4f))
                            .clickable(enabled = !isUploading && !isTransferring && selectedItems.isNotEmpty()) {
                                showDeleteConfirm = true
                            }
                            .padding(horizontal = 6.dp, vertical = 4.dp)
                    )
                    // 전체 삭제 버튼 (빨간색 채움)
                    Text(
                        text = "전체 삭제",
                        color = Color.White,
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        softWrap = false,
                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(Color.Red)
                            .clickable(enabled = !isUploading && !isTransferring) {
                                showDeleteAllConfirm = true
                            }
                            .padding(horizontal = 6.dp, vertical = 4.dp)
                    )
                }
            }



            // 미디어 그리드
            if (images.isEmpty()) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "촬영한 미디어가 없습니다",
                        fontSize = 16.sp,
                        color = Color.White.copy(alpha = 0.6f)
                    )
                }
            } else {
                LazyVerticalGrid(
                    columns = GridCells.Fixed(3),
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(4.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(images) { mediaUri ->
                        val isVideo = isVideoUri(context, mediaUri)
                        val isSelected = selectedItems.contains(mediaUri)
                        var videoThumbnail by remember(mediaUri) { mutableStateOf<Bitmap?>(null) }

                        // 동영상 썸네일 로드
                        if (isVideo) {
                            LaunchedEffect(mediaUri) {
                                videoThumbnail = withContext(Dispatchers.IO) {
                                    try {
                                    val retriever = MediaMetadataRetriever()
                                    try {
                                        retriever.setDataSource(context, mediaUri)
                                        retriever.getFrameAtTime(0)
                                    } finally {
                                        retriever.release()
                                    }
                                    } catch (e: Exception) {
                                        e.printStackTrace()
                                        null
                                    }
                                }
                            }
                        }

                        Box(
                            modifier = Modifier
                                .aspectRatio(1f)
                                .clip(RoundedCornerShape(8.dp))
                                .border(
                                    width = if (isSelected) 4.dp else 0.dp,
                                    color = Color.Blue,
                                    shape = RoundedCornerShape(8.dp)
                                )
                                .pointerInput(mediaUri, isEditMode, isSelected) {
                                    detectTapGestures(
                                        onTap = {
                                            if (isEditMode) {
                                                // 편집 모드: 이미지 선택/해제 토글
                                                // 이미 선택된 이미지를 다시 누르면 선택 해제
                                                // 선택되지 않은 이미지를 누르면 선택
                                                selectedItems = if (isSelected) {
                                                    // 선택 해제: selectedItems에서 해당 mediaUri 제거
                                                    selectedItems - mediaUri
                                                } else {
                                                    // 선택: selectedItems에 해당 mediaUri 추가
                                                    selectedItems + mediaUri
                                                }
                                            } else {
                                                // 일반 모드: 상세 보기 화면으로 이동
                                                onMediaSelected(mediaUri, images)
                                            }
                                        },
                                        onLongPress = {
                                            // 길게 누르면 편집 모드로 전환하고 해당 이미지 선택
                                            if (!isEditMode) {
                                                isEditMode = true
                                                selectedItems = setOf(mediaUri)
                                            }
                                        }
                                    )
                                }
                        ) {
                            if (isVideo) {
                                // 동영상 썸네일 또는 배경
                                if (videoThumbnail != null) {
                                    Image(
                                        bitmap = videoThumbnail!!.asImageBitmap(),
                                        contentDescription = "동영상 썸네일",
                                        modifier = Modifier.fillMaxSize(),
                                        contentScale = ContentScale.Crop
                                    )
                                } else {
                                    // 썸네일 로딩 중 배경
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(Color.Black)
                                    )
                                    Icon(
                                        imageVector = Icons.Filled.Videocam,
                                        contentDescription = "동영상",
                                        tint = Color.White.copy(alpha = 0.7f),
                                        modifier = Modifier
                                            .align(Alignment.Center)
                                            .size(48.dp)
                                    )
                                }
                                // 동영상 아이콘 오버레이 (우측 하단)
                                Box(
                                    modifier = Modifier
                                        .align(Alignment.BottomEnd)
                                        .padding(8.dp)
                                        .background(Color.Black.copy(alpha = 0.6f), CircleShape)
                                        .padding(4.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Filled.Videocam,
                                        contentDescription = "동영상",
                                        tint = Color.White,
                                        modifier = Modifier.size(16.dp)
                                    )
                                }
                            } else {
                                // 이미지
                                Image(
                                    painter = rememberAsyncImagePainter(mediaUri),
                                    contentDescription = "촬영한 미디어",
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Crop
                                )
                            }

                            // 선택 표시
                            if (isSelected) {
                                Box(
                                    modifier = Modifier
                                        .align(Alignment.TopEnd)
                                        .padding(8.dp)
                                        .size(28.dp)
                                        .background(Color(0xFF7ED321), CircleShape),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Filled.Check,
                                        contentDescription = "선택됨",
                                        tint = Color.White,
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        // 업로드 팝업(UI) - 탭별로 분리 표시
        if (uploadSourceTab == libraryTab) {
            if (isUploading) {
                val total = uploadProgress.second.takeIf { it > 0 } ?: 100
                val current = uploadProgress.first.coerceIn(0, total)
                val percent = ((current.toFloat() / total.toFloat()) * 100f).toInt().coerceIn(0, 100)
                val progressFraction = current.toFloat() / total.toFloat()

                Dialog(onDismissRequest = { /* 전송 중에는 닫지 않음 */ }) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                            .padding(20.dp)
                    ) {
                        Column {
                            Text(
                                text = "서버로 전송 중",
                                color = Color.White,
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(10.dp))
                            Text(
                                text = "${percent}%",
                                color = Color.White,
                                fontSize = 28.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(10.dp))
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(10.dp)
                                    .background(Color.White.copy(alpha = 0.2f), RoundedCornerShape(6.dp))
                            ) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxHeight()
                                        .fillMaxWidth(progressFraction.coerceIn(0f, 1f))
                                        .background(Color(0xFF7ED321), RoundedCornerShape(6.dp))
                                )
                            }
                            Spacer(modifier = Modifier.height(10.dp))
                            Text(
                                text = uploadMessage ?: "처리 중...",
                                color = Color.White.copy(alpha = 0.85f),
                                fontSize = 13.sp
                            )
                        }
                    }
                }
            } else {
                // 업로드 결과 팝업 (완료/무응답/실패)
                if (showUploadResultPopup && uploadResultPopupMessage != null) {
                    Dialog(
                        onDismissRequest = {
                            showUploadResultPopup = false
                            uploadResultPopupMessage = null
                        }
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                                .padding(20.dp)
                        ) {
                            Column {
                                Text(
                                    text = "알림",
                                    color = Color.White,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = uploadResultPopupMessage!!,
                                    color = Color.White.copy(alpha = 0.9f),
                                    fontSize = 14.sp
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.End
                                ) {
                                    Text(
                                        text = "확인",
                                        color = Color.White,
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(12.dp))
                                            .background(Color(0xFF7ED321))
                                            .clickable {
                                                showUploadResultPopup = false
                                                uploadResultPopupMessage = null
                                            }
                                            .padding(horizontal = 16.dp, vertical = 8.dp)
                                    )
                                }
                            }
                        }
                    }
                }

                // 업로드 메시지(완료/실패)는 기존처럼 배너로 잠깐 표시
                uploadMessage?.let { message ->
                    LaunchedEffect(message) {
                        delay(3000)
                        uploadMessage = null
                    }
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(
                                if (message.contains("실패") || message.contains("20장 이상")) Color.Red.copy(alpha = 0.8f)
                                else Color.Green.copy(alpha = 0.8f)
                            )
                            .padding(16.dp)
                    ) {
                        Text(
                            text = message,
                            color = Color.White,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }

        // --- 다이얼로그 구역 (어떤 탭에서도 보일 수 있도록 탭 분기 바깥에 배치) ---

        // [추가] 1차 배경제거 진행률 팝업 (온디바이스)
        if (isBgRemoving) {
            val fraction = (sam3ProgressPercent.coerceIn(0, 100) / 100f)
            Dialog(onDismissRequest = { /* 처리 중 닫기 불가 */ }) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                        .padding(24.dp)
                ) {
                    Column {
                        Text(
                            text = "1차 배경 제거 중",
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = "MediaPipe Image Segmentation",
                            color = Color(0xFF9CD83B),
                            fontSize = 11.sp
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        // 퍼센트 숫자
                        Text(
                            text = "${sam3ProgressPercent.coerceIn(0, 100)}%",
                            color = Color(0xFF9CD83B),
                            fontSize = 32.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        // 진행 바
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(10.dp)
                                .background(Color.White.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxHeight()
                                    .fillMaxWidth(fraction)
                                    .background(Color(0xFF9CD83B), RoundedCornerShape(6.dp))
                            )
                        }
                        Spacer(modifier = Modifier.height(12.dp))
                        // 단계 메시지
                        Text(
                            text = sam3ProgressMessage,
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 13.sp
                        )
                    }
                }
            }
        }

        // [추가] SAM3 완료 결과 팝업
        sam3ResultMessage?.let { resultMsg ->
            Dialog(onDismissRequest = { sam3ResultMessage = null }) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                        .padding(24.dp)
                ) {
                    Column {
                        Text(
                            text = if (resultMsg.contains("실패")) "처리 결과" else "완료",
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = resultMsg,
                            color = Color.White.copy(alpha = 0.9f),
                            fontSize = 14.sp,
                            lineHeight = 20.sp
                        )
                        Spacer(modifier = Modifier.height(20.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End
                        ) {
                            Text(
                                text = "확인",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(
                                        if (resultMsg.contains("실패")) Color.Red
                                        else Color(0xFF1A6B2F)
                                    )
                                    .clickable { sam3ResultMessage = null }
                                    .padding(horizontal = 20.dp, vertical = 10.dp)
                            )
                        }
                    }
                }
            }
        }

        // [추가] 1차 배경제거 프롬프트 입력 다이얼로그
        if (showBgRemoveDialog) {
            val isPromptValid = !bgRemovePromptError
            val canApply = isPromptValid

            Dialog(onDismissRequest = { showBgRemoveDialog = false }) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                        .padding(20.dp)
                ) {
                    Column {
                        Text(
                            text = "어떤 사물을 추출하시겠습니까?",
                            color = Color.White,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(14.dp))
                        androidx.compose.material3.OutlinedTextField(
                            value = bgRemovePrompt,
                            onValueChange = { input ->
                                bgRemovePrompt = input
                                bgRemovePromptError = input.isNotEmpty() && !input.all { it.isLetter() && it.code < 128 || it == ' ' }
                            },
                            placeholder = {
                                Text(
                                    text = "예: cup, mouse, bottle",
                                    color = Color.White.copy(alpha = 0.35f),
                                    fontSize = 13.sp
                                )
                            },
                            singleLine = true,
                            isError = bgRemovePromptError,
                            colors = androidx.compose.material3.OutlinedTextFieldDefaults.colors(
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                focusedBorderColor = if (bgRemovePromptError) Color(0xFFFF5252) else Color(0xFF9CD83B),
                                unfocusedBorderColor = if (bgRemovePromptError) Color(0xFFFF5252) else Color.White.copy(alpha = 0.4f),
                                cursorColor = Color(0xFF9CD83B),
                                focusedContainerColor = Color(0xFF1E1E1E),
                                unfocusedContainerColor = Color(0xFF1E1E1E),
                                errorBorderColor = Color(0xFFFF5252),
                                errorContainerColor = Color(0xFF1E1E1E),
                                errorTextColor = Color.White,
                                errorCursorColor = Color(0xFFFF5252)
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                        if (bgRemovePromptError) {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "영어로 입력해 주세요",
                                color = Color(0xFFFF5252),
                                fontSize = 12.sp
                            )
                        } else {
                            Spacer(modifier = Modifier.height(6.dp + 18.sp.value.dp))
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End
                        ) {
                            Text(
                                text = "취소",
                                color = Color.White,
                                fontSize = 14.sp,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .border(1.dp, Color.White.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                                    .clickable { showBgRemoveDialog = false }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "적용",
                                color = if (canApply) Color.White else Color.White.copy(alpha = 0.35f),
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(
                                        if (canApply) Color(0xFF1A6B2F)
                                        else Color(0xFF1A6B2F).copy(alpha = 0.3f)
                                    )
                                    .clickable(enabled = canApply) {
                                        showBgRemoveDialog = false
                                        isBgRemoving = true
                                        sam3ProgressPercent = 0
                                        sam3ProgressMessage = "준비 중..."
                                        val prompt = bgRemovePrompt.trim()
                                        val items = selectedItems.toList()
                                        val total = items.size
                                        val outputDir = context.getExternalFilesDir(null)
                                        transferScope.launch {
                                            var successCount = 0
                                            var failCount = 0
                                            val savedUris = mutableListOf<Uri>()
                                            items.forEachIndexed { index, uri ->
                                                val itemLabel = if (total > 1) " (${index + 1}/$total)" else ""
                                                val basePercent = index * 95 / total
                                                val nextPercent  = (index + 1) * 95 / total
                                                val itemRange = (nextPercent - basePercent).coerceAtLeast(1)

                                                sam3ProgressPercent = basePercent
                                                sam3ProgressMessage = "이미지 로드 중...$itemLabel"

                                                // 비트맵 디코딩
                                                val bitmap = withContext(Dispatchers.IO) {
                                                    decodeBitmapWithMaxDimension(context, uri, 1024)
                                                }
                                                if (bitmap == null) {
                                                    failCount++
                                                    return@forEachIndexed
                                                }

                                                sam3ProgressPercent = basePercent + itemRange / 8
                                                sam3ProgressMessage = "MediaPipe 세그멘테이션 준비 중...$itemLabel"

                                                // MediaPipe Image Segmentation 기반 온디바이스 배경 제거
                                                val result = withContext(Dispatchers.IO) {
                                                    BackgroundRemovalProcessor.removeBackground(
                                                        context    = context,
                                                        sourceBitmap = bitmap,
                                                        userPrompt = prompt,
                                                        outputDir  = outputDir,
                                                        onProgress = { iter, totalIter ->
                                                            if (totalIter > 0) {
                                                                val frac = iter.toFloat() / totalIter
                                                                val mapped = basePercent +
                                                                    (itemRange * 0.15f + frac * itemRange * 0.75f).toInt()
                                                                sam3ProgressPercent = mapped.coerceIn(basePercent, nextPercent - 1)
                                                                sam3ProgressMessage =
                                                                    "MediaPipe 처리 중 ${"%.0f".format(frac * 100)}%$itemLabel"
                                                            }
                                                        }
                                                    )
                                                }
                                                bitmap.recycle()

                                                when (result) {
                                                    is BackgroundRemovalProcessor.Result.Success -> {
                                                        result.savedUri?.let { savedUris.add(it) }
                                                        successCount++
                                                        sam3ProgressPercent = nextPercent
                                                        sam3ProgressMessage = "완료$itemLabel"
                                                    }
                                                    is BackgroundRemovalProcessor.Result.Error -> {
                                                        failCount++
                                                        sam3ProgressMessage = "오류: ${result.message}"
                                                    }
                                                }
                                            }
                                            sam3ProgressPercent = 100
                                            isBgRemoving = false
                                            sam3ResultMessage = when {
                                                successCount > 0 && failCount == 0 ->
                                                    "배경 제거 완료\n${successCount}장이 앱 갤러리에 저장되었습니다."
                                                successCount > 0 ->
                                                    "배경 제거 완료\n${successCount}장 성공, ${failCount}장 실패"
                                                else ->
                                                    "배경 제거 실패\n객체를 찾지 못했거나 모델이 없습니다.\n(assets/models/ 폴더를 확인하세요)"
                                            }
                                            if (successCount > 0) {
                                                selectedItems = emptySet()
                                                isEditMode = false
                                                onImageDeleted()
                                            }
                                        }
                                    }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                        }
                    }
                }
            }
        }

        // 3D 모델링 프롬프트 입력 다이얼로그
        if (show3DModelingDialog) {
            val isValid3DPrompt = modelingPromptText.isNotEmpty() && !modelingPromptError
            val noServerResponseMsg = "서버에 대한 응답이 없습니다.\n서버 연결을 확인해주십시오."

            Dialog(onDismissRequest = { show3DModelingDialog = false }) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                        .padding(20.dp)
                ) {
                    Column {
                        Text(
                            text = "3D 모델링",
                            color = Color(0xFF9CD83B),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "어떤 사물을 3D 모델링하시겠습니까?",
                            color = Color.White,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(14.dp))
                        androidx.compose.material3.OutlinedTextField(
                            value = modelingPromptText,
                            onValueChange = { input ->
                                modelingPromptText = input
                                // 영문자, 숫자, 스페이스, 특수기호(ASCII 0x20~0x7E)만 허용
                                modelingPromptError = input.isNotEmpty() && !input.all { it.code in 0x20..0x7E }
                            },
                            placeholder = {
                                Text(
                                    text = "예: gundam figure, white cup",
                                    color = Color.White.copy(alpha = 0.35f),
                                    fontSize = 13.sp
                                )
                            },
                            singleLine = true,
                            isError = modelingPromptError,
                            colors = androidx.compose.material3.OutlinedTextFieldDefaults.colors(
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                focusedBorderColor = if (modelingPromptError) Color(0xFFFF5252) else Color(0xFF9CD83B),
                                unfocusedBorderColor = if (modelingPromptError) Color(0xFFFF5252) else Color.White.copy(alpha = 0.4f),
                                cursorColor = Color(0xFF9CD83B),
                                focusedContainerColor = Color(0xFF1E1E1E),
                                unfocusedContainerColor = Color(0xFF1E1E1E),
                                errorBorderColor = Color(0xFFFF5252),
                                errorContainerColor = Color(0xFF1E1E1E),
                                errorTextColor = Color.White,
                                errorCursorColor = Color(0xFFFF5252)
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                        if (modelingPromptError) {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "영어, 숫자, 특수기호만 입력 가능합니다 (한국어 불가)",
                                color = Color(0xFFFF5252),
                                fontSize = 12.sp
                            )
                        } else {
                            Spacer(modifier = Modifier.height(6.dp + 18.sp.value.dp))
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End
                        ) {
                            Text(
                                text = "취소",
                                color = Color.White,
                                fontSize = 14.sp,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .border(1.dp, Color.White.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                                    .clickable { show3DModelingDialog = false }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "전송",
                                color = if (isValid3DPrompt) Color.White else Color.White.copy(alpha = 0.35f),
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(
                                        if (isValid3DPrompt) Color(0xFF1B4F8A)
                                        else Color(0xFF1B4F8A).copy(alpha = 0.3f)
                                    )
                                    .clickable(enabled = isValid3DPrompt) {
                                        val promptSnapshot = modelingPromptText.trim()
                                        show3DModelingDialog = false
                                        uploadSourceTab = pending3DSourceTab

                                        when (pending3DSourceTab) {
                                            LibraryTab.GALLERY -> {
                                                val urisSnapshot = pending3DGalleryUris
                                                isUploading = true
                                                uploadProgress = 0 to 100
                                                uploadMessage = "업로드 준비 중..."
                                                CoroutineScope(Dispatchers.IO).launch {
                                                    try {
                                                        val zipFile = createZipFromUris(
                                                            context = context,
                                                            uris = urisSnapshot,
                                                            zipPrefix = "media"
                                                        )
                                                        val success = zipFile != null && uploadZipAndRunPipeline(
                                                            context = context,
                                                            zipFile = zipFile,
                                                            prompt = promptSnapshot,
                                                            onProgress = { p, msg ->
                                                                CoroutineScope(Dispatchers.Main).launch {
                                                                    uploadProgress = p to 100
                                                                    uploadMessage = msg
                                                                }
                                                            }
                                                        )
                                                        withContext(Dispatchers.Main) {
                                                            isUploading = false
                                                            if (success) {
                                                                uploadResultPopupMessage = "서버에 업로드가 완료되었습니다."
                                                                showUploadResultPopup = true
                                                                selectedItems = emptySet()
                                                                isEditMode = false
                                                            } else {
                                                                uploadResultPopupMessage =
                                                                    if (uploadMessage == noServerResponseMsg) noServerResponseMsg else "업로드 실패"
                                                                showUploadResultPopup = true
                                                                if (uploadMessage != noServerResponseMsg) uploadMessage = "업로드 실패"
                                                            }
                                                        }
                                                    } catch (e: Exception) {
                                                        e.printStackTrace()
                                                        withContext(Dispatchers.Main) {
                                                            isUploading = false
                                                            uploadMessage = "업로드 실패"
                                                        }
                                                    }
                                                }
                                            }
                                            LibraryTab.DATASET -> {
                                                val foldersSnapshot = pending3DDatasetFolders
                                                isUploading = true
                                                uploadProgress = 0 to 100
                                                uploadMessage = "업로드 준비 중..."
                                                CoroutineScope(Dispatchers.IO).launch {
                                                    try {
                                                        val folderFiles = foldersSnapshot
                                                            .map { File(it) }
                                                            .filter { it.exists() && it.isDirectory }
                                                        val zipFile = createZipFromFolders(
                                                            context = context,
                                                            folders = folderFiles,
                                                            zipPrefix = "dataset"
                                                        )
                                                        val success = zipFile != null && uploadZipAndRunPipeline(
                                                            context = context,
                                                            zipFile = zipFile,
                                                            prompt = promptSnapshot,
                                                            onProgress = { p, msg ->
                                                                CoroutineScope(Dispatchers.Main).launch {
                                                                    uploadProgress = p to 100
                                                                    uploadMessage = msg
                                                                }
                                                            }
                                                        )
                                                        withContext(Dispatchers.Main) {
                                                            isUploading = false
                                                            if (success) {
                                                                uploadResultPopupMessage = "서버에 업로드가 완료되었습니다."
                                                                showUploadResultPopup = true
                                                                selectedDatasetFolders = emptySet()
                                                                isDatasetEditMode = false
                                                            } else {
                                                                uploadResultPopupMessage =
                                                                    if (uploadMessage == noServerResponseMsg) noServerResponseMsg else "업로드 실패"
                                                                showUploadResultPopup = true
                                                                if (uploadMessage != noServerResponseMsg) uploadMessage = "업로드 실패"
                                                            }
                                                        }
                                                    } catch (e: Exception) {
                                                        e.printStackTrace()
                                                        withContext(Dispatchers.Main) {
                                                            isUploading = false
                                                            uploadMessage = "업로드 실패"
                                                        }
                                                    }
                                                }
                                            }
                                            else -> {}
                                        }
                                    }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                        }
                    }
                }
            }
        }

        if (showDeleteConfirm) {
            Dialog(onDismissRequest = { showDeleteConfirm = false }) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                        .padding(20.dp)
                ) {
                    Column {
                        Text(
                            text = "삭제 확인",
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "정말로 삭제하겠습니까?",
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 14.sp
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End
                        ) {
                            Text(
                                text = "취소",
                                color = Color.White,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .border(1.dp, Color.White.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                                    .clickable { showDeleteConfirm = false }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "삭제",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color.Red)
                                    .clickable {
                                        val itemsToDelete = selectedItems.toList()
                                        itemsToDelete.forEach { uri ->
                                            deleteMediaByUri(context, uri)
                                        }
                                        selectedItems = emptySet()
                                        isEditMode = false
                                        showDeleteConfirm = false
                                        onImageDeleted()
                                    }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                        }
                    }
                }
            }
        }

        if (showDeleteAllConfirm) {
            Dialog(onDismissRequest = { showDeleteAllConfirm = false }) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                        .padding(20.dp)
                ) {
                    Column {
                        Text(
                            text = "전체 삭제",
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "정말로 모든 미디어를 삭제하겠습니까?",
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 14.sp
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End
                        ) {
                            Text(
                                text = "취소",
                                color = Color.White,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .border(1.dp, Color.White.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                                    .clickable { showDeleteAllConfirm = false }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "삭제",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color.Red)
                                    .clickable {
                                        deleteAllMedia(context)
                                        selectedItems = emptySet()
                                        isEditMode = false
                                        showDeleteAllConfirm = false
                                        onImageDeleted()
                                    }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                        }
                    }
                }
            }
        }

        if (showDatasetDeleteConfirm) {
            Dialog(onDismissRequest = { showDatasetDeleteConfirm = false }) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                        .padding(20.dp)
                ) {
                    Column {
                        Text(
                            text = "폴더 삭제",
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "정말로 삭제하시겠습니까?",
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 14.sp
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End
                        ) {
                            Text(
                                text = "취소",
                                color = Color.White,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .border(1.dp, Color.White.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                                    .clickable { showDatasetDeleteConfirm = false }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "삭제",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color.Red)
                                    .clickable {
                                        val pathsToDelete = selectedDatasetFolders.toList()
                                        pathsToDelete.forEach { path ->
                                            try {
                                                File(path).deleteRecursively()
                                            } catch (e: Exception) {
                                                e.printStackTrace()
                                            }
                                        }
                                        selectedDatasetFolders = emptySet()
                                        isDatasetEditMode = false
                                        showDatasetDeleteConfirm = false
                                        if (currentDatasetFolder != null &&
                                            pathsToDelete.contains(currentDatasetFolder!!.dir.absolutePath)
                                        ) {
                                            currentDatasetFolder = null
                                            datasetImages = emptyList()
                                            libraryDetailScreen = LibraryDetailScreen.NONE
                                        }
                                        loadDatasetFolders(context) { folders ->
                                            datasetFolders = folders
                                        }
                                    }
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}
@Composable
fun MediaDetailScreen(
    mediaList: List<Uri>,
    initialIndex: Int,
    onBack: () -> Unit,
    onMediaChanged: (Int) -> Unit,
    onGalleryUpdated: () -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    // 내부 변경 가능한 리스트 (삭제 시 즉시 반영)
    val mutableMediaList = remember(mediaList) { mediaList.toMutableStateList() }
    var currentIndex by remember { mutableStateOf(initialIndex.coerceIn(0, mediaList.size - 1)) }

    // 핀치 줌 / 패닝 상태
    var scale by remember { mutableStateOf(1f) }
    var offsetX by remember { mutableStateOf(0f) }
    var offsetY by remember { mutableStateOf(0f) }

    // 필름스트립 스크롤 상태
    val filmstripState = rememberLazyListState()

    // 옵션 바 상태
    var showDeleteConfirm by remember { mutableStateOf(false) }
    var actionMessage by remember { mutableStateOf<String?>(null) }
    var showImageEdit by remember { mutableStateOf(false) }

    // 인덱스 변경 시 콜백 + 줌 리셋 + 필름스트립 자동 스크롤
    LaunchedEffect(currentIndex) {
        onMediaChanged(currentIndex)
        scale = 1f; offsetX = 0f; offsetY = 0f
        if (mutableMediaList.size > 1) {
            filmstripState.animateScrollToItem(maxOf(0, currentIndex - 2))
        }
    }

    // 액션 메시지 3초 후 자동 소거
    LaunchedEffect(actionMessage) {
        if (actionMessage != null) {
            kotlinx.coroutines.delay(3000)
            actionMessage = null
        }
    }

    val currentMediaUri = if (currentIndex in mutableMediaList.indices) mutableMediaList[currentIndex] else null
    val isVideo = currentMediaUri?.let { isVideoUri(context, it) } ?: false

    BackHandler { onBack() }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // ─── 메인 미디어 영역 (필름스트립 + 옵션 바 위쪽) ────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = if (mutableMediaList.size > 1) 168.dp else 64.dp)
                .fillMaxHeight()
        ) {
            if (currentMediaUri != null) {
                if (isVideo) {
                    // 동영상 재생
                    var videoViewRef by remember { mutableStateOf<VideoView?>(null) }
                    LaunchedEffect(currentMediaUri) {
                        videoViewRef?.let { view ->
                            view.stopPlayback()
                            view.setVideoURI(currentMediaUri)
                            val mc = MediaController(context)
                            mc.setAnchorView(view)
                            view.setMediaController(mc)
                            view.start()
                        }
                    }
                    AndroidView(
                        factory = { ctx ->
                            VideoView(ctx).apply {
                                setVideoURI(currentMediaUri)
                                val mc = MediaController(ctx)
                                mc.setAnchorView(this)
                                setMediaController(mc)
                                start()
                                videoViewRef = this
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                        update = { view -> videoViewRef = view }
                    )
                } else {
                    // 이미지: 핀치 줌(2손가락) + 패닝(줌된 상태) + 스와이프 내비게이션(1손가락·줌 해제)
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .pointerInput(currentIndex) {
                                // size는 PointerInputScope의 프로퍼티 → awaitPointerEventScope 진입 전 캡처
                                val viewWidth  = size.width.toFloat()
                                val viewHeight = size.height.toFloat()

                                // 오프셋 경계 클램핑 헬퍼:
                                // scale 배율에서 이미지가 뷰 밖으로 나가지 않도록 최대 이동 범위를 제한
                                // maxOffset = (scale - 1) × viewDim / 2
                                fun clampOffsets() {
                                    val maxX = ((scale - 1f) * viewWidth  / 2f).coerceAtLeast(0f)
                                    val maxY = ((scale - 1f) * viewHeight / 2f).coerceAtLeast(0f)
                                    offsetX = offsetX.coerceIn(-maxX, maxX)
                                    offsetY = offsetY.coerceIn(-maxY, maxY)
                                }

                                // 단일 awaitPointerEventScope 블록으로 핀치줌+스와이프를 모두 처리
                                awaitPointerEventScope {
                                    while (true) {
                                        awaitFirstDown(requireUnconsumed = false)

                                        var dragX = 0f
                                        var pinching = false
                                        var prevDist = 0f

                                        // 손가락이 모두 떼어질 때까지 이벤트 처리
                                        while (true) {
                                            val event = awaitPointerEvent()
                                            val pressed = event.changes.filter { it.pressed }
                                            if (pressed.isEmpty()) break

                                            if (pressed.size >= 2) {
                                                // ─ 두 손가락: 핀치 줌 + 패닝 ─
                                                pinching = true
                                                val p1 = pressed[0]; val p2 = pressed[1]
                                                val dx = p1.position.x - p2.position.x
                                                val dy = p1.position.y - p2.position.y
                                                val dist = kotlin.math.sqrt(dx * dx + dy * dy)
                                                if (prevDist > 0f && dist > 0f) {
                                                    scale = (scale * dist / prevDist).coerceIn(1f, 5f)
                                                    if (scale > 1f) {
                                                        val cx  = (p1.position.x + p2.position.x) / 2f
                                                        val pcx = (p1.previousPosition.x + p2.previousPosition.x) / 2f
                                                        val cy  = (p1.position.y + p2.position.y) / 2f
                                                        val pcy = (p1.previousPosition.y + p2.previousPosition.y) / 2f
                                                        offsetX += cx - pcx
                                                        offsetY += cy - pcy
                                                    } else {
                                                        offsetX = 0f; offsetY = 0f
                                                    }
                                                }
                                                prevDist = dist
                                                // 핀치 후 경계 클램핑 적용
                                                clampOffsets()
                                                pressed.forEach { it.consume() }

                                            } else if (pressed.size == 1 && !pinching) {
                                                // ─ 한 손가락: 줌 상태이면 패닝, 아니면 내비게이션 스와이프 ─
                                                val delta = pressed[0].position - pressed[0].previousPosition
                                                if (scale > 1.05f) {
                                                    offsetX += delta.x
                                                    offsetY += delta.y
                                                    // 패닝 후 경계 클램핑 적용
                                                    clampOffsets()
                                                } else {
                                                    dragX += delta.x
                                                }
                                                pressed[0].consume()
                                            }
                                        }

                                        // 스와이프 내비게이션 판정 (핀치 없음 + 줌 해제 상태)
                                        if (!pinching && scale <= 1.05f) {
                                            val threshold = viewWidth * 0.3f
                                            when {
                                                dragX > threshold && currentIndex > 0 -> currentIndex--
                                                dragX < -threshold && currentIndex < mediaList.size - 1 -> currentIndex++
                                            }
                                        }
                                        // scale 1 미만 방지 + 최종 경계 클램핑
                                        if (scale < 1f) { scale = 1f; offsetX = 0f; offsetY = 0f }
                                        else clampOffsets()
                                    }
                                }
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Image(
                            painter = rememberAsyncImagePainter(currentMediaUri),
                            contentDescription = "미디어 상세",
                            modifier = Modifier
                                .fillMaxSize()
                                .graphicsLayer {
                                    scaleX = scale
                                    scaleY = scale
                                    translationX = offsetX
                                    translationY = offsetY
                                },
                            contentScale = ContentScale.Fit
                        )
                    }
                }
            }
        }

        // ─── 하단: 필름스트립 + 옵션 바 ────────────────────────────────────────
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
        ) {
            // 필름스트립 (2장 이상일 때)
            if (mutableMediaList.size > 1) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color.Black)
                        .padding(vertical = 8.dp)
                ) {
                    LazyRow(
                        state = filmstripState,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(88.dp),
                        contentPadding = PaddingValues(horizontal = 20.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        itemsIndexed(mutableMediaList) { idx, uri ->
                            val isCurrent = idx == currentIndex
                            Box(
                                modifier = Modifier
                                    .size(if (isCurrent) 74.dp else 56.dp)
                                    .clip(RoundedCornerShape(6.dp))
                                    .border(
                                        width = if (isCurrent) 2.dp else 0.dp,
                                        color = if (isCurrent) Color(0xFF9CD83B) else Color.Transparent,
                                        shape = RoundedCornerShape(6.dp)
                                    )
                                    .clickable(
                                        interactionSource = remember { MutableInteractionSource() },
                                        indication = null
                                    ) {
                                        scope.launch { currentIndex = idx }
                                    }
                            ) {
                                Image(
                                    painter = rememberAsyncImagePainter(uri),
                                    contentDescription = null,
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Crop,
                                    alpha = if (isCurrent) 1f else 0.55f
                                )
                            }
                        }
                    }
                }
            }

            // ─── 옵션 바 (항상 표시) ──────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color.Black)
                    .padding(vertical = 14.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // 이미지 편집 (연필 아이콘)
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null
                        ) { if (currentMediaUri != null && !isVideo) showImageEdit = true },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Edit,
                        contentDescription = "이미지 편집",
                        tint = Color.White,
                        modifier = Modifier.size(26.dp)
                    )
                }
                // 구분선
                Box(modifier = Modifier.width(0.5.dp).height(28.dp).background(Color(0xFF444444)))
                // 2차 사물 배경 분리 (브러시 아이콘)
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null
                        ) { /* TODO: 2차 사물 배경 분리 */ },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Brush,
                        contentDescription = "2차 사물 배경 분리",
                        tint = Color.White,
                        modifier = Modifier.size(26.dp)
                    )
                }
                // 구분선
                Box(modifier = Modifier.width(0.5.dp).height(28.dp).background(Color(0xFF444444)))
                // 이미지 내보내기 (업로드 아이콘)
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null
                        ) {
                            currentMediaUri?.let { uri ->
                                scope.launch {
                                    val result = exportImagesToSystemGallery(context, listOf(uri))
                                    actionMessage = if (result.successCount > 0) "갤러리에 내보냈습니다" else "내보내기 실패"
                                }
                            }
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.FileUpload,
                        contentDescription = "이미지 내보내기",
                        tint = Color.White,
                        modifier = Modifier.size(26.dp)
                    )
                }
                // 구분선
                Box(modifier = Modifier.width(0.5.dp).height(28.dp).background(Color(0xFF444444)))
                // 이미지 삭제 (휴지통 아이콘)
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null
                        ) { showDeleteConfirm = true },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "이미지 삭제",
                        tint = Color.White,
                        modifier = Modifier.size(26.dp)
                    )
                }
            }
        }

        // 액션 메시지 토스트 (내보내기 결과 등)
        actionMessage?.let { msg ->
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 180.dp)
                    .background(Color(0xCC000000), RoundedCornerShape(20.dp))
                    .padding(horizontal = 20.dp, vertical = 10.dp)
            ) {
                Text(msg, color = Color.White, fontSize = 14.sp)
            }
        }

        // 이미지 편집 화면: 외부 Box 안에서 오버레이로 표시해야 전체 화면 덮기 가능
        if (showImageEdit && currentMediaUri != null) {
            ImageEditScreen(
                imageUri = currentMediaUri,
                onBack = { showImageEdit = false },
                onSaved = { newUri ->
                    if (newUri != currentMediaUri) {
                        mutableMediaList[currentIndex] = newUri
                    }
                    onGalleryUpdated()
                    showImageEdit = false
                }
            )
        }
    }

    // 삭제 확인 다이얼로그
    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text("이미지 삭제") },
            text = { Text("이 이미지를 삭제하시겠습니까?") },
            confirmButton = {
                TextButton(onClick = {
                    showDeleteConfirm = false
                    currentMediaUri?.let { uri ->
                        scope.launch {
                            try {
                                val file = java.io.File(uri.path ?: "")
                                if (file.exists()) file.delete()
                            } catch (e: Exception) { e.printStackTrace() }
                            val removedIdx = currentIndex
                            mutableMediaList.removeAt(removedIdx)
                            if (mutableMediaList.isEmpty()) {
                                onGalleryUpdated()
                                onBack()
                            } else {
                                currentIndex = removedIdx.coerceIn(0, mutableMediaList.size - 1)
                                onGalleryUpdated()
                            }
                        }
                    }
                }) {
                    Text("삭제", color = Color(0xFFFF6B6B))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) {
                    Text("취소", color = Color(0xFFAAAAAA))
                }
            },
            containerColor = Color(0xFF2A2A2A),
            titleContentColor = Color.White,
            textContentColor = Color(0xFFCCCCCC)
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 자르기 모드 지원 타입
// ──────────────────────────────────────────────────────────────────────────────
private enum class CropHandle { NONE, MOVE, NW, NE, SW, SE, N, E, S, W }

private data class PendingCropData(
    val left: Float,
    val top: Float,
    val width: Float,
    val height: Float,
    val imgAreaTopPx: Float,
    val imgAreaHeightPx: Float,
    val containerW: Float,
    val containerH: Float,
    val cropZoom: Float = 1f,
    val cropPanX: Float = 0f,
    val cropPanY: Float = 0f
)

// ──────────────────────────────────────────────────────────────────────────────
// 이미지 편집 화면
// ──────────────────────────────────────────────────────────────────────────────
@Composable
fun ImageEditScreen(
    imageUri: Uri,
    onBack: () -> Unit,
    onSaved: (Uri) -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // 변환 상태
    var buttonRotation by remember { mutableStateOf(0f) }
    var dialDegrees    by remember { mutableStateOf(0f) }
    var isFlipped      by remember { mutableStateOf(false) }
    var pendingCrop    by remember { mutableStateOf<PendingCropData?>(null) }
    var isCropMode     by remember { mutableStateOf(false) }
    // 자르기 모드 전용 줌/패닝 상태
    var cropZoom  by remember { mutableStateOf(1f) }
    var cropPanX  by remember { mutableStateOf(0f) }
    var cropPanY  by remember { mutableStateOf(0f) }

    val totalRotation = buttonRotation + dialDegrees
    val hasChanges    = buttonRotation != 0f || dialDegrees != 0f || isFlipped || pendingCrop != null

    var showSaveDialog by remember { mutableStateOf(false) }
    var isSaving       by remember { mutableStateOf(false) }

    fun requestBack() { if (hasChanges) showSaveDialog = true else onBack() }

    BackHandler { requestBack() }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // ── 메인 이미지 (자르기 모드 여부에 따라 하단 패딩 다름) ──
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 56.dp, bottom = if (isCropMode) 60.dp else 148.dp)
                .fillMaxHeight(),
            contentAlignment = Alignment.Center
        ) {
            Image(
                painter = rememberAsyncImagePainter(imageUri),
                contentDescription = null,
                modifier = Modifier
                    .fillMaxSize()
                    .graphicsLayer {
                        rotationZ    = totalRotation
                        val flipSign = if (isFlipped) -1f else 1f
                        scaleX       = flipSign * if (isCropMode) cropZoom else 1f
                        scaleY       = if (isCropMode) cropZoom else 1f
                        translationX = if (isCropMode) cropPanX else 0f
                        translationY = if (isCropMode) cropPanY else 0f
                    },
                contentScale = ContentScale.Fit
            )
        }

        // ── 상단 바: X | 이미지 편집 | 원본 복원  저장 ──
        Row(
            modifier = Modifier
                .align(Alignment.TopStart)
                .fillMaxWidth()
                .background(Color.Black)
                .padding(horizontal = 4.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = { requestBack() }) {
                Icon(Icons.Default.Close, contentDescription = "닫기", tint = Color.White)
            }
            Spacer(modifier = Modifier.weight(1f))
            // 원본 복원 버튼
            TextButton(
                onClick = {
                    buttonRotation = 0f
                    dialDegrees    = 0f
                    isFlipped      = false
                },
                enabled = hasChanges
            ) {
                Text(
                    "원본 복원",
                    color    = if (hasChanges) Color(0xFFCCCCCC) else Color(0xFF555555),
                    fontSize = 14.sp
                )
            }
            // 저장 버튼
            TextButton(
                onClick  = { showSaveDialog = true },
                enabled  = hasChanges
            ) {
                Text(
                    "저장",
                    color      = if (hasChanges) Color.White else Color(0xFF555555),
                    fontSize   = 14.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }

        // ── 하단: 다이얼 바 + 기능 버튼 (자르기 모드가 아닐 때만) ──
        if (!isCropMode) {
            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .background(Color.Black)
            ) {
                EditDialBar(
                    degrees = dialDegrees,
                    onDegreesChanged = { dialDegrees = it }
                )
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color.Black)
                        .padding(vertical = 16.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    EditActionButton(icon = Icons.Default.RotateLeft, label = "회전") {
                        buttonRotation = ((buttonRotation - 90f) % 360f + 360f) % 360f
                    }
                    EditActionButton(icon = Icons.Default.Flip, label = "좌우반전") {
                        isFlipped = !isFlipped
                    }
                    EditActionButton(icon = Icons.Default.Crop, label = "자르기") {
                        isCropMode = true
                    }
                    EditActionButton(icon = Icons.Default.AutoFixHigh, label = "누끼따기") {
                        /* TODO: 누끼따기 */
                    }
                }
            }
        }

        // ── 자르기 모드 오버레이 ──
        if (isCropMode) {
            CropModeOverlay(
                imageUri          = imageUri,
                topBarDp          = 56.dp,
                bottomBarDp       = 60.dp,
                buttonRotation    = buttonRotation,
                cropZoom          = cropZoom,
                cropPanX          = cropPanX,
                cropPanY          = cropPanY,
                onZoomChange      = { z, px, py ->
                    cropZoom = z; cropPanX = px; cropPanY = py
                },
                onConfirm         = { crop ->
                    pendingCrop    = crop
                    isCropMode     = false
                    cropZoom = 1f; cropPanX = 0f; cropPanY = 0f
                    showSaveDialog = true
                },
                onRestoreOriginal = {
                    pendingCrop    = null
                    buttonRotation = 0f
                    dialDegrees    = 0f
                    isFlipped      = false
                    cropZoom = 1f; cropPanX = 0f; cropPanY = 0f
                    isCropMode     = false
                },
                onCancel = {
                    cropZoom = 1f; cropPanX = 0f; cropPanY = 0f
                    isCropMode = false
                }
            )
        }

        // 저장 중 오버레이
        if (isSaving) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.65f)),
                contentAlignment = Alignment.Center
            ) {
                androidx.compose.material3.CircularProgressIndicator(color = Color(0xFF9CD83B))
            }
        }
    }

    // ── 저장 확인 다이얼로그 ──
    if (showSaveDialog) {
        androidx.compose.ui.window.Dialog(onDismissRequest = { showSaveDialog = false }) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF2A2A2A), RoundedCornerShape(16.dp))
                    .padding(horizontal = 8.dp, vertical = 20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    "변경된 이미지를 저장하시겠습니까?",
                    color = Color.White,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 16.dp)
                )
                Spacer(Modifier.height(16.dp))
                HorizontalDivider(color = Color(0xFF444444))
                // 편집 이미지로 저장 (원본 덮어쓰기)
                TextButton(
                    onClick = {
                        showSaveDialog = false
                        scope.launch {
                            isSaving = true
                            val saved = saveEditedImage(context, imageUri, totalRotation, isFlipped, overwrite = true, cropData = pendingCrop)
                            isSaving = false
                            onSaved(saved ?: imageUri)
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("편집 이미지로 저장", color = Color(0xFF9CD83B), fontSize = 15.sp)
                }
                HorizontalDivider(color = Color(0xFF444444))
                // 복사본으로 저장
                TextButton(
                    onClick = {
                        showSaveDialog = false
                        scope.launch {
                            isSaving = true
                            val saved = saveEditedImage(context, imageUri, totalRotation, isFlipped, overwrite = false, cropData = pendingCrop)
                            isSaving = false
                            onSaved(saved ?: imageUri)
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("복사본으로 저장", color = Color.White, fontSize = 15.sp)
                }
                HorizontalDivider(color = Color(0xFF444444))
                // 저장하지 않음
                TextButton(
                    onClick = { showSaveDialog = false; onBack() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("저장하지 않음", color = Color(0xFFAAAAAA), fontSize = 15.sp)
                }
            }
        }
    }
}

// ── 다이얼 바: 50칸 / 양끝 페이드 아웃 / 검은 배경 ────────────────────────
// ──────────────────────────────────────────────────────────────────────────────
// 자르기 오버레이 (이미지 위에 전체 화면으로 덮음)
// ──────────────────────────────────────────────────────────────────────────────
@Composable
private fun CropModeOverlay(
    imageUri: Uri,
    topBarDp: androidx.compose.ui.unit.Dp,
    bottomBarDp: androidx.compose.ui.unit.Dp,
    buttonRotation: Float,
    cropZoom: Float,
    cropPanX: Float,
    cropPanY: Float,
    onZoomChange: (zoom: Float, panX: Float, panY: Float) -> Unit,
    onConfirm: (PendingCropData) -> Unit,
    onRestoreOriginal: () -> Unit,
    onCancel: () -> Unit
) {
    val context = LocalContext.current
    val density = LocalDensity.current
    // pointerInput 안에서 항상 최신값을 읽기 위한 래퍼
    val latestZoom = androidx.compose.runtime.rememberUpdatedState(cropZoom)
    val latestPanX = androidx.compose.runtime.rememberUpdatedState(cropPanX)
    val latestPanY = androidx.compose.runtime.rememberUpdatedState(cropPanY)

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val containerW = with(density) { maxWidth.toPx() }
        val containerH = with(density) { maxHeight.toPx() }
        val topPx      = with(density) { topBarDp.toPx() }
        val imgAreaTop = topPx
        val imgAreaH   = containerH - topPx - with(density) { bottomBarDp.toPx() }

        // 실제 이미지 표시 경계 (이미지 로드 후 업데이트)
        var dispLeft   by remember { mutableStateOf(0f) }
        var dispTop    by remember { mutableStateOf(imgAreaTop) }
        var dispRight  by remember { mutableStateOf(containerW) }
        var dispBottom by remember { mutableStateOf(imgAreaTop + imgAreaH) }

        // 자르기 사각형 상태 (dispLeft/dispTop 기준 초기값)
        var cropLeft  by remember { mutableStateOf(0f) }
        var cropTop   by remember { mutableStateOf(imgAreaTop) }
        var cropW     by remember { mutableStateOf(containerW) }
        var cropH     by remember { mutableStateOf(imgAreaH) }
        var is1to1    by remember { mutableStateOf(false) }
        var activeHdl by remember { mutableStateOf(CropHandle.NONE) }

        val handleR    = with(density) { 34.dp.toPx() }
        val minCropPx  = with(density) { 80.dp.toPx() }
        val hLen       = with(density) { 22.dp.toPx() }
        val hLineW     = with(density) { 3.dp.toPx() }
        val edgeLen    = hLen * 1.1f   // 모서리 핸들 선 길이 (꼭짓점 L자와 비슷한 크기)
        val edgeLineW  = hLineW * 1.2f // 꼭짓점보다 살짝 두껍게
        // 가로/세로 터치 인식 범위
        val edgeTouchX = with(density) { 48.dp.toPx() }  // N/S: 가로 넓게
        val edgeTouchY = with(density) { 28.dp.toPx() }  // N/S: 세로 보통
        val edgeTouchXv = with(density) { 28.dp.toPx() } // E/W: 가로 보통
        val edgeTouchYv = with(density) { 48.dp.toPx() } // E/W: 세로 넓게

        // 이미지 실제 표시 경계 비동기 계산
        LaunchedEffect(imageUri, buttonRotation) {
            val bounds = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                val opts = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
                context.contentResolver.openInputStream(imageUri)?.use {
                    android.graphics.BitmapFactory.decodeStream(it, null, opts)
                }
                var rawW = opts.outWidth.toFloat()
                var rawH = opts.outHeight.toFloat()
                if (rawW <= 0f || rawH <= 0f) return@withContext null
                val rot90 = ((buttonRotation / 90f).toInt() % 4 + 4) % 4
                if (rot90 % 2 == 1) { rawW = rawH.also { rawH = rawW } }
                val s  = minOf(containerW / rawW, imgAreaH / rawH)
                val dW = rawW * s;  val dH = rawH * s
                val dL = (containerW - dW) / 2f
                val dT = imgAreaTop + (imgAreaH - dH) / 2f
                floatArrayOf(dL, dT, dL + dW, dT + dH)
            } ?: return@LaunchedEffect
            dispLeft   = bounds[0]; dispTop    = bounds[1]
            dispRight  = bounds[2]; dispBottom = bounds[3]
            // 이미지 영역의 90% 크기로 초기 자르기 설정
            val iW = (dispRight - dispLeft) * 0.90f
            val iH = (dispBottom - dispTop) * 0.90f
            cropLeft = dispLeft + (dispRight - dispLeft - iW) / 2f
            cropTop  = dispTop  + (dispBottom - dispTop  - iH) / 2f
            cropW    = iW;  cropH = iH
        }

        // ── 어두운 마스크 + 자르기 테두리 + 격자 + 핸들 ──
        Canvas(modifier = Modifier.fillMaxSize()) {
            val cr = cropLeft + cropW
            val cb = cropTop  + cropH
            val ov = Color.Black.copy(alpha = 0.6f)
            fun cs(w: Float, h: Float) = androidx.compose.ui.geometry.Size(w, h)

            // 외부 4면 어둡게
            drawRect(ov, Offset.Zero,        cs(containerW, cropTop))
            drawRect(ov, Offset(0f, cb),      cs(containerW, containerH - cb))
            drawRect(ov, Offset(0f, cropTop), cs(cropLeft, cropH))
            drawRect(ov, Offset(cr, cropTop), cs(containerW - cr, cropH))

            // 흰색 테두리
            drawRect(
                color   = Color.White,
                topLeft = Offset(cropLeft, cropTop),
                size    = cs(cropW, cropH),
                style   = androidx.compose.ui.graphics.drawscope.Stroke(width = hLineW * 0.55f)
            )

            // 삼등분 격자선
            for (i in 1..2) {
                val gx = cropLeft + cropW * i / 3f
                val gy = cropTop  + cropH * i / 3f
                drawLine(Color.White.copy(alpha = 0.4f), Offset(gx, cropTop), Offset(gx, cb), strokeWidth = hLineW * 0.3f)
                drawLine(Color.White.copy(alpha = 0.4f), Offset(cropLeft, gy), Offset(cr, gy), strokeWidth = hLineW * 0.3f)
            }

            // L자 꼭짓점 핸들
            fun corner(cx: Float, cy: Float, dx: Float, dy: Float) {
                drawLine(Color.White, Offset(cx, cy), Offset(cx + dx, cy), strokeWidth = hLineW)
                drawLine(Color.White, Offset(cx, cy), Offset(cx, cy + dy), strokeWidth = hLineW)
            }
            corner(cropLeft, cropTop,  hLen,  hLen)
            corner(cr,       cropTop, -hLen,  hLen)
            corner(cropLeft, cb,       hLen, -hLen)
            corner(cr,       cb,      -hLen, -hLen)

            // 모서리 중간 핸들 (굵은 가로/세로선)
            val mx = cropLeft + cropW / 2f
            val my = cropTop  + cropH / 2f
            // N (상단 중앙 – 가로선)
            drawLine(Color.White, Offset(mx - edgeLen / 2f, cropTop), Offset(mx + edgeLen / 2f, cropTop), strokeWidth = edgeLineW)
            // S (하단 중앙 – 가로선)
            drawLine(Color.White, Offset(mx - edgeLen / 2f, cb),      Offset(mx + edgeLen / 2f, cb),      strokeWidth = edgeLineW)
            // E (우측 중앙 – 세로선)
            drawLine(Color.White, Offset(cr, my - edgeLen / 2f),      Offset(cr, my + edgeLen / 2f),      strokeWidth = edgeLineW)
            // W (좌측 중앙 – 세로선)
            drawLine(Color.White, Offset(cropLeft, my - edgeLen / 2f), Offset(cropLeft, my + edgeLen / 2f), strokeWidth = edgeLineW)
        }

        // ── 제스처 핸들러 ──
        Box(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(is1to1, dispLeft, dispTop, dispRight, dispBottom) {
                    // 이미지 컨테이너 중심 (graphicsLayer 축)
                    val cx = containerW / 2f
                    val cy = imgAreaTop + imgAreaH / 2f
                    val imgDispW = dispRight - dispLeft
                    val imgDispH = dispBottom - dispTop

                    // 현재 줌/패닝을 적용한 유효 이미지 표시 경계 계산
                    fun effBounds(): FloatArray {
                        val z  = latestZoom.value
                        val px = latestPanX.value
                        val py = latestPanY.value
                        return floatArrayOf(
                            cx + (dispLeft   - cx) * z + px,
                            cy + (dispTop    - cy) * z + py,
                            cx + (dispRight  - cx) * z + px,
                            cy + (dispBottom - cy) * z + py
                        )
                    }

                    awaitPointerEventScope {
                        while (true) {
                            val down = awaitFirstDown(requireUnconsumed = false)
                            val pos  = down.position
                            val (eL, eT, eR, eB) = effBounds().let {
                                arrayOf(it[0], it[1], it[2], it[3])
                            }
                            val cr2  = cropLeft + cropW
                            val cb2  = cropTop  + cropH
                            fun near(x: Float, y: Float) =
                                kotlin.math.abs(pos.x - x) < handleR && kotlin.math.abs(pos.y - y) < handleR
                            fun nearH(x: Float, y: Float) =
                                kotlin.math.abs(pos.x - x) < edgeTouchX && kotlin.math.abs(pos.y - y) < edgeTouchY
                            fun nearV(x: Float, y: Float) =
                                kotlin.math.abs(pos.x - x) < edgeTouchXv && kotlin.math.abs(pos.y - y) < edgeTouchYv

                            activeHdl = when {
                                near(cropLeft, cropTop)                       -> CropHandle.NW
                                near(cr2,      cropTop)                       -> CropHandle.NE
                                near(cropLeft, cb2)                           -> CropHandle.SW
                                near(cr2,      cb2)                           -> CropHandle.SE
                                nearH(cropLeft + cropW / 2f, cropTop)         -> CropHandle.N
                                nearH(cropLeft + cropW / 2f, cb2)             -> CropHandle.S
                                nearV(cr2,      cropTop + cropH / 2f)         -> CropHandle.E
                                nearV(cropLeft, cropTop + cropH / 2f)         -> CropHandle.W
                                pos.x in cropLeft..cr2 && pos.y in cropTop..cb2 -> CropHandle.MOVE
                                else -> CropHandle.NONE
                            }

                            while (true) {
                                val event   = awaitPointerEvent()
                                val pressed = event.changes.filter { it.pressed }
                                if (pressed.isEmpty()) break

                                if (pressed.size >= 2) {
                                    // ── 두 손가락 핀치: 확대/축소 + 이동 ──
                                    val p1 = pressed[0]; val p2 = pressed[1]
                                    val curDist  = (p2.position - p1.position).getDistance()
                                    val prevDist = (p2.previousPosition - p1.previousPosition).getDistance()
                                    if (prevDist > 1f) {
                                        val z  = latestZoom.value
                                        val px = latestPanX.value; val py = latestPanY.value
                                        val scaleFactor = curDist / prevDist
                                        val newZoom = (z * scaleFactor).coerceIn(1f, 5f)
                                        // 두 손가락 중점이 동일한 이미지 점에 고정되도록 패닝 조정
                                        val midX = (p1.previousPosition.x + p2.previousPosition.x) / 2f
                                        val midY = (p1.previousPosition.y + p2.previousPosition.y) / 2f
                                        val newMidX = (p1.position.x + p2.position.x) / 2f
                                        val newMidY = (p1.position.y + p2.position.y) / 2f
                                        var newPanX = newMidX - cx - ((midX - cx - px) / z) * newZoom
                                        var newPanY = newMidY - cy - ((midY - cy - py) / z) * newZoom
                                        // 패닝 범위 제한 (이미지가 화면에서 너무 벗어나지 않도록)
                                        val maxPX = imgDispW * (newZoom - 1f) / 2f + kotlin.math.abs(dispLeft - cx)
                                        val maxPY = imgDispH * (newZoom - 1f) / 2f + kotlin.math.abs(dispTop  - cy)
                                        newPanX = newPanX.coerceIn(-maxPX, maxPX)
                                        newPanY = newPanY.coerceIn(-maxPY, maxPY)
                                        onZoomChange(newZoom, newPanX, newPanY)
                                    }
                                    pressed.forEach { it.consume() }
                                } else if (pressed.size == 1) {
                                    val ch = pressed[0]
                                    val d  = ch.position - ch.previousPosition
                                    val b  = effBounds()
                                    val efl = b[0]; val eft = b[1]; val efr = b[2]; val efb = b[3]
                                    if (is1to1) {
                                        // ── 1:1 고정 ──
                                        when (activeHdl) {
                                            CropHandle.MOVE -> {
                                                cropLeft = (cropLeft + d.x).coerceIn(efl, efr - cropW)
                                                cropTop  = (cropTop  + d.y).coerceIn(eft,  efb - cropH)
                                            }
                                            CropHandle.NW, CropHandle.N, CropHandle.W -> {
                                                val pivR = cropLeft + cropW; val pivB = cropTop + cropH
                                                val ds = -(d.x + d.y) / 2f
                                                val ns = (cropW + ds).coerceIn(minCropPx, minOf(pivR - efl, pivB - eft))
                                                cropLeft = (pivR - ns).coerceAtLeast(efl)
                                                cropTop  = (pivB - ns).coerceAtLeast(eft)
                                                cropW = ns; cropH = ns
                                            }
                                            CropHandle.NE -> {
                                                val pivB = cropTop + cropH
                                                val ds = (d.x - d.y) / 2f
                                                val ns = (cropW + ds).coerceIn(minCropPx, minOf(efr - cropLeft, pivB - eft))
                                                cropTop = (pivB - ns).coerceAtLeast(eft)
                                                cropW = ns; cropH = ns
                                            }
                                            CropHandle.SW -> {
                                                val pivR = cropLeft + cropW
                                                val ds = (-d.x + d.y) / 2f
                                                val ns = (cropW + ds).coerceIn(minCropPx, minOf(pivR - efl, efb - cropTop))
                                                cropLeft = (pivR - ns).coerceAtLeast(efl)
                                                cropW = ns; cropH = ns
                                            }
                                            CropHandle.SE, CropHandle.E, CropHandle.S -> {
                                                val ds = (d.x + d.y) / 2f
                                                val ns = (cropW + ds).coerceIn(minCropPx, minOf(efr - cropLeft, efb - cropTop))
                                                cropW = ns; cropH = ns
                                            }
                                            CropHandle.NONE -> {}
                                        }
                                    } else {
                                        // ── 자유 크롭 ──
                                        when (activeHdl) {
                                            CropHandle.MOVE -> {
                                                cropLeft = (cropLeft + d.x).coerceIn(efl, efr - cropW)
                                                cropTop  = (cropTop  + d.y).coerceIn(eft,  efb - cropH)
                                            }
                                            CropHandle.NW -> {
                                                val pivR = cropLeft + cropW; val pivB = cropTop + cropH
                                                val nw = (cropW - d.x).coerceIn(minCropPx, pivR - efl)
                                                val nh = (cropH - d.y).coerceIn(minCropPx, pivB - eft)
                                                cropLeft = pivR - nw; cropTop = pivB - nh
                                                cropW = nw; cropH = nh
                                            }
                                            CropHandle.NE -> {
                                                val pivB = cropTop + cropH
                                                cropW = (cropW + d.x).coerceIn(minCropPx, efr - cropLeft)
                                                val nh = (cropH - d.y).coerceIn(minCropPx, pivB - eft)
                                                cropTop = pivB - nh; cropH = nh
                                            }
                                            CropHandle.SW -> {
                                                val pivR = cropLeft + cropW
                                                val nw = (cropW - d.x).coerceIn(minCropPx, pivR - efl)
                                                cropLeft = pivR - nw; cropW = nw
                                                cropH = (cropH + d.y).coerceIn(minCropPx, efb - cropTop)
                                            }
                                            CropHandle.SE -> {
                                                cropW = (cropW + d.x).coerceIn(minCropPx, efr - cropLeft)
                                                cropH = (cropH + d.y).coerceIn(minCropPx, efb - cropTop)
                                            }
                                            CropHandle.N -> {
                                                val pivB = cropTop + cropH
                                                val nh = (cropH - d.y).coerceIn(minCropPx, pivB - eft)
                                                cropTop = pivB - nh; cropH = nh
                                            }
                                            CropHandle.E -> {
                                                cropW = (cropW + d.x).coerceIn(minCropPx, efr - cropLeft)
                                            }
                                            CropHandle.S -> {
                                                cropH = (cropH + d.y).coerceIn(minCropPx, efb - cropTop)
                                            }
                                            CropHandle.W -> {
                                                val pivR = cropLeft + cropW
                                                val nw = (cropW - d.x).coerceIn(minCropPx, pivR - efl)
                                                cropLeft = pivR - nw; cropW = nw
                                            }
                                            CropHandle.NONE -> {}
                                        }
                                    }
                                    ch.consume()
                                }
                            }
                        }
                    }
                }
        )

        // ── 상단 바 오버레이 (기존 편집 상단 바 위에 덮음) ──
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(topBarDp)
                .background(Color.Black)
                .padding(horizontal = 4.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onCancel) {
                Icon(Icons.Default.Close, contentDescription = "닫기", tint = Color.White)
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onRestoreOriginal) {
                    Text("원본 복원", color = Color.White, fontSize = 14.sp)
                }
                TextButton(onClick = {
                    onConfirm(
                        PendingCropData(
                            left            = cropLeft,
                            top             = cropTop,
                            width           = cropW,
                            height          = cropH,
                            imgAreaTopPx    = imgAreaTop,
                            imgAreaHeightPx = imgAreaH,
                            containerW      = containerW,
                            containerH      = containerH,
                            cropZoom        = latestZoom.value,
                            cropPanX        = latestPanX.value,
                            cropPanY        = latestPanY.value
                        )
                    )
                }) {
                    Text("저장", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                }
            }
        }

        // ── 하단 바: 1:1 토글만 ──
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .height(bottomBarDp)
                .background(Color.Black),
            contentAlignment = Alignment.Center
        ) {
            val btnBorder = if (is1to1) Color(0xFF9CD83B) else Color(0xFF666666)
            val btnText   = if (is1to1) Color(0xFF9CD83B) else Color(0xFF888888)
            val btnBg     = if (is1to1) Color(0xFF9CD83B).copy(alpha = 0.15f) else Color.Transparent
            Box(
                modifier = Modifier
                    .border(1.5.dp, btnBorder, RoundedCornerShape(6.dp))
                    .background(btnBg, RoundedCornerShape(6.dp))
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null
                    ) {
                        is1to1 = !is1to1
                        if (is1to1) {
                            val sq = minOf(cropW, cropH)
                            val cx = cropLeft + cropW / 2f
                            val cy = cropTop  + cropH / 2f
                            cropW    = sq; cropH = sq
                            cropLeft = (cx - sq / 2f).coerceIn(dispLeft, dispRight  - sq)
                            cropTop  = (cy - sq / 2f).coerceIn(dispTop,  dispBottom - sq)
                        }
                    }
                    .padding(horizontal = 18.dp, vertical = 6.dp),
                contentAlignment = Alignment.Center
            ) {
                Text("1 : 1", color = btnText, fontSize = 13.sp, fontWeight = FontWeight.Medium)
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
@Composable
private fun EditDialBar(
    degrees: Float,
    onDegreesChanged: (Float) -> Unit
) {
    val density = LocalDensity.current

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxWidth()
            .height(52.dp)   // Galaxy 수준으로 더 슬림하게
            .background(Color.Black)
    ) {
        val viewWidthPx = with(density) { maxWidth.toPx() }
        val tickWidthPx = viewWidthPx / 50f   // 화면에 50칸
        val visibleHalf = 27

        Box(
            modifier = Modifier
                .fillMaxSize()
                .draggable(
                    orientation = Orientation.Horizontal,
                    state = rememberDraggableState { delta ->
                        onDegreesChanged(degrees + delta / tickWidthPx)
                    }
                )
        ) {
            // 현재 각도 텍스트 (중앙 상단)
            Text(
                text     = "${"%.1f".format(degrees)}°",
                color    = Color(0xFF9CD83B),
                fontSize = 10.sp,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 2.dp)
            )

            // 눈금 Canvas
            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(36.dp)
                    .align(Alignment.BottomCenter)
            ) {
                val centerX  = size.width / 2f
                val midY     = size.height / 2f
                val tickH    = size.height * 0.40f  // 균일 작은 눈금
                val originH  = size.height * 0.80f  // 원점만 약간 길게
                val halfW    = size.width / 2f

                val from = (degrees - visibleHalf).toInt()
                val to   = (degrees + visibleHalf).toInt()

                for (i in from..to) {
                    val x = centerX + (i - degrees) * tickWidthPx
                    if (x < 0f || x > size.width) continue

                    // 중심에서 멀수록 점점 투명하게 (페이드 아웃)
                    val dist       = kotlin.math.abs(x - centerX) / halfW   // 0=중심, 1=끝
                    val fadeAlpha  = ((1f - dist / 0.82f) * 1.1f).coerceIn(0f, 1f)

                    val isOrigin   = i == 0
                    val baseColor  = if (isOrigin) Color(0xFF9CD83B) else Color(0xFFAAAAAA)

                    drawLine(
                        color       = baseColor.copy(alpha = fadeAlpha),
                        start       = Offset(x, midY - (if (isOrigin) originH else tickH) / 2f),
                        end         = Offset(x, midY + (if (isOrigin) originH else tickH) / 2f),
                        strokeWidth = if (isOrigin) 2.dp.toPx() else 1.dp.toPx()
                    )
                }
            }

            // 중앙 기준 포인터 (고정 녹색 선)
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 4.dp)
                    .width(2.dp)
                    .height(18.dp)
                    .background(Color(0xFF9CD83B))
            )
        }
    }
}

// ── 편집 기능 버튼 아이템 ──────────────────────────────────────────────────────
@Composable
private fun EditActionButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null
            ) { onClick() }
            .padding(horizontal = 16.dp, vertical = 10.dp),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector        = icon,
            contentDescription = label,
            tint               = Color.White,
            modifier           = Modifier.size(30.dp)
        )
    }
}

// ── 편집된 이미지 저장 (회전·반전 적용 후 파일 쓰기) ─────────────────────────
private suspend fun saveEditedImage(
    context: Context,
    originalUri: Uri,
    rotationDegrees: Float,
    isFlipped: Boolean,
    overwrite: Boolean,
    cropData: PendingCropData? = null
): Uri? = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
    try {
        // 비트맵 로드
        val src = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            val source = android.graphics.ImageDecoder.createSource(context.contentResolver, originalUri)
            android.graphics.ImageDecoder.decodeBitmap(source) { dec, _, _ ->
                dec.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            android.provider.MediaStore.Images.Media.getBitmap(context.contentResolver, originalUri)
        }.copy(android.graphics.Bitmap.Config.ARGB_8888, true)

        // 회전/반전 행렬 적용
        val matrix = android.graphics.Matrix()
        if (rotationDegrees != 0f) matrix.postRotate(rotationDegrees)
        if (isFlipped)             matrix.postScale(-1f, 1f, src.width / 2f, src.height / 2f)

        var result = if (!matrix.isIdentity) {
            android.graphics.Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
        } else src

        // 자르기 적용 (회전/반전 후 화면상 좌표 → 비트맵 좌표 변환, 줌/패닝 포함)
        if (cropData != null) {
            val rotW = result.width.toFloat()
            val rotH = result.height.toFloat()
            // ContentScale.Fit 기준 디스플레이 스케일 및 오프셋
            val s    = minOf(cropData.containerW / rotW, cropData.imgAreaHeightPx / rotH)
            val baseDispLeft = (cropData.containerW - rotW * s) / 2f
            val baseDispTop  = cropData.imgAreaTopPx + (cropData.imgAreaHeightPx - rotH * s) / 2f
            // graphicsLayer 중심 (축 기준점)
            val cx = cropData.containerW / 2f
            val cy = cropData.imgAreaTopPx + cropData.imgAreaHeightPx / 2f
            // 줌/패닝 적용 후 실제 화면상 이미지 좌상단 위치
            val z  = cropData.cropZoom
            val px = cropData.cropPanX; val py = cropData.cropPanY
            val effDispLeft = cx + (baseDispLeft - cx) * z + px
            val effDispTop  = cy + (baseDispTop  - cy) * z + py
            val effScale    = s * z   // 유효 px-per-image-pixel

            val imgCropLeft = ((cropData.left - effDispLeft) / effScale).coerceIn(0f, rotW - 1f).toInt()
            val imgCropTop  = ((cropData.top  - effDispTop)  / effScale).coerceIn(0f, rotH - 1f).toInt()
            val imgCropW = (cropData.width  / effScale).coerceIn(1f, rotW - imgCropLeft).toInt().coerceAtLeast(1)
            val imgCropH = (cropData.height / effScale).coerceIn(1f, rotH - imgCropTop ).toInt().coerceAtLeast(1)

            result = android.graphics.Bitmap.createBitmap(result, imgCropLeft, imgCropTop, imgCropW, imgCropH)
        }

        // 저장 경로 결정
        val originalPath = originalUri.path ?: return@withContext null
        val originalFile = java.io.File(originalPath)
        val outFile = if (overwrite) {
            originalFile
        } else {
            java.io.File(
                originalFile.parent ?: context.getExternalFilesDir(null)!!.absolutePath,
                "edited_${System.currentTimeMillis()}.jpg"
            )
        }

        java.io.FileOutputStream(outFile).use { fos ->
            result.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, fos)
        }
        android.net.Uri.fromFile(outFile)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

private fun takePhoto(
    context: Context,
    imageCapture: ImageCapture,
    onPhotoTaken: (Uri) -> Unit
) {
    val photoFile = File(
        context.getExternalFilesDir(null),
        SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US)
            .format(System.currentTimeMillis()) + ".jpg"
    )

    val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

    imageCapture.takePicture(
        outputOptions,
        ContextCompat.getMainExecutor(context),
        object : ImageCapture.OnImageSavedCallback {
            override fun onError(exception: ImageCaptureException) {
                exception.printStackTrace()
            }

            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                onPhotoTaken(Uri.fromFile(photoFile))
            }
        }
    )
}

private fun startVideoRecording(
    context: Context,
    videoCapture: VideoCapture<androidx.camera.video.Recorder>,
    onRecordingStarted: (Recording) -> Unit,
    onVideoSaved: (Uri) -> Unit
) {
    try {
        val videoFile = File(
            context.getExternalFilesDir(null),
            SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US)
                .format(System.currentTimeMillis()) + ".mp4"
        )

        val fileOutputOptions = androidx.camera.video.FileOutputOptions.Builder(videoFile).build()

        val recording = videoCapture.output
            .prepareRecording(context, fileOutputOptions)
            .apply {
                // 오디오 권한이 있으면 오디오 활성화, 없으면 오디오 없이 촬영
                if (ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    withAudioEnabled()
                }
            }
            .start(ContextCompat.getMainExecutor(context)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> {
                        // 촬영 시작됨
                    }
                    is VideoRecordEvent.Finalize -> {
                        if (!event.hasError()) {
                            // 동영상 저장 완료
                            val videoUri = Uri.fromFile(videoFile)
                            onVideoSaved(videoUri)
                        } else {
                            // 오류 발생
                            event.cause?.printStackTrace()
                        }
                    }
                }
            }
        
        // recording 객체를 콜백으로 전달
        onRecordingStarted(recording)
    } catch (e: Exception) {
        e.printStackTrace()
        // 오류 발생 시 빈 Recording 객체 전달하지 않음
    }
}

// (삭제) TFLite 기반 광택/반사 제거 기능: 요청에 따라 Inpainting 기반으로 전면 교체됨

// [추가] 이미지 유사도 계산 (픽셀 기반 구조 비교, 0.0 ~ 1.0)
private fun calculateImageSimilarity(bitmap1: android.graphics.Bitmap, bitmap2: android.graphics.Bitmap): Float {
    val width = 64
    val height = 64
    // 성능을 위해 리사이징하여 비교
    val s1 = android.graphics.Bitmap.createScaledBitmap(bitmap1, width, height, true)
    val s2 = android.graphics.Bitmap.createScaledBitmap(bitmap2, width, height, true)
    
    val pixels1 = IntArray(width * height)
    val pixels2 = IntArray(width * height)
    
    s1.getPixels(pixels1, 0, width, 0, 0, width, height)
    s2.getPixels(pixels2, 0, width, 0, 0, width, height)
    
    var similaritySum = 0.0
    val maxDist = Math.sqrt(255.0 * 255.0 * 3.0) // 가능한 최대 색상 거리
    
    for (i in pixels1.indices) {
        val c1 = pixels1[i]
        val c2 = pixels2[i]
        
        val r1 = (c1 shr 16) and 0xFF
        val g1 = (c1 shr 8) and 0xFF
        val b1 = c1 and 0xFF
        
        val r2 = (c2 shr 16) and 0xFF
        val g2 = (c2 shr 8) and 0xFF
        val b2 = c2 and 0xFF
        
        // 유클리드 거리 기반 색상 차이
        val dist = Math.sqrt(
            ((r1 - r2) * (r1 - r2) + (g1 - g2) * (g1 - g2) + (b1 - b2) * (b1 - b2)).toDouble()
        )
        // 유사도 누적 (1.0 = 일치, 0.0 = 불일치)
        similaritySum += (1.0 - (dist / maxDist))
    }
    
    if (s1 != bitmap1) s1.recycle()
    if (s2 != bitmap2) s2.recycle()
    
    return (similaritySum / (width * height)).toFloat()
}

// [수정] 모델링 적합성 판단 (이동식 공간 촬영용)
// - 기존: "가상 점 5개가 모두 같은 공간"과 유사한 개념을 십자선 전체 표준편차로 간접 판단
// - 변경: 3x3 배열(9개)의 "가상 점(샘플 포인트)"을 중앙 주변에 배치하고,
//        9개 중 7개 이상이 동일한 공간(=주변 RGB/밝기 특성이 동일한 영역)으로 판정되면 경고(false)
// true: 적합, false: 부적합(경고 필요)
private fun checkModelingSuitability(bitmap: android.graphics.Bitmap): Boolean {
    val width = bitmap.width
    val height = bitmap.height
    if (width <= 2 || height <= 2) return true

    val cx = width / 2
    val cy = height / 2

    // 3x3 샘플 포인트 배치 (중앙 기준)
    // - 화면 해상도에 따라 자동 스케일
    val spacing = (minOf(width, height) * 0.18f).toInt().coerceIn(80, 260)
    val offsets = intArrayOf(-spacing, 0, spacing)

    // 각 포인트에서 작은 패치(주변 픽셀) 평균 밝기를 구해 "공간 ID"로 사용
    // - quantStep이 작을수록 민감 (동일 판정이 어려움), 클수록 둔감
    val patchRadius = 10 // 21x21
    val quantStep = 16   // 0~255 -> 16단계(0~15)

    val spaceIdCounts = HashMap<Int, Int>(16)
    var totalPoints = 0

    fun lumaOf(pixel: Int): Int {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        return (0.299 * r + 0.587 * g + 0.114 * b).toInt()
    }

    for (dy in offsets) {
        for (dx in offsets) {
            val px = (cx + dx).coerceIn(0, width - 1)
            val py = (cy + dy).coerceIn(0, height - 1)

            var sum = 0L
            var count = 0
            val y0 = (py - patchRadius).coerceIn(0, height - 1)
            val y1 = (py + patchRadius).coerceIn(0, height - 1)
            val x0 = (px - patchRadius).coerceIn(0, width - 1)
            val x1 = (px + patchRadius).coerceIn(0, width - 1)

            for (y in y0..y1) {
                for (x in x0..x1) {
                    sum += lumaOf(bitmap.getPixel(x, y)).toLong()
                    count++
                }
            }

            if (count <= 0) continue
            val meanLuma = (sum / count).toInt().coerceIn(0, 255)
            val spaceId = (meanLuma / quantStep).coerceIn(0, 255 / quantStep)

            spaceIdCounts[spaceId] = (spaceIdCounts[spaceId] ?: 0) + 1
            totalPoints++
        }
    }

    if (totalPoints <= 0) return true

    // 9개 중 7개 이상이 동일 공간이면 "깊이/텍스처 정보 부족"으로 간주
    val maxSame = spaceIdCounts.values.maxOrNull() ?: 0
    if (maxSame >= 7) return false

    // 보조 안전장치: 중앙 십자선 영역의 텍스처(표준편차)도 너무 낮으면 부적합 처리
    // (샘플링만으로 놓치는 케이스 방지)
    val halfLen = 350
    val halfThick = 10
    val lumValues = ArrayList<Int>(4000)
    for (x in (cx - halfLen) until (cx + halfLen)) {
        for (y in (cy - halfThick) until (cy + halfThick)) {
            if (x in 0 until width && y in 0 until height) lumValues.add(lumaOf(bitmap.getPixel(x, y)))
        }
    }
    for (y in (cy - halfLen) until (cy + halfLen)) {
        for (x in (cx - halfThick) until (cx + halfThick)) {
            if (x in 0 until width && y in 0 until height) lumValues.add(lumaOf(bitmap.getPixel(x, y)))
        }
    }
    if (lumValues.isEmpty()) return true
    val mean = lumValues.sum().toDouble() / lumValues.size
    var varianceSum = 0.0
    for (lum in lumValues) varianceSum += (lum - mean) * (lum - mean)
    val stdDev = Math.sqrt(varianceSum / lumValues.size)

    return stdDev > 30.0
}

// (삭제) IID/MSA 기반 광택 제거 기능: 요청에 따라 Inpainting 기반으로 전면 교체됨

// [추가] 이미지 파일 로드 시 EXIF 회전 정보 반영
private fun loadBitmapWithRotation(path: String): android.graphics.Bitmap? {
    return try {
        val bitmap = BitmapFactory.decodeFile(path) ?: return null
        val exif = ExifInterface(path)
        val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        
        val rotation = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        
        if (rotation != 0f) {
            val matrix = Matrix()
            matrix.postRotate(rotation)
            android.graphics.Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

// (삭제) 픽셀 억제/금속/반광/통합 파이프라인: 요청에 따라 Inpainting 기반으로 전면 교체됨


private fun saveBitmapToFile(bitmap: android.graphics.Bitmap, file: File) {
    try {
        java.io.FileOutputStream(file).use { out ->
            bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 100, out)
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

private fun captureDatasetImage(
    context: Context,
    sectorIndex: Int,
    pitchAngle: Int,
    dir: File,
    capture: ImageCapture,
    customFileName: String? = null,
    validationCallback: ((android.graphics.Bitmap) -> Boolean)? = null
) {
    try {
        if (!dir.exists()) {
            dir.mkdirs()
        }
        val file = if (customFileName != null) {
            File(dir, customFileName)
        } else {
            File(dir, "${pitchAngle}_${sectorIndex + 1}.jpg")
        }
        
        // 메모리로 캡처 후 후처리 적용
        capture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    var bitmap: android.graphics.Bitmap? = null
                    try {
                        val buffer = image.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)
                        bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        
                        // 회전 보정
                        val rotation = image.imageInfo.rotationDegrees
                        if (rotation != 0 && bitmap != null) {
                            val matrix = Matrix()
                            matrix.postRotate(rotation.toFloat())
                            bitmap = android.graphics.Bitmap.createBitmap(
                                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                            )
                        }

                        if (bitmap != null) {
                            // [추가] 1:1 비율로 중앙 크롭 및 1024x1024 리사이징
                            val originalWidth = bitmap.width
                            val originalHeight = bitmap.height
                            val size = if (originalWidth < originalHeight) originalWidth else originalHeight
                            val xOffset = (originalWidth - size) / 2
                            val yOffset = (originalHeight - size) / 2
                            
                            // 중앙 정사각형 크롭
                            var croppedBitmap = android.graphics.Bitmap.createBitmap(
                                bitmap, xOffset, yOffset, size, size
                            )
                            
                            // 1024x1024로 리사이징
                            if (size != 1024) {
                                croppedBitmap = android.graphics.Bitmap.createScaledBitmap(
                                    croppedBitmap, 1024, 1024, true
                                )
                            }
                            bitmap = croppedBitmap

                            // [추가] 저장 전 검증 (유사도 체크 등)
                            if (validationCallback != null) {
                                if (!validationCallback(bitmap)) {
                                    // 검증 실패 시 저장하지 않음 (유사도 조건 불만족 등)
                                    return
                                }
                            }

                            // [변경] 촬영 시에는 원본을 그대로 저장 (광택/반사 보정은 폴더 선택 후 옵션에서 수행)
                            saveBitmapToFile(bitmap, file)
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                    } finally {
                        image.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    exception.printStackTrace()
                }
            }
        )
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

private fun formatTime(millis: Long): String {
    val seconds = TimeUnit.MILLISECONDS.toSeconds(millis) % 60
    val minutes = TimeUnit.MILLISECONDS.toMinutes(millis) % 60
    return String.format(Locale.US, "%02d:%02d", minutes, seconds)
}

private fun deleteMediaByUri(context: Context, uri: Uri) {
    try {
        if (uri.scheme == "content") {
            context.contentResolver.delete(uri, null, null)
        } else {
            val file = File(uri.path ?: "")
            if (file.exists()) {
                file.delete()
            }
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

private fun deleteAllMedia(context: Context) {
    try {
        val mediaDir = context.getExternalFilesDir(null)
        if (mediaDir != null && mediaDir.exists()) {
            mediaDir.listFiles()?.forEach { file ->
                if (file.isDirectory) {
                    file.deleteRecursively()
                } else {
                    file.delete()
                }
            }
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

private fun deleteDatasetFolder(folder: DatasetFolder) {
    try {
        if (folder.dir.exists()) {
            folder.dir.deleteRecursively()
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

private fun isVideoUri(context: Context, uri: Uri): Boolean {
    return try {
        if (uri.scheme == "content") {
            val type = context.contentResolver.getType(uri)
            type?.startsWith("video/") == true
        } else {
            uri.toString().endsWith(".mp4", ignoreCase = true)
        }
    } catch (e: Exception) {
        false
    }
}

private fun loadDatasetFolders(
    context: Context,
    onLoaded: (List<DatasetFolder>) -> Unit
) {
    onLoaded(loadDatasetFoldersSync(context))
}

// [추가] 데이터셋 폴더 로드(동기) + 빈 폴더 자동 정리
private fun loadDatasetFoldersSync(context: Context): List<DatasetFolder> {
    val root = File(context.getExternalFilesDir(null), "datasets")
    if (!root.exists()) {
        return emptyList()
    }

    // [추가] 0장(빈) 데이터셋 폴더 자동 정리
    // - 너무 최근에 생성된 폴더(촬영 직후 등)는 오탐 방지를 위해 잠시 유예
    val now = System.currentTimeMillis()
    val minAgeMs = 60_000L // 60초보다 오래된 "빈" 폴더만 삭제

    val imageExts = setOf("jpg", "jpeg", "png", "webp")
    return root.listFiles { file -> file.isDirectory }
        ?.mapNotNull { dir ->
            val images = dir.listFiles { f ->
                f.isFile && imageExts.contains(f.extension.lowercase())
            }?.sortedBy { it.nameWithoutExtension.toIntOrNull() ?: Int.MAX_VALUE } ?: emptyList()

            if (images.isEmpty()) {
                // 이미지가 0개인 폴더는 주기적으로 자동 삭제
                if (now - dir.lastModified() >= minAgeMs) {
                    try {
                        dir.deleteRecursively()
                    } catch (_: Exception) {
                    }
                    return@mapNotNull null
                }
            }

            val cover = images.firstOrNull()?.let { Uri.fromFile(it) }
            DatasetFolder(
                name = dir.name,
                dir = dir,
                coverUri = cover,
                count = images.size
            )
        }
        ?.sortedByDescending { it.name }
        ?: emptyList()
}

/** Gemini 전송용: 모든 데이터셋 폴더의 이미지 URI 목록 로드 */
private fun loadAllDatasetImages(context: Context, onLoaded: (List<Uri>) -> Unit) {
    val folders = loadDatasetFoldersSync(context)
    val imageExts = setOf("jpg", "jpeg", "png", "webp")
    val uris = folders.flatMap { folder ->
        folder.dir.listFiles { f ->
            f.isFile && imageExts.contains(f.extension.lowercase())
        }?.map { Uri.fromFile(it) } ?: emptyList()
    }
    onLoaded(uris)
}

private fun loadDatasetImages(
    dir: File,
    onLoaded: (List<Uri>) -> Unit
) {
    if (!dir.exists()) {
        onLoaded(emptyList())
        return
    }
    val images = dir.listFiles { f ->
        f.isFile && f.name.endsWith(".jpg", ignoreCase = true)
    }?.sortedBy { it.nameWithoutExtension.toIntOrNull() ?: Int.MAX_VALUE } ?: emptyList()

    onLoaded(images.map { Uri.fromFile(it) })
}

private fun loadPlyModels(
    context: Context,
    onLoaded: (List<PlyModel>) -> Unit
) {
    val modelsDir = File(context.getExternalFilesDir(null), "models")
    if (!modelsDir.exists()) {
        onLoaded(emptyList())
        return
    }
    val models = modelsDir.listFiles { f ->
        f.isFile && f.name.endsWith(".ply", ignoreCase = true)
    }?.map { f ->
        PlyModel(
            name = f.nameWithoutExtension,
            file = f,
            lastModified = f.lastModified()
        )
    }?.sortedByDescending { it.lastModified } ?: emptyList()

    onLoaded(models)
}

private fun loadCapturedMedia(
    context: Context,
    onMediaLoaded: (List<Uri>) -> Unit
) {
    val mediaDir = context.getExternalFilesDir(null)
    if (mediaDir == null || !mediaDir.exists()) {
        onMediaLoaded(emptyList())
        return
    }

    val datasetsRoot = File(mediaDir, "datasets").absolutePath
    val entries = mediaDir.walkTopDown()
        .filter { file ->
            file.isFile &&
                (file.name.endsWith(".jpg", ignoreCase = true) ||
                    file.name.endsWith(".jpeg", ignoreCase = true) ||
                    file.name.endsWith(".png", ignoreCase = true) ||
                    file.name.endsWith(".webp", ignoreCase = true) ||
                    file.name.endsWith(".heic", ignoreCase = true) ||
                    file.name.endsWith(".heif", ignoreCase = true) ||
                    file.name.endsWith(".mp4", ignoreCase = true)) &&
                !file.absolutePath.startsWith(datasetsRoot)
        }
        .map { file -> Uri.fromFile(file) to file.lastModified() }
        .toList()

    onMediaLoaded(entries.sortedByDescending { it.second }.map { it.first })
}

/**
 * HTTPS를 지원하는 OkHttpClient 생성
 */
private fun createOkHttpClient(useHttps: Boolean = DEFAULT_USE_HTTPS): OkHttpClient {
    val builder = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
    
    // HTTPS 사용 시 SSL 인증서 검증 우회 (개발/테스트 환경)
    if (useHttps) {
        val trustAllCerts = arrayOf<TrustManager>(object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
            override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
            override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
        })

        val sslContext = SSLContext.getInstance("TLS")
        sslContext.init(null, trustAllCerts, java.security.SecureRandom())
        val sslSocketFactory = sslContext.socketFactory

        builder.sslSocketFactory(sslSocketFactory, trustAllCerts[0] as X509TrustManager)
        builder.hostnameVerifier(HostnameVerifier { _, _ -> true })
    }
    
    return builder.build()
}

private fun resolveDisplayName(context: Context, uri: Uri): String? {
    return try {
        if (uri.scheme == "content") {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor ->
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

private suspend fun createZipFromUris(
    context: Context,
    uris: List<Uri>,
    zipPrefix: String
): File? {
    return withContext(Dispatchers.IO) {
        try {
            val zipFileName =
                "${zipPrefix}_${SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US).format(System.currentTimeMillis())}.zip"
            val zipFile = File(context.getExternalFilesDir(null), zipFileName)

            FileOutputStream(zipFile).use { fos ->
                ZipOutputStream(fos).use { zos ->
                    uris.forEachIndexed { index, uri ->
                        val name = resolveDisplayName(context, uri)
                            ?: "item_${index + 1}"
                        val safeName = name.replace('\\', '_').replace('/', '_')

                        val input = if (uri.scheme == "content") {
                            context.contentResolver.openInputStream(uri)
                        } else {
                            val path = uri.path
                            if (path.isNullOrBlank()) null else FileInputStream(File(path))
                        } ?: return@forEachIndexed

                        input.use { ins ->
                            zos.putNextEntry(ZipEntry(safeName))
                            ins.copyTo(zos)
                            zos.closeEntry()
                        }
                    }
                }
            }

            zipFile
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

private suspend fun createZipFromFolders(
    context: Context,
    folders: List<File>,
    zipPrefix: String
): File? {
    return withContext(Dispatchers.IO) {
        try {
            val zipFileName =
                "${zipPrefix}_${SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US).format(System.currentTimeMillis())}.zip"
            val zipFile = File(context.getExternalFilesDir(null), zipFileName)

            FileOutputStream(zipFile).use { fos ->
                ZipOutputStream(fos).use { zos ->
                    folders.forEach { folder ->
                        if (!folder.exists() || !folder.isDirectory) return@forEach
                        folder.walkTopDown()
                            .filter { it.isFile }
                            .forEach { file ->
                                val rel = try {
                                    file.relativeTo(folder).path.replace('\\', '/')
                                } catch (_: Exception) {
                                    file.name
                                }
                                val entryName = "${folder.name}/$rel"
                                zos.putNextEntry(ZipEntry(entryName))
                                FileInputStream(file).use { fis ->
                                    fis.copyTo(zos)
                                }
                                zos.closeEntry()
                            }
                    }
                }
            }

            zipFile
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

/**
 * 미디어 파일을 서버에 업로드하는 함수
 * @param context 컨텍스트
 * @param file 업로드할 파일
 * @return 업로드 성공 여부
 */
private suspend fun startServerTaskWithZip(
    context: Context,
    zipFile: File,
    prompt: String = ""
): String? {
    return withContext(Dispatchers.IO) {
        try {
            val serverAddress = getServerAddress(context)
            val serverPort = getServerPort(context)
            val useHttps = getUseHttps(context)
            val client = createOkHttpClient(useHttps).newBuilder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .callTimeout(60, TimeUnit.SECONDS)
                .build()

            if (!zipFile.name.endsWith(".zip", ignoreCase = true)) {
                return@withContext null
            }

            val multipartBuilder = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    zipFile.name,
                    zipFile.asRequestBody("application/zip".toMediaType())
                )
            if (prompt.isNotBlank()) {
                multipartBuilder.addFormDataPart("prompt", prompt)
            }
            val requestBody = multipartBuilder.build()

            val protocol = if (useHttps) "https" else "http"
            val url = "$protocol://$serverAddress:$serverPort$UPLOAD_ENDPOINT"
            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val body = response.body?.string()
            val success = response.isSuccessful && body != null
            response.close()

            if (!success) return@withContext null
            val json = JSONObject(body)
            json.optString("task_id").takeIf { it.isNotBlank() }
        } catch (e: IOException) {
            e.printStackTrace()
            null
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

// (삭제) 서버 기반 광택/반사 제거 파이프라인: 요청에 따라 로컬 Inpainting 기반으로 교체됨

private data class ServerTaskStatus(
    val status: String,
    val progressPercent: Int,
    val message: String,
    val downloadUrl: String?
)

private suspend fun fetchServerTaskStatus(context: Context, taskId: String): ServerTaskStatus? {
    return withContext(Dispatchers.IO) {
        try {
            val serverAddress = getServerAddress(context)
            val serverPort = getServerPort(context)
            val useHttps = getUseHttps(context)
            val client = createOkHttpClient(useHttps).newBuilder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .callTimeout(60, TimeUnit.SECONDS)
                .build()

            val protocol = if (useHttps) "https" else "http"
            val url = "$protocol://$serverAddress:$serverPort$STATUS_ENDPOINT/$taskId"
            val request = Request.Builder().url(url).get().build()
            val response = client.newCall(request).execute()
            val body = response.body?.string()
            val ok = response.isSuccessful && body != null
            response.close()
            if (!ok) return@withContext null

            val json = JSONObject(body)
            ServerTaskStatus(
                status = json.optString("status"),
                progressPercent = json.optInt("progress_percent", 0),
                message = json.optString("message", "처리 중..."),
                downloadUrl = json.optString("download_url").takeIf { it.isNotBlank() }
            )
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

/**
 * SAM2 서버(/bg-remove)에 이미지와 텍스트 프롬프트를 전송하여
 * 배경이 제거된 투명 PNG를 받아 앱 라이브러리에 저장합니다.
 *
 * 서버: scripts/sam2_server.py (포트 8001)
 *
 * @param context   앱 컨텍스트
 * @param imageUri  처리할 원본 이미지 URI
 * @param prompt    남길 사물의 영문 텍스트 (예: "cup", "mouse")
 * @param itemIndex 현재 처리 중인 이미지 순번 (0-based)
 * @param itemTotal 처리할 이미지 총 수
 * @param onProgress 진행률 콜백 (percent 0~100, message)
 * @return 앱 라이브러리에 저장된 PNG Uri, 실패 시 null
 */
private suspend fun sam2RemoveBackground(
    context: Context,
    imageUri: Uri,
    prompt: String,
    itemIndex: Int = 0,
    itemTotal: Int = 1,
    onProgress: suspend (percent: Int, message: String) -> Unit = { _, _ -> },
): Uri? = withContext(Dispatchers.IO) {
    val itemLabel = if (itemTotal > 1) " (${itemIndex + 1}/$itemTotal)" else ""
    try {
        onProgress(5, "이미지 로드 중...$itemLabel")

        val serverAddress = getServerAddress(context)
        val useHttps = getUseHttps(context)
        val protocol = if (useHttps) "https" else "http"
        val url = "$protocol://$serverAddress:$SAM3_DEFAULT_PORT$SAM3_BG_REMOVE_ENDPOINT"

        // URI → 바이트 배열 읽기
        val imageBytes = context.contentResolver.openInputStream(imageUri)?.use { it.readBytes() }
            ?: return@withContext null

        // MIME 타입 판별
        val mimeType = context.contentResolver.getType(imageUri) ?: "image/jpeg"
        val extension = when {
            mimeType.contains("png")  -> "png"
            mimeType.contains("webp") -> "webp"
            else                      -> "jpg"
        }

        onProgress(20, "SAM2 서버로 전송 중...$itemLabel")

        val client = createOkHttpClient(useHttps).newBuilder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .readTimeout(180, TimeUnit.SECONDS) // SAM2 추론 대기
            .callTimeout(240, TimeUnit.SECONDS)
            .build()

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image",
                "input.$extension",
                imageBytes.toRequestBody(mimeType.toMediaType())
            )
            .addFormDataPart("prompt", prompt)
            .build()

        val httpRequest = Request.Builder()
            .url(url)
            .post(requestBody)
            .build()

        onProgress(35, "SAM2 객체 감지 및 세그멘테이션 중...$itemLabel")

        val response = client.newCall(httpRequest).execute()

        if (!response.isSuccessful) {
            val errBody = response.body?.string() ?: ""
            android.util.Log.e("SAM2", "서버 오류 ${response.code}: $errBody")
            response.close()
            return@withContext null
        }

        onProgress(85, "결과 수신 중...$itemLabel")

        val pngBytes = response.body?.bytes()
        response.close()
        if (pngBytes == null || pngBytes.isEmpty()) return@withContext null

        onProgress(93, "앱 라이브러리 저장 중...$itemLabel")

        // 결과 PNG를 앱 라이브러리에 저장
        val outputDir = context.getExternalFilesDir(null) ?: return@withContext null
        outputDir.mkdirs()
        val outFile = File(outputDir, "sam2_bg_removed_${System.currentTimeMillis()}.png")
        FileOutputStream(outFile).use { it.write(pngBytes) }

        onProgress(98, "완료$itemLabel")
        Uri.fromFile(outFile)
    } catch (e: Exception) {
        android.util.Log.e("SAM2", "SAM2 서버 통신 실패", e)
        null
    }
}

private suspend fun downloadPlyResult(context: Context, taskId: String): File? {
    return withContext(Dispatchers.IO) {
        try {
            val serverAddress = getServerAddress(context)
            val serverPort = getServerPort(context)
            val useHttps = getUseHttps(context)
            val client = createOkHttpClient(useHttps)

            val protocol = if (useHttps) "https" else "http"
            val url = "$protocol://$serverAddress:$serverPort$DOWNLOAD_ENDPOINT/$taskId"
            val request = Request.Builder().url(url).get().build()
            val response = client.newCall(request).execute()
            val body = response.body
            if (!response.isSuccessful || body == null) {
                response.close()
                return@withContext null
            }

            val modelsDir = File(context.getExternalFilesDir(null), "models").apply { mkdirs() }
            val outFile = File(modelsDir, "3d_model_$taskId.ply")
            body.byteStream().use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            response.close()
            outFile
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

private suspend fun uploadZipAndRunPipeline(
    context: Context,
    zipFile: File,
    prompt: String = "",
    onProgress: (progress: Int, message: String) -> Unit
): Boolean {
    // [추가] 서비스 알림 업데이트를 위한 헬퍼 함수
    val updateNotification = { p: Int, msg: String ->
        onProgress(p, msg)
        val intent = Intent(context, AppForegroundService::class.java).apply {
            putExtra(AppForegroundService.EXTRA_PROGRESS, p)
            putExtra(AppForegroundService.EXTRA_MESSAGE, msg)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(intent)
        } else {
            context.startService(intent)
        }
    }

    // [추가] 업로드 및 서버 작업 중 백그라운드 유지를 위한 서비스 시작
    val serviceIntent = Intent(context, AppForegroundService::class.java)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        context.startForegroundService(serviceIntent)
    } else {
        context.startService(serviceIntent)
    }

    try {
        // 1) 업로드 -> task_id 확보
        updateNotification(5, "파일 업로드 중...")
        val noResponseMsg = "서버에 대한 응답이 없습니다.\n서버 연결을 확인해주십시오."
        val taskId = startServerTaskWithZip(context, zipFile, prompt)
        if (taskId.isNullOrBlank()) {
            updateNotification(0, noResponseMsg)
            context.stopService(serviceIntent)
            return false
        }

        // 로컬 ZIP은 서버에 올라갔으면 삭제(저장공간 확보)
        try { zipFile.delete() } catch (_: Exception) {}

        // 2) 상태 폴링
        val start = System.currentTimeMillis()
        val timeoutMs = 30L * 60L * 1000L // 30분
        var lastServerResponseAt = System.currentTimeMillis()
        while (System.currentTimeMillis() - start < timeoutMs) {
            val st = fetchServerTaskStatus(context, taskId)
            if (st != null) {
                lastServerResponseAt = System.currentTimeMillis()
                updateNotification(st.progressPercent.coerceIn(0, 100), st.message)
                when (st.status) {
                    "COMPLETED" -> break
                    "FAILED" -> {
                        context.stopService(serviceIntent)
                        return false
                    }
                }
            } else {
                // 60초 이상 서버 응답이 없으면 중단
                if (System.currentTimeMillis() - lastServerResponseAt >= 60_000L) {
                    updateNotification(0, noResponseMsg)
                    context.stopService(serviceIntent)
                    return false
                }
            }
            delay(1000)
        }

        // 3) 결과 다운로드 (.ply)
        updateNotification(95, "결과 다운로드 중...")
        val result = downloadPlyResult(context, taskId) ?: run {
            context.stopService(serviceIntent)
            return false
        }
        
        updateNotification(100, "완료되었습니다!")
        context.stopService(serviceIntent)
        return true
    } catch (e: Exception) {
        e.printStackTrace()
        context.stopService(serviceIntent)
        return false
    }
}

private fun listImageFiles(dir: File): List<File> {
    val exts = setOf("jpg", "jpeg", "png", "webp")
    return dir.listFiles()
        ?.filter { it.isFile && exts.contains(it.extension.lowercase()) }
        ?.sortedBy { it.name }
        ?: emptyList()
}

private fun sharpenBitmapFast(src: Bitmap, amount: Float = 1.2f): Bitmap {
    val w = src.width
    val h = src.height
    if (w < 3 || h < 3) return src.copy(src.config ?: Bitmap.Config.ARGB_8888, true)

    val inPixels = IntArray(w * h)
    src.getPixels(inPixels, 0, w, 0, 0, w, h)
    val outPixels = IntArray(w * h)

    fun clamp(v: Int): Int = when {
        v < 0 -> 0
        v > 255 -> 255
        else -> v
    }

    fun r(p: Int) = (p shr 16) and 0xFF
    fun g(p: Int) = (p shr 8) and 0xFF
    fun b(p: Int) = p and 0xFF
    fun a(p: Int) = (p ushr 24) and 0xFF

    for (y in 0 until h) {
        val row = y * w
        for (x in 0 until w) {
            val idx = row + x
            if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
                outPixels[idx] = inPixels[idx]
                continue
            }

            val c = inPixels[idx]
            val up = inPixels[idx - w]
            val down = inPixels[idx + w]
            val left = inPixels[idx - 1]
            val right = inPixels[idx + 1]

            val cr = r(c); val cg = g(c); val cb = b(c)
            val blurR = (cr + r(up) + r(down) + r(left) + r(right)) / 5
            val blurG = (cg + g(up) + g(down) + g(left) + g(right)) / 5
            val blurB = (cb + b(up) + b(down) + b(left) + b(right)) / 5

            val outR = (cr + amount * (cr - blurR)).roundToInt()
            val outG = (cg + amount * (cg - blurG)).roundToInt()
            val outB = (cb + amount * (cb - blurB)).roundToInt()

            outPixels[idx] =
                (a(c) shl 24) or (clamp(outR) shl 16) or (clamp(outG) shl 8) or clamp(outB)
        }
    }

    val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    out.setPixels(outPixels, 0, w, 0, 0, w, h)
    return out
}

private fun applyExifOrientation(src: Bitmap, orientation: Int): Bitmap {
    if (orientation == ExifInterface.ORIENTATION_NORMAL || orientation == ExifInterface.ORIENTATION_UNDEFINED) {
        return src
    }
    val m = Matrix()
    when (orientation) {
        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.preScale(-1f, 1f)
        ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
        ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.preScale(1f, -1f)
        ExifInterface.ORIENTATION_TRANSPOSE -> { // flip + rotate 90
            m.preScale(-1f, 1f)
            m.postRotate(90f)
        }
        ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
        ExifInterface.ORIENTATION_TRANSVERSE -> { // flip + rotate 270
            m.preScale(-1f, 1f)
            m.postRotate(270f)
        }
        ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
        else -> return src
    }
    return try {
        Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    } catch (_: Exception) {
        src
    }
}

private suspend fun enhanceDatasetFolders(
    context: Context,
    folders: List<File>,
    onProgress: (processed: Int, total: Int, message: String, etaMs: Long?) -> Unit
): Boolean {
    return withContext(Dispatchers.IO) {
        try {
            if (folders.isEmpty()) return@withContext false

            val allImageFilesByFolder = folders.associateWith { listImageFiles(it) }
            val total = allImageFilesByFolder.values.sumOf { it.size }
            if (total <= 0) return@withContext false

            val startAt = SystemClock.elapsedRealtime()
            var processed = 0

            fun computeEtaMs(): Long? {
                if (processed <= 0) return null
                val elapsed = SystemClock.elapsedRealtime() - startAt
                val avg = elapsed.toDouble() / processed.toDouble()
                val remain = (total - processed).coerceAtLeast(0)
                return (avg * remain.toDouble()).toLong().coerceAtLeast(0L)
            }

            onProgress(0, total, "선명도 보정 시작...", computeEtaMs())

            for ((folder, files) in allImageFilesByFolder) {
                if (files.isEmpty()) continue

                val parent = folder.parentFile ?: continue
                val tmp = File(parent, "${folder.name}__enhance_tmp_${System.currentTimeMillis()}").apply {
                    if (exists()) deleteRecursively()
                    mkdirs()
                }

                for (f in files) {
                    val raw = BitmapFactory.decodeFile(f.absolutePath)
                    if (raw == null) {
                        processed++
                        onProgress(processed, total, "이미지 로드 실패: ${f.name}", computeEtaMs())
                        continue
                    }
                    val oriented = try {
                        val exif = ExifInterface(f.absolutePath)
                        val orientation = exif.getAttributeInt(
                            ExifInterface.TAG_ORIENTATION,
                            ExifInterface.ORIENTATION_NORMAL
                        )
                        applyExifOrientation(raw, orientation)
                    } catch (_: Exception) {
                        raw
                    }.also { bmp2 ->
                        if (bmp2 !== raw) {
                            try { raw.recycle() } catch (_: Exception) {}
                        }
                    }

                    val out = try {
                        sharpenBitmapFast(oriented, amount = 1.25f)
                    } finally {
                        try { oriented.recycle() } catch (_: Exception) {}
                    }

                    val outFile = File(tmp, f.name)
                    try {
                        FileOutputStream(outFile).use { fos ->
                            val fmt = when (f.extension.lowercase()) {
                                "png" -> Bitmap.CompressFormat.PNG
                                "webp" -> if (Build.VERSION.SDK_INT >= 30) Bitmap.CompressFormat.WEBP_LOSSY else Bitmap.CompressFormat.WEBP
                                else -> Bitmap.CompressFormat.JPEG
                            }
                            val quality = if (fmt == Bitmap.CompressFormat.PNG) 100 else 95
                            out.compress(fmt, quality, fos)
                        }

                        // 저장된 결과는 EXIF 회전이 다시 적용되지 않도록 NORMAL로 고정
                        if (outFile.extension.lowercase() == "jpg" || outFile.extension.lowercase() == "jpeg") {
                            try {
                                val outExif = ExifInterface(outFile.absolutePath)
                                outExif.setAttribute(
                                    ExifInterface.TAG_ORIENTATION,
                                    ExifInterface.ORIENTATION_NORMAL.toString()
                                )
                                outExif.saveAttributes()
                            } catch (_: Exception) {}
                        }
                    } finally {
                        try { out.recycle() } catch (_: Exception) {}
                    }

                    processed++
                    onProgress(processed, total, "처리 중... (${processed}/${total})", computeEtaMs())
                }

                // 원본 폴더 삭제 + 임시 폴더를 원래 이름으로 교체
                val backup = File(parent, "${folder.name}__old_${System.currentTimeMillis()}")
                val originalPath = folder.absolutePath
                val original = File(originalPath)

                // 1) 원본 폴더를 백업 이름으로 이동(가능하면)
                if (original.exists()) {
                    val renamed = original.renameTo(backup)
                    if (!renamed) {
                        // rename 실패 시, 그래도 요구사항(자동 삭제)을 위해 삭제 시도
                        try { original.deleteRecursively() } catch (_: Exception) {}
                    }
                }

                // 2) tmp를 원래 경로로 rename 시도
                val target = File(originalPath)
                val moved = tmp.renameTo(target)
                if (!moved) {
                    // rename 실패 시 직접 이동
                    target.mkdirs()
                    tmp.listFiles()?.forEach { child ->
                        try {
                            val dest = File(target, child.name)
                            child.copyTo(dest, overwrite = true)
                            child.delete()
                        } catch (_: Exception) {}
                    }
                    try { tmp.deleteRecursively() } catch (_: Exception) {}
                }

                // 3) 백업 삭제
                try { backup.deleteRecursively() } catch (_: Exception) {}
            }

            onProgress(total, total, "선명도 보정 완료", 0L)
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
}

data class PlyParseResult(
    val points: FloatArray,
    val count: Int
)

fun parsePlyPoints(file: File): PlyParseResult? {
    if (!file.exists()) return null
    return try {
        var vertexCount = 0
        var headerEnded = false
        val rawPoints = ArrayList<Float>(1024)

        file.bufferedReader().useLines { lines ->
            lines.forEach { line ->
                val trimmed = line.trim()
                if (trimmed.isEmpty()) return@forEach
                if (!headerEnded) {
                    if (trimmed.startsWith("element vertex")) {
                        val parts = trimmed.split(" ")
                        if (parts.size >= 3) {
                            vertexCount = parts[2].toIntOrNull() ?: 0
                        }
                    } else if (trimmed == "end_header") {
                        headerEnded = true
                    }
                } else {
                    if (vertexCount > 0 && rawPoints.size >= vertexCount * 3) return@forEach
                    val parts = trimmed.split(Regex("\\s+"))
                    if (parts.size >= 3) {
                        rawPoints.add(parts[0].toFloatOrNull() ?: 0f)
                        rawPoints.add(parts[1].toFloatOrNull() ?: 0f)
                        rawPoints.add(parts[2].toFloatOrNull() ?: 0f)
                    }
                }
            }
        }

        val count = rawPoints.size / 3
        if (count == 0) return null

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

        PlyParseResult(points, count)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

private class PlyRenderer : GLSurfaceView.Renderer {
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
        aspectRatio = if (height > 0) width.toFloat() / height.toFloat() else 1f
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val buffer = pointBuffer ?: return
        if (vertexCount <= 0) return

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

private class PlySurfaceView(context: Context) : GLSurfaceView(context) {
    private val renderer = PlyRenderer()
    private var lastX = 0f
    private var lastY = 0f

    init {
        setEGLContextClientVersion(2)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun loadModel(file: File) {
        Thread {
            val parsed = parsePlyPoints(file)
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

/**
 * 서버 연결 테스트 함수
 * @param context 컨텍스트
 * @param serverAddress 테스트할 서버 주소 (IP 또는 도메인)
 * @param serverPort 테스트할 서버 포트
 * @param useHttps HTTPS 사용 여부
 * @return 연결 성공 여부
 */
private suspend fun testServerConnection(
    context: Context,
    serverAddress: String,
    serverPort: Int,
    useHttps: Boolean
): Boolean {
    return withContext(Dispatchers.IO) {
        try {
            val client = createOkHttpClient(useHttps).newBuilder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(10, TimeUnit.SECONDS)
                .build()

            // 간단한 HEAD 요청으로 서버 연결 테스트
            val protocol = if (useHttps) "https" else "http"
            val url = "$protocol://$serverAddress:$serverPort$UPLOAD_ENDPOINT"
            val request = Request.Builder()
                .url(url)
                .head() // HEAD 요청으로 서버 응답 확인
                .build()

            val response = client.newCall(request).execute()
            val success = response.code in 200..499 // 4xx는 서버가 응답했다는 의미
            response.close()
            
            success
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
}

@Composable
fun ServerSettingsScreen(
    onBack: () -> Unit
) {
    val context = LocalContext.current
    var serverAddress by remember { mutableStateOf(getServerAddress(context)) }
    var serverPort by remember { mutableStateOf(getServerPort(context).toString()) }
    var useHttps by remember { mutableStateOf(getUseHttps(context)) }
    var isTesting by remember { mutableStateOf(false) }
    var testResult by remember { mutableStateOf<String?>(null) }
    val coroutineScope = rememberCoroutineScope()

    BackHandler {
        onBack()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppBackgroundColor)
            .padding(16.dp)
    ) {
        // 헤더
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "서버 설정",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // 서버 주소 입력 필드 (IP 또는 도메인)
        OutlinedTextField(
            value = serverAddress,
            onValueChange = { serverAddress = it },
            label = { Text("서버 주소 (IP 또는 도메인)", color = Color.White) },
            placeholder = { Text("예: 192.168.0.88 또는 example.com", color = Color.LightGray) },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedBorderColor = Color.White,
                unfocusedBorderColor = Color.White.copy(alpha = 0.6f),
                cursorColor = Color.White,
                focusedLabelColor = Color.White,
                unfocusedLabelColor = Color.White.copy(alpha = 0.7f),
                focusedPlaceholderColor = Color.LightGray,
                unfocusedPlaceholderColor = Color.LightGray
            )
        )

        Spacer(modifier = Modifier.height(16.dp))

        // 포트 입력 필드
        OutlinedTextField(
            value = serverPort,
            onValueChange = { newValue ->
                // 숫자만 입력 가능하도록
                if (newValue.all { it.isDigit() } || newValue.isEmpty()) {
                    serverPort = newValue
                }
            },
            label = { Text("포트 번호", color = Color.White) },
            placeholder = { Text("예: 8000", color = Color.LightGray) },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedBorderColor = Color.White,
                unfocusedBorderColor = Color.White.copy(alpha = 0.6f),
                cursorColor = Color.White,
                focusedLabelColor = Color.White,
                unfocusedLabelColor = Color.White.copy(alpha = 0.7f),
                focusedPlaceholderColor = Color.LightGray,
                unfocusedPlaceholderColor = Color.LightGray
            )
        )

        Spacer(modifier = Modifier.height(16.dp))

        // HTTPS 사용 여부 선택
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "HTTPS 사용",
                fontSize = 16.sp,
                modifier = Modifier.weight(1f),
                color = Color.White
            )
            androidx.compose.material3.Switch(
                checked = useHttps,
                onCheckedChange = { useHttps = it },
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Color.White,
                    checkedTrackColor = Color.DarkGray,
                    uncheckedThumbColor = Color.White,
                    uncheckedTrackColor = Color.Gray
                )
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        // 엔드포인트 정보 표시
        Text(
            text = "업로드: $UPLOAD_ENDPOINT",
            fontSize = 14.sp,
            color = Color.White.copy(alpha = 0.6f)
        )
        Text(
            text = "상태조회: $STATUS_ENDPOINT/{task_id}",
            fontSize = 12.sp,
            color = Color.White.copy(alpha = 0.5f)
        )
        Text(
            text = "다운로드: $DOWNLOAD_ENDPOINT/{task_id}",
            fontSize = 12.sp,
            color = Color.White.copy(alpha = 0.5f)
        )

        Spacer(modifier = Modifier.height(8.dp))

        // 전체 URL 미리보기
        val portValue = serverPort.toIntOrNull() ?: DEFAULT_SERVER_PORT
        val protocol = if (useHttps) "https" else "http"
        val previewUrl = if (serverAddress.isNotBlank() && serverPort.isNotBlank()) {
            "$protocol://$serverAddress:$portValue$UPLOAD_ENDPOINT"
        } else {
            ""
        }
        
        if (previewUrl.isNotEmpty()) {
            Text(
                text = "URL: $previewUrl",
                fontSize = 12.sp,
                color = Color.White.copy(alpha = 0.8f),
                modifier = Modifier.fillMaxWidth()
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // 연결 테스트 버튼
        val canTest = !isTesting && serverAddress.isNotBlank() && serverPort.isNotBlank()
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(24.dp))
                .background(if (canTest) Color.White else Color.Gray)
                .clickable(enabled = canTest) {
                    isTesting = true
                    testResult = null
                    coroutineScope.launch {
                        val port = serverPort.toIntOrNull() ?: DEFAULT_SERVER_PORT
                        val success = testServerConnection(context, serverAddress, port, useHttps)
                        testResult = if (success) {
                            "연결 성공!"
                        } else {
                            "연결 실패. 서버 주소, 포트, 프로토콜을 확인하세요."
                        }
                        isTesting = false
                    }
                }
                .padding(vertical = 14.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = if (isTesting) "테스트 중..." else "서버 연결 테스트",
                color = Color.Black,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }

        // 테스트 결과 표시
        testResult?.let { result ->
            Spacer(modifier = Modifier.height(16.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Color.White.copy(alpha = 0.15f),
                        RoundedCornerShape(8.dp)
                    )
                    .padding(16.dp)
            ) {
                Text(
                    text = result,
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        // 저장 버튼
        val canSave = serverAddress.isNotBlank() && serverPort.isNotBlank()
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(24.dp))
                .background(if (canSave) Color.White else Color.Gray)
                .clickable(enabled = canSave) {
                    val port = serverPort.toIntOrNull() ?: DEFAULT_SERVER_PORT
                    saveServerSettings(context, serverAddress, port, useHttps)
                    onBack()
                }
                .padding(vertical = 14.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "저장",
                color = Color.Black,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
// [추가] 백그라운드 작업 유지를 위한 포그라운드 서비스
class AppForegroundService : Service() {
    companion object {
        const val CHANNEL_ID = "AppForegroundServiceChannel"
        const val NOTIFICATION_ID = 1
        const val EXTRA_PROGRESS = "extra_progress"
        const val EXTRA_MESSAGE = "extra_message"
    }

    override fun onCreate() {
        super.onCreate()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "앱 실행 유지 서비스",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            if (manager != null) {
                manager.createNotificationChannel(channel)
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val progress = intent?.getIntExtra(EXTRA_PROGRESS, -1) ?: -1
        val message = intent?.getStringExtra(EXTRA_MESSAGE) ?: "백그라운드에서 작업을 유지하고 있습니다."
        
        val notificationBuilder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("어플리케이션 작업 중")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOnlyAlertOnce(true) // 알림 업데이트 시 소리/진동 반복 방지

        if (progress in 0..100) {
            notificationBuilder.setContentText("$message ($progress%)")
            notificationBuilder.setProgress(100, progress, false)
        } else {
            notificationBuilder.setContentText(message)
        }

        val notification = notificationBuilder.build()
        startForeground(NOTIFICATION_ID, notification)
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null
}