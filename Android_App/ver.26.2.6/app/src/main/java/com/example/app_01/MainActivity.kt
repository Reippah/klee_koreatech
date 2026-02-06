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
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.FocusMeteringResult
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.MeteringPointFactory
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
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
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Public
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.LockOpen
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import kotlin.math.roundToInt
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.PointerInputChange
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
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
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.delay
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
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
    RESOLUTION_1920x1080(1080, 1920),  // FHD 세로 방향 (가로 1080, 세로 1920)
    RESOLUTION_518x518(518, 518),      // VGGT 정사각형
    RESOLUTION_1024x1024(1024, 1024)   // 사물 스캔 전용 정사각형
}

enum class MainTab {
    LIBRARY, CAMERA, CREATE, PROFILE
}

enum class LibraryTab {
    GALLERY, DATASET, MODEL_3D
}

enum class LibraryDetailScreen {
    NONE, DATASET_FOLDER, MODEL_VIEWER
}

enum class CameraEntryMode {
    OBJECT, SPACE_2D, SPACE_3D
}

private fun CameraEntryMode.isSpaceMode(): Boolean = this != CameraEntryMode.OBJECT
private fun CameraEntryMode.isObjectMode(): Boolean = this == CameraEntryMode.OBJECT

private val AppBackgroundColor = Color.Black
private val BottomBarBackgroundColor = Color(0xFF9CD83B)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
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
            load(MediaActionSound.SHUTTER_CLICK)
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
    var focusPointInPreview by remember { mutableStateOf<Offset?>(null) }
    var focusPointInRoot by remember { mutableStateOf<Offset?>(null) }
    var isFocusLocked by remember { mutableStateOf(false) }
    var lockedFocusPoint by remember { mutableStateOf<Offset?>(null) }
    var previewOriginInRoot by remember { mutableStateOf<Offset?>(null) }
    var datasetDir by remember { mutableStateOf<File?>(null) }
    var isDatasetCollectionEnabled by remember { mutableStateOf(true) }
    val focusScope = rememberCoroutineScope()
    var selectedResolution by remember { mutableStateOf(ResolutionPreset.RESOLUTION_1920x1080) }
    var azimuthDegrees by remember { mutableStateOf(0f) }
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

    LaunchedEffect(cameraEntryMode) {
        if (cameraEntryMode.isObjectMode()) {
            selectedResolution = ResolutionPreset.RESOLUTION_1024x1024
        } else if (selectedResolution == ResolutionPreset.RESOLUTION_1024x1024) {
            selectedResolution = ResolutionPreset.RESOLUTION_1920x1080
        }
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

    // 초점 고정 상태에 따라 데이터셋 수집 제어 (고정 후 3초 뒤 시작)
    LaunchedEffect(isRecording, captureMode, isFocusLocked) {
        if (captureMode != CaptureMode.VIDEO || !isRecording) {
            isDatasetCollectionEnabled = true
            return@LaunchedEffect
        }
        if (!isFocusLocked) {
            isDatasetCollectionEnabled = false
        } else {
            isDatasetCollectionEnabled = false
            delay(3000)
            if (isRecording && isFocusLocked && captureMode == CaptureMode.VIDEO) {
                isDatasetCollectionEnabled = true
            }
        }
    }

    // 동영상 촬영 시 구역 체크: 1초 간격으로 현재 구역 기록 (사물/공간 모두)
    LaunchedEffect(isRecording, cameraEntryMode, captureMode, currentPitchIndex, isDatasetCollectionEnabled, datasetDir) {
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
                            // [수정] 촬영 전 초점 재조정 (선명한 화질 확보)
                            val cam = camera
                            val preview = previewView
                            val focusPt = lockedFocusPoint
                            
                            if (cam != null && preview != null && focusPt != null) {
                                try {
                                    withContext(Dispatchers.Main) {
                                        val factory = preview.meteringPointFactory
                                        val point = factory.createPoint(focusPt.x, focusPt.y)
                                        val action = FocusMeteringAction.Builder(point)
                                            .disableAutoCancel()
                                            .build()
                                        
                                        val future = cam.cameraControl.startFocusAndMetering(action)
                                        // 초점 완료 대기 (최대 1초)
                                        withContext(Dispatchers.IO) {
                                            try {
                                                future.get(1000, TimeUnit.MILLISECONDS)
                                            } catch (e: Exception) {
                                                // 타임아웃 등 무시하고 진행
                                            }
                                        }
                                    }
                                } catch (e: Exception) {
                                    e.printStackTrace()
                                }
                            }
                            
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
        if (captureMode == CaptureMode.VIDEO &&
            isRecording &&
            currentPitchIndex < pitchTargets.size &&
            capturedSectors.size >= sectorCount
        ) {
            currentPitchIndex += 1
            capturedSectors = emptySet()
        }
    }

    // 초점 UI 자동 숨김 (3초 후, 잠금 상태는 유지)
    LaunchedEffect(focusPointInRoot) {
        if (focusPointInRoot != null) {
            delay(3000)
            focusPointInPreview = null
            focusPointInRoot = null
            // 초점 고정 상태는 유지 (isFocusLocked는 변경하지 않음)
        }
    }

    // 카메라 바인딩 함수
    fun bindCamera(view: PreviewView) {
        val executor = ContextCompat.getMainExecutor(context)
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(lensFacing)
                    .build()

                cameraProvider.unbindAll()

                // 선택된 해상도
                val isVggtModeInBind = selectedResolution == ResolutionPreset.RESOLUTION_518x518
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

                // Preview에도 해상도 설정 적용
                val preview = Preview.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .build()
                    .also {
                        it.setSurfaceProvider(view.surfaceProvider)
                    }

                val newImageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setResolutionSelector(resolutionSelector)
                    .build()

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
                val isVggt = selectedResolution == ResolutionPreset.RESOLUTION_518x518
                val recorder = androidx.camera.video.Recorder.Builder()
                    .setQualitySelector(
                        androidx.camera.video.QualitySelector.from(
                            if (isVggt) androidx.camera.video.Quality.SD 
                            else androidx.camera.video.Quality.HIGHEST
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

    // 초점 설정 함수
    fun setFocus(x: Float, y: Float) {
        val cameraInstance = camera
        val previewViewInstance = previewView
        if (cameraInstance != null && previewViewInstance != null) {
            try {
                // PreviewView의 스케일/크롭을 반영한 메터링 포인트 팩토리 사용
                val point = previewViewInstance.meteringPointFactory.createPoint(x, y)
                val action = FocusMeteringAction.Builder(point)
                    .setAutoCancelDuration(3, TimeUnit.SECONDS)
                    .build()
                
                val future = cameraInstance.cameraControl.startFocusAndMetering(action)
                future.addListener({
                    try {
                        val result = future.get()
                        if (result.isFocusSuccessful) {
                            // 초점 성공
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }, ContextCompat.getMainExecutor(context))
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    // lensFacing, captureMode, selectedResolution 변경 시 카메라 재바인딩
    LaunchedEffect(lensFacing, captureMode, selectedResolution, previewView) {
        isCameraReady = false
        previewView?.let { bindCamera(it) }
    }

    // 선택된 해상도의 비율 계산 (정사각형: 518/1024, FHD는 1920:1080)
    val isSquareMode = selectedResolution.width == selectedResolution.height
    val density = LocalDensity.current

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
        // 해상도 변경 시 PreviewView를 완전히 재생성하기 위해 key 사용
        key(selectedResolution) {
            if (isSquareMode) {
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
                        modifier = Modifier
                            .fillMaxSize()
                            .pointerInput(isFocusLocked) {
                                detectTapGestures { offset ->
                                    val origin = previewOriginInRoot
                                    if (!isFocusLocked) {
                                        focusPointInPreview = offset
                                        focusPointInRoot = if (origin != null) {
                                            Offset(origin.x + offset.x, origin.y + offset.y)
                                        } else {
                                            offset
                                        }
                                        setFocus(offset.x, offset.y)
                                    } else {
                                        // 초점이 고정된 상태에서 터치하면 자물쇠 UI 다시 표시
                                        focusPointInPreview = offset
                                        focusPointInRoot = if (origin != null) {
                                            Offset(origin.x + offset.x, origin.y + offset.y)
                                        } else {
                                            offset
                                        }
                                    }
                                }
                            }
                    )
                }
            } else {
                // FHD 모드: 9:16 세로 비율
                AndroidView(
                    factory = { ctx ->
                        PreviewView(ctx).apply {
                            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                            scaleType = PreviewView.ScaleType.FIT_CENTER
                            previewView = this
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(1080f / 1920f)
                        .onGloballyPositioned { coords ->
                            previewOriginInRoot = coords.positionInRoot()
                        }
                        .pointerInput(isFocusLocked) {
                            detectTapGestures { offset ->
                                val origin = previewOriginInRoot
                                if (!isFocusLocked) {
                                    focusPointInPreview = offset
                                    focusPointInRoot = if (origin != null) {
                                        Offset(origin.x + offset.x, origin.y + offset.y)
                                    } else {
                                        offset
                                    }
                                    setFocus(offset.x, offset.y)
                                } else {
                                    // 초점이 고정된 상태에서 터치하면 자물쇠 UI 다시 표시
                                    focusPointInPreview = offset
                                    focusPointInRoot = if (origin != null) {
                                        Offset(origin.x + offset.x, origin.y + offset.y)
                                    } else {
                                        offset
                                    }
                                }
                            }
                        }
                )
            }
        }

        // 초점 표시 - 원형 테두리와 자물쇠 UI
        focusPointInRoot?.let { point ->
            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .offset {
                        with(density) {
                            IntOffset(
                                (point.x - 40.dp.toPx()).toInt(),
                                (point.y - 40.dp.toPx()).toInt()
                            )
                        }
                    }
                    .size(80.dp)
            ) {
                // 원형 테두리 (얇은 선)
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .border(1.5.dp, Color.White, CircleShape)
                )
                
                // 자물쇠 아이콘 (오른쪽 위에 배치)
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .offset(x = 4.dp, y = (-4).dp)
                        .size(20.dp)
                        .clickable {
                            isFocusLocked = !isFocusLocked
                            if (isFocusLocked) {
                                lockedFocusPoint = focusPointInPreview
                            } else {
                                lockedFocusPoint = null
                            }
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isFocusLocked) Icons.Filled.Lock else Icons.Filled.LockOpen,
                        contentDescription = if (isFocusLocked) "초점 잠금" else "초점 해제",
                        tint = if (isFocusLocked) Color(0xFFFFD700) else Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                }
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

        if (captureMode == CaptureMode.VIDEO && isRecording && !isFocusLocked) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 110.dp)
                    .background(Color.Red, RoundedCornerShape(8.dp))
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Text(
                    text = "초점을 고정하지 않았습니다.\n초점 설정 후 고정해주세요.",
                    color = Color.White,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
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
            if (cameraEntryMode.isObjectMode()) {
                Box(
                    modifier = Modifier
                        .background(Color(0xFF1A1A1A), RoundedCornerShape(16.dp))
                        .border(1.dp, Color.White.copy(alpha = 0.6f), RoundedCornerShape(16.dp))
                        .padding(horizontal = 14.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = "1024x1024",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            } else {
                TopMenuSegmented(
                    leftText = "FHD",
                    rightText = "VGGT",
                    isLeftSelected = selectedResolution == ResolutionPreset.RESOLUTION_1920x1080,
                    onLeftClick = { selectedResolution = ResolutionPreset.RESOLUTION_1920x1080 },
                    onRightClick = { selectedResolution = ResolutionPreset.RESOLUTION_518x518 }
                )
            }
        }

        // 동영상 촬영 중 구역 수집 정보 표시 (사물/공간 모두)
        if (captureMode == CaptureMode.VIDEO && isRecording) {
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
            if (!isAllPitchCompleted && !isPitchAligned) {
                val targetPitch = currentTargetPitch
                val warningText = if (targetPitch == null) {
                    null
                } else {
                    val delta = effectivePitchDegrees - targetPitch
                    when {
                        delta < -pitchTolerance -> "기기를 아래쪽으로 기울여주세요"
                        delta > pitchTolerance -> "기기를 위쪽으로 기울여주세요"
                        else -> null
                    }
                }
                if (warningText != null) {
                    Text(
                        text = warningText,
                        color = Color(0xFFFF5252),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .align(Alignment.TopCenter)
                            .padding(
                                top = when {
                                    isRecording && selectedResolution == ResolutionPreset.RESOLUTION_518x518 -> 140.dp
                                    isRecording -> 190.dp
                                    selectedResolution == ResolutionPreset.RESOLUTION_518x518 -> 120.dp
                                    else -> 160.dp
                                }
                            )
                            .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }
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

            // 동영상 촬영 중 구역 링 표시 (사물/공간 모두)
            val showRing = captureMode == CaptureMode.VIDEO && isRecording
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
                                    mediaActionSound.play(MediaActionSound.SHUTTER_CLICK)
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
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                            CameraSelector.LENS_FACING_FRONT
                        } else {
                            CameraSelector.LENS_FACING_BACK
                        }
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
            SensorInfo("회전 벡터", Sensor.TYPE_ROTATION_VECTOR, Icons.Default.Explore),
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
                val isPresent = sensorManager.getDefaultSensor(item.type) != null
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
    var isEnhancing by remember { mutableStateOf(false) }
    var enhanceProgress by remember { mutableStateOf(0 to 100) } // (current, total)
    var enhanceMessage by remember { mutableStateOf<String?>(null) }
    var enhanceEtaMs by remember { mutableStateOf<Long?>(null) }
    var enhanceSourceTab by remember { mutableStateOf<LibraryTab?>(null) }
    var showEnhanceResultPopup by remember { mutableStateOf(false) }
    var enhanceResultPopupMessage by remember { mutableStateOf<String?>(null) }
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
            if (libraryDetailScreen == LibraryDetailScreen.MODEL_VIEWER && currentPlyModel != null) {
                BackHandler {
                    libraryDetailScreen = LibraryDetailScreen.NONE
                    currentPlyModel = null
                }
                Box(modifier = Modifier.fillMaxSize()) {
                    AndroidView(
                        factory = { ctx ->
                            PlySurfaceView(ctx).apply {
                                loadModel(currentPlyModel!!.file)
                            }
                        },
                        modifier = Modifier.fillMaxSize()
                    )
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
                                        libraryDetailScreen = LibraryDetailScreen.MODEL_VIEWER
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
                                            isEnhancing -> {
                                                uploadMessage = "선명도 보정 중에는 업로드할 수 없습니다"
                                            }
                                            selectedDatasetFolders.isEmpty() -> {
                                                uploadMessage = "선택된 폴더가 없습니다"
                                            }
                                            else -> {
                                            isUploading = true
                                            uploadProgress = 0 to 100
                                            uploadMessage = "업로드 준비 중..."

                                            CoroutineScope(Dispatchers.IO).launch {
                                                try {
                                                    val folderFiles = selectedDatasetFolders
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
                                                            if (uploadMessage != noServerResponseMsg) {
                                                                uploadMessage = "업로드 실패"
                                                            }
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
                                        }
                                    }
                            )
                            Text(
                                text = "선명도 보정",
                                color = if (selectedDatasetFolders.isNotEmpty()) Color.White else Color.White.copy(alpha = 0.6f),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(
                                        if (selectedDatasetFolders.isNotEmpty()) Color(0xFF7ED321)
                                        else Color(0xFF7ED321).copy(alpha = 0.4f)
                                    )
                                    .clickable {
                                        enhanceSourceTab = LibraryTab.DATASET
                                        when {
                                            isUploading || isEnhancing -> {
                                                enhanceMessage = "다른 작업이 진행 중입니다"
                                            }
                                            selectedDatasetFolders.isEmpty() -> {
                                                enhanceMessage = "선택된 폴더가 없습니다"
                                            }
                                            else -> {
                                                isEnhancing = true
                                                enhanceProgress = 0 to 100
                                                enhanceMessage = "선명도 보정 준비 중..."
                                                enhanceEtaMs = null

                                                CoroutineScope(Dispatchers.IO).launch {
                                                    try {
                                                        val folders = selectedDatasetFolders
                                                            .map { File(it) }
                                                            .filter { it.exists() && it.isDirectory }
                                                        val ok = enhanceDatasetFolders(
                                                            context = context,
                                                            folders = folders,
                                                            onProgress = { processed, total, msg, eta ->
                                                                CoroutineScope(Dispatchers.Main).launch {
                                                                    enhanceProgress = processed to total
                                                                    enhanceMessage = msg
                                                                    enhanceEtaMs = eta
                                                                }
                                                            }
                                                        )
                                                        withContext(Dispatchers.Main) {
                                                            isEnhancing = false
                                                            if (ok) {
                                                                enhanceResultPopupMessage = "선명도 보정이 완료되었습니다."
                                                                showEnhanceResultPopup = true
                                                                selectedDatasetFolders = emptySet()
                                                                isDatasetEditMode = false
                                                                loadDatasetFolders(context) { folders2 ->
                                                                    datasetFolders = folders2
                                                                }
                                                            } else {
                                                                enhanceResultPopupMessage = "선명도 보정 실패"
                                                                showEnhanceResultPopup = true
                                                            }
                                                        }
                                                    } catch (e: Exception) {
                                                        e.printStackTrace()
                                                        withContext(Dispatchers.Main) {
                                                            isEnhancing = false
                                                            enhanceResultPopupMessage = "선명도 보정 실패"
                                                            showEnhanceResultPopup = true
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    .padding(horizontal = 12.dp, vertical = 6.dp)
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
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // 업로드 버튼
                    Icon(
                        imageVector = Icons.Filled.CloudUpload,
                        contentDescription = "업로드",
                        tint = if (selectedItems.isNotEmpty()) Color.Green else Color.Green.copy(alpha = 0.4f),
                        modifier = Modifier
                            .size(32.dp)
                            .clickable {
                                uploadSourceTab = LibraryTab.GALLERY
                                when {
                                    isUploading -> {
                                        uploadMessage = "이미 업로드 중입니다"
                                    }
                                    selectedItems.isEmpty() -> {
                                        uploadMessage = "선택된 미디어가 없습니다"
                                    }
                                    else -> {
                                    // 업로드 시작
                                    isUploading = true
                                    uploadProgress = 0 to 100
                                    uploadMessage = "업로드 준비 중..."

                                    // 백그라운드에서 업로드 실행
                                    CoroutineScope(Dispatchers.IO).launch {
                                        val itemsToUpload = selectedItems.toList()
                                        try {
                                            val zipFile = createZipFromUris(
                                                context = context,
                                                uris = itemsToUpload,
                                                zipPrefix = "media"
                                            )
                                            val success = zipFile != null && uploadZipAndRunPipeline(
                                                context = context,
                                                zipFile = zipFile,
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
                                                        if (uploadMessage != noServerResponseMsg) {
                                                            uploadMessage = "업로드 실패"
                                                        }
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
                                }
                            }
                    )
                    // 선택 삭제 버튼
                    Text(
                        text = "삭제",
                        color = if (selectedItems.isNotEmpty()) Color.White else Color.White.copy(alpha = 0.4f),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .clip(RoundedCornerShape(12.dp))
                            .background(if (selectedItems.isNotEmpty()) Color.Red else Color.Red.copy(alpha = 0.4f))
                            .clickable(enabled = !isUploading && selectedItems.isNotEmpty()) {
                                showDeleteConfirm = true
                            }
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                    // 전체 삭제 버튼 (빨간색 채움)
                    Text(
                        text = "전체 삭제",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .clip(RoundedCornerShape(12.dp))
                            .background(Color.Red)
                            .clickable(enabled = !isUploading) {
                                showDeleteAllConfirm = true
                            }
                            .padding(horizontal = 12.dp, vertical = 6.dp)
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

        // 선명도 보정 진행/결과 팝업(UI) - 데이터셋 탭에서만 표시
        if (enhanceSourceTab == libraryTab) {
            if (isEnhancing) {
                val total = enhanceProgress.second.takeIf { it > 0 } ?: 100
                val current = enhanceProgress.first.coerceIn(0, total)
                val percent = ((current.toFloat() / total.toFloat()) * 100f).toInt().coerceIn(0, 100)
                val progressFraction = current.toFloat() / total.toFloat()

                fun formatEta(ms: Long): String {
                    val sec = (ms / 1000L).coerceAtLeast(0L)
                    val m = sec / 60L
                    val s = sec % 60L
                    return String.format("%02d:%02d", m, s)
                }

                Dialog(onDismissRequest = { /* 처리 중에는 닫지 않음 */ }) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF2F2F2F), RoundedCornerShape(16.dp))
                            .padding(20.dp)
                    ) {
                        Column {
                            Text(
                                text = "선명도 보정 중",
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
                                        .background(Color(0xFF4FC3F7), RoundedCornerShape(6.dp))
                                )
                            }
                            Spacer(modifier = Modifier.height(10.dp))
                            enhanceEtaMs?.let { eta ->
                                Text(
                                    text = "예상 남은 시간: ${formatEta(eta)}",
                                    color = Color.White.copy(alpha = 0.85f),
                                    fontSize = 13.sp
                                )
                                Spacer(modifier = Modifier.height(6.dp))
                            }
                            Text(
                                text = enhanceMessage ?: "처리 중...",
                                color = Color.White.copy(alpha = 0.85f),
                                fontSize = 13.sp
                            )
                        }
                    }
                }
            } else {
                if (showEnhanceResultPopup && enhanceResultPopupMessage != null) {
                    Dialog(
                        onDismissRequest = {
                            showEnhanceResultPopup = false
                            enhanceResultPopupMessage = null
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
                                    text = enhanceResultPopupMessage!!,
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
                                                showEnhanceResultPopup = false
                                                enhanceResultPopupMessage = null
                                            }
                                            .padding(horizontal = 16.dp, vertical = 8.dp)
                                    )
                                }
                            }
                        }
                    }
                }

                // 간단 메시지 배너(선택 없음 등)
                enhanceMessage?.let { message ->
                    LaunchedEffect(message) {
                        delay(3000)
                        enhanceMessage = null
                    }
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 10.dp)
                            .background(Color(0xFF2F2F2F), RoundedCornerShape(12.dp))
                            .padding(12.dp)
                    ) {
                        Text(
                            text = message,
                            color = Color.White.copy(alpha = 0.9f),
                            fontSize = 13.sp
                        )
                    }
                }
            }
        }

        // --- 다이얼로그 구역 (어떤 탭에서도 보일 수 있도록 탭 분기 바깥에 배치) ---
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
    onMediaChanged: (Int) -> Unit
) {
    val context = LocalContext.current
    var currentIndex by remember { mutableStateOf(initialIndex.coerceIn(0, mediaList.size - 1)) }

    // 인덱스 변경 시 콜백 호출
    LaunchedEffect(currentIndex) {
        onMediaChanged(currentIndex)
    }

    BackHandler {
        onBack()
    }

    val currentMediaUri = if (currentIndex in mediaList.indices) mediaList[currentIndex] else null
    val isVideo = currentMediaUri?.let { isVideoUri(context, it) } ?: false

    Box(modifier = Modifier.fillMaxSize()) {
        // 현재 미디어 표시
        if (currentMediaUri != null) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        var totalDragX = 0f
                        detectDragGestures(
                            onDragEnd = {
                                // 드래그 종료 시 인덱스 변경
                                val threshold = size.width * 0.3f
                                if (totalDragX > threshold && currentIndex > 0) {
                                    // 오른쪽으로 스와이프 (이전 이미지)
                                    currentIndex--
                                } else if (totalDragX < -threshold && currentIndex < mediaList.size - 1) {
                                    // 왼쪽으로 스와이프 (다음 이미지)
                                    currentIndex++
                                }
                                totalDragX = 0f
                            },
                            onDrag = { _: PointerInputChange, dragAmount: Offset ->
                                // 수평 드래그만 처리
                                totalDragX += dragAmount.x
                            }
                        )
                    }
            ) {
                if (isVideo && currentMediaUri != null) {
                    // 동영상 재생 - LaunchedEffect를 사용하여 URI 변경 시 VideoView 업데이트
                    var videoViewRef by remember { mutableStateOf<VideoView?>(null) }
                    
                    LaunchedEffect(currentMediaUri) {
                        videoViewRef?.let { view ->
                            view.stopPlayback()
                            view.setVideoURI(currentMediaUri)
                            val mediaController = MediaController(context)
                            mediaController.setAnchorView(view)
                            view.setMediaController(mediaController)
                            view.start()
                        }
                    }
                    
                    AndroidView(
                        factory = { ctx ->
                            VideoView(ctx).apply {
                                setVideoURI(currentMediaUri)
                                val mediaController = MediaController(ctx)
                                mediaController.setAnchorView(this)
                                setMediaController(mediaController)
                                start()
                                videoViewRef = this
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                        update = { view ->
                            videoViewRef = view
                        }
                    )
                } else {
                    // 이미지 보기
                    Image(
                        painter = rememberAsyncImagePainter(currentMediaUri),
                        contentDescription = "미디어 상세",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit
                    )
                }
            }
        }

        // 페이지 인디케이터 (하단 중앙)
        if (mediaList.size > 1) {
            Text(
                text = "${currentIndex + 1} / ${mediaList.size}",
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(16.dp))
                    .padding(horizontal = 16.dp, vertical = 8.dp)
                    .padding(bottom = 16.dp)
            )
        }
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

private fun captureDatasetImage(
    context: Context,
    sectorIndex: Int,
    pitchAngle: Int,
    dir: File,
    capture: ImageCapture
) {
    try {
        if (!dir.exists()) {
            dir.mkdirs()
        }
        val file = File(dir, "${pitchAngle}_${sectorIndex + 1}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exception: ImageCaptureException) {
                    exception.printStackTrace()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    // 저장 완료
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
    val root = File(context.getExternalFilesDir(null), "datasets")
    if (!root.exists()) {
        onLoaded(emptyList())
        return
    }

    val folders = root.listFiles { file -> file.isDirectory }?.map { dir ->
        val images = dir.listFiles { f ->
            f.isFile && f.name.endsWith(".jpg", ignoreCase = true)
        }?.sortedBy { it.nameWithoutExtension.toIntOrNull() ?: Int.MAX_VALUE } ?: emptyList()

        val cover = images.firstOrNull()?.let { Uri.fromFile(it) }
        DatasetFolder(
            name = dir.name,
            dir = dir,
            coverUri = cover,
            count = images.size
        )
    }?.sortedByDescending { it.name } ?: emptyList()

    onLoaded(folders)
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
private suspend fun startServerTaskWithZip(context: Context, zipFile: File): String? {
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

            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    zipFile.name,
                    zipFile.asRequestBody("application/zip".toMediaType())
                )
                .build()

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
    onProgress: (progress: Int, message: String) -> Unit
): Boolean {
    // 1) 업로드 -> task_id 확보
    onProgress(5, "파일 업로드 중...")
    val noResponseMsg = "서버에 대한 응답이 없습니다.\n서버 연결을 확인해주십시오."
    val taskId = startServerTaskWithZip(context, zipFile)
    if (taskId.isNullOrBlank()) {
        onProgress(0, noResponseMsg)
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
            onProgress(st.progressPercent.coerceIn(0, 100), st.message)
            when (st.status) {
                "COMPLETED" -> break
                "FAILED" -> return false
            }
        } else {
            // 60초 이상 서버 응답이 없으면 중단
            if (System.currentTimeMillis() - lastServerResponseAt >= 60_000L) {
                onProgress(0, noResponseMsg)
                return false
            }
        }
        delay(1000)
    }

    // 3) 결과 다운로드 (.ply)
    onProgress(95, "결과 다운로드 중...")
    val result = downloadPlyResult(context, taskId) ?: return false
    onProgress(100, "완료되었습니다! 저장 위치: ${result.name}")
    return true
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

private data class PlyParseResult(
    val points: FloatArray,
    val count: Int
)

private fun parsePlyPoints(file: File): PlyParseResult? {
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