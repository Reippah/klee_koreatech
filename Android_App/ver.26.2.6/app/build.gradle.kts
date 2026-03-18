import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.app_01"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        applicationId = "com.example.app_01"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // API 키: local.properties에 claude_api_key, gemini_api_key 추가
        val localProperties = Properties()
        val localPropertiesFile = rootProject.file("local.properties")
        if (localPropertiesFile.exists()) {
            localProperties.load(localPropertiesFile.inputStream())
        }
        buildConfigField(
            "String",
            "CLAUDE_API_KEY",
            "\"${localProperties.getProperty("claude_api_key", "")}\""
        )
        buildConfigField(
            "String",
            "GEMINI_API_KEY",
            "\"${localProperties.getProperty("gemini_api_key", "")}\""
        )
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    // 모델 파일 압축 방지 (assets에서 직접 로드)
    androidResources {
        noCompress += listOf("tflite", "onnx")
    }
    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    
    // CameraX (16KB 페이지 크기 호환성 개선 포함)
    val cameraxVersion = "1.4.0"
    implementation("androidx.camera:camera-core:${cameraxVersion}")
    implementation("androidx.camera:camera-camera2:${cameraxVersion}")
    implementation("androidx.camera:camera-lifecycle:${cameraxVersion}")
    implementation("androidx.camera:camera-view:${cameraxVersion}")
    implementation("androidx.camera:camera-video:${cameraxVersion}")
    
    // Coil for image loading
    implementation("io.coil-kt:coil-compose:2.5.0")
    
    // Material Icons Extended
    implementation("androidx.compose.material:material-icons-extended")
    
    // OkHttp for HTTP requests
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // EXIF (이미지 회전/방향 보정)
    implementation("androidx.exifinterface:exifinterface:1.3.7")

    // (삭제) OpenCV: 광택/반사 제거 옵션 제거에 따라 의존성 제거

    // MediaPipe Tasks (사물 경계/크기 분석 - ObjectDetector, Interactive Segmenter)
    implementation("com.google.mediapipe:tasks-vision:0.10.14")

    // ML Kit 제거: 네이티브 GPU/드라이버 비호환 크래시 문제로 MediaPipe InteractiveSegmenter로 교체
    // InteractiveSegmenter는 tasks-vision에 포함되어 있음 (별도 의존성 불필요)

    // ONNX Runtime Android — U²-Net(u2netp) 카테고리 제한 없는 배경 제거
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}