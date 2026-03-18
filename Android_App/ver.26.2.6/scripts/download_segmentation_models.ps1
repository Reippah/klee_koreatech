# MediaPipe / 세그멘테이션 모델 다운로드 (Windows PowerShell)
# SAM2 다운로드 실패 시 대체용 - DeepLab v3 등

$assetsPath = "$PSScriptRoot\..\app\src\main\assets\models"
$modelsPath = "$assetsPath\segmentation"

if (-not (Test-Path $assetsPath)) {
    New-Item -ItemType Directory -Path $assetsPath -Force
}
if (-not (Test-Path $modelsPath)) {
    New-Item -ItemType Directory -Path $modelsPath -Force
}

# MediaPipe DeepLab v3 (세그멘테이션 - Pascal VOC 21 클래스)
$deeplabUrl = "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
$deeplabPath = "$modelsPath\deeplab_v3.tflite"

Write-Host "다운로드: DeepLab v3 (세그멘테이션)"
try {
    Invoke-WebRequest -Uri $deeplabUrl -OutFile $deeplabPath -UseBasicParsing
    Write-Host "저장됨: $deeplabPath"
} catch {
    Write-Host "다운로드 실패: $_"
}

Write-Host "`n완료."
