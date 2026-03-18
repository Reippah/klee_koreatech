# 모델 다운로드 스크립트

## 배경 제거 (세그멘테이션)

### 1. MediaPipe DeepLab v3 (권장, 앱에서 사용)

```powershell
.\download_segmentation_models.ps1
```

- `app/src/main/assets/models/segmentation/deeplab_v3.tflite` 에 저장
- Pascal VOC 21 클래스 지원 (사람, 병, 자동차 등)
- 배경 제거 버튼 클릭 → "무엇을 모델링할까요?" 입력 → 해당 객체만 남기고 배경 제거

### 2. SAM2 Hiera-Tiny (Qualcomm, Android TFLite)

```bash
py scripts/download_sam2_models.py
```

- Qualcomm SAM2 TFLite (w8a8) 다운로드
- `app/src/main/assets/models/sam2/` 에 저장 (SAM2Encoder.tflite, SAM2Decoder.tflite)
- Snapdragon NPU 가속 지원 (QNN)

### 3. SAM3 (Meta, Hugging Face - PyTorch)

```bash
# 1) Hugging Face 접근 권한 요청: https://huggingface.co/facebook/sam3
# 2) 인증: huggingface-cli login
# 3) 다운로드
py scripts/download_sam3_model.py
```

- `models/sam3/` 에 저장 (config, tokenizer 등)
- 전체 모델(~6.5GB): `SAM3_DOWNLOAD_FULL=1 py scripts/download_sam3_model.py`
- ※ SAM3는 PyTorch 모델로 Android 앱에서 직접 사용 불가. 서버/참조용.

### 4. SAM2 배경 제거 (PC용 Python 스크립트)

GitHub SAM2 (`C:\Users\dongh\Downloads\sam2-main`) API를 사용해 원하는 사물만 남기고 배경 제거.

```bash
# 1) SAM2 설치 (로컬 레포)
cd C:\Users\dongh\Downloads\sam2-main\sam2-main
pip install -e ".[notebooks]"

# 2) 배경 제거 실행 (HuggingFace 모델 사용)
py scripts/sam2_background_removal.py -i photo.jpg -p person -o result.png
```

| 옵션 | 설명 |
|------|------|
| `-i`, `--image` | 입력 이미지 경로 |
| `-o`, `--output` | 출력 경로 (기본: 입력파일명_bg_removed.png) |
| `-p`, `--prompt` | 남길 사물 (person, car, bottle 등) |

※ PyTorch, torchvision, CUDA 권장. PC에서 실행하는 스크립트이며 Android 앱과 별개.

### 5. SAM3 배경 제거 (PC용 Python 스크립트)

GitHub SAM3 (`C:\Users\dongh\Downloads\sam3-main`) API를 사용해 텍스트 프롬프트 기반 배경 제거. SAM2보다 정확한 세그멘테이션.

```bash
# 1) SAM3 설치 (로컬 레포)
cd C:\Users\dongh\Downloads\sam3-main
pip install -e ".[notebooks]"

# 2) Hugging Face 접근 (facebook/sam3 gated)
huggingface-cli login

# 3) 배경 제거 실행
py scripts/sam3_background_removal.py -i photo.jpg -p "person" -o result.png
py scripts/sam3_background_removal.py -i photo.jpg -p "yellow car" -o result.png
```

| 옵션 | 설명 |
|------|------|
| `-i`, `--image` | 입력 이미지 경로 |
| `-o`, `--output` | 출력 경로 |
| `-p`, `--prompt` | 남길 사물 (텍스트, 예: "person", "yellow car") |

※ SAM3는 PyTorch 전용. Android 앱에서는 SAM2 TFLite + NNAPI 가속 사용.

## 사용 가능한 텍스트 프롬프트 예시

| 한글 | 영어 |
|------|------|
| 사람, 인물 | person |
| 병, 캔, 컵 | bottle, can, cup |
| 자동차, 차 | car |
| 의자 | chair |
| 개 | dog |
| 고양이 | cat |
| 식물, 화분 | potted plant |
