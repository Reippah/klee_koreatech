import os
import shutil
import logging
import uuid
import time
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException

# 전처리 모듈 임포트
from preprocessor import process_zip_task

# VGGT 및 모델 라이브러리 (파일이 있다고 가정)
try:
    from viser_and_colmap_based_demo import VGGTJetsonIntegrated
except ImportError:
    # 테스트용 모킹(Mocking) - 실제 환경에선 제거하세요
    class VGGTJetsonIntegrated:
        def __init__(self, **kwargs): pass
        def process_scene(self, **kwargs): return {}
        def save_to_ply(self, *args): pass

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VideoPipeline")

app = FastAPI()

# --- 환경 및 경로 설정 ---
BASE_DATA_DIR = "/mnt/sdcard/klee_koreatech/init"
MODEL_PATH = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

# --- VGGT 파이프라인 전역 초기화 ---
logger.info("VGGT 모델 로드 중...")
vggt_processor = VGGTJetsonIntegrated(model_url=MODEL_PATH, device="cuda")
logger.info("VGGT 모델 로드 완료.")

# 작업 상태 저장소
task_db: Dict[str, Dict] = {}

def run_full_pipeline(task_id: str, file_path: str):
    pipeline_start_time = time.time()
    
    try:
        # [Step 1] 전처리 단계 (ZIP 내 이미지 추출 및 보정)
        task_db[task_id]["status"] = "STEP1_PREPROCESSING"
        logger.info(f"[{task_id}] 단계 1: 전처리 시작")
        
        step1_start = time.time()
        
        # preprocessor.py의 파라미터에 맞춰 수정 (interval_seconds 삭제)
        # target_size는 VGGT(518x518)에 최적화되어 있습니다.
        processed_folder = process_zip_task(
            file_path=file_path, 
            task_id=task_id, 
            target_size=(518, 518), 
            blur_threshold=100.0
        )
        
        step1_end = time.time()
        step1_duration = round(step1_end - step1_start, 2)
        task_db[task_id]["durations"]["step1_preprocessing"] = step1_duration
        logger.info(f"[{task_id}] 단계 1 완료: {step1_duration}초 소요")

        # [Step 2] VGGT 분석 단계
        task_db[task_id]["status"] = "STEP2_VGGT"
        logger.info(f"[{task_id}] 단계 2: VGGT 분석 및 3D 복원 중...")
        
        step2_start = time.time()
        
        # 전처리된 폴더의 이미지를 사용하여 3D 복원 실행
        vggt_results = vggt_processor.process_scene(
            image_folder=processed_folder, 
            use_ba=False, 
            mask_sky=True
        )
        
        output_ply_path = os.path.join(processed_folder, "reconstruction.ply")
        vggt_processor.save_to_ply(vggt_results, output_ply_path)
        
        step2_end = time.time()
        step2_duration = round(step2_end - step2_start, 2)
        task_db[task_id]["durations"]["step2_vggt"] = step2_duration
        task_db[task_id]["ply_path"] = output_ply_path
        logger.info(f"[{task_id}] 단계 2 완료: {step2_duration}초 소요")

        # [Step 3] 완료 처리
        task_db[task_id]["status"] = "COMPLETED"
        total_duration = round(time.time() - pipeline_start_time, 2)
        task_db[task_id]["durations"]["total_pipeline"] = total_duration
        logger.info(f"[{task_id}] 파이프라인 종료. 총 소요시간: {total_duration}초")

    except Exception as e:
        task_db[task_id]["status"] = "FAILED"
        task_db[task_id]["error"] = str(e)
        logger.error(f"[{task_id}] 오류 발생: {e}")
    finally:
        # 원본 업로드 파일 삭제 (결과 폴더는 유지)
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload")
async def upload_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 파일 확장자 체크 (현재 preprocessor는 .zip만 지원)
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="현재는 이미지들을 압축한 .zip 파일만 지원합니다.")

    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{task_id}{file_extension}"
    
    upload_dir = os.path.join(BASE_DATA_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    task_db[task_id] = {
        "status": "PENDING",
        "original_filename": file.filename,
        "task_id": task_id,
        "durations": {"step1_preprocessing": 0, "step2_vggt": 0, "total_pipeline": 0},
        "ply_path": None
    }
    
    background_tasks.add_task(run_full_pipeline, task_id, file_path)
    
    return {
        "message": "데이터 업로드 완료. 전처리 및 VGGT 분석을 시작합니다.",
        "task_id": task_id
    }

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    status_info = task_db.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="해당 작업 ID를 찾을 수 없습니다.")
    return status_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)