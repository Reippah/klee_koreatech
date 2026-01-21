import os
import shutil
import logging
import uuid
import time
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException

# 전처리 모듈 (기존 유지)
from preprocessor import process_video_task

# viser_and_colmap_based_demo.py 내의 클래스 및 필요한 라이브러리 임포트
# (파일이 같은 경로에 있다고 가정하거나 아래에 클래스를 정의해야 합니다)
from viser_and_colmap_based_demo import VGGTJetsonIntegrated

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
# 앱 시작 시 모델을 한 번만 로드하여 메모리 효율을 높입니다.
logger.info("VGGT 모델 로드 중...")
vggt_processor = VGGTJetsonIntegrated(model_url=MODEL_PATH, device="cuda")
logger.info("VGGT 모델 로드 완료.")

# 작업 상태 및 소요 시간을 저장할 메모리 DB
task_db: Dict[str, Dict] = {}

def run_full_pipeline(task_id: str, file_path: str, original_filename: str):
    pipeline_start_time = time.time()
    
    try:
        # [Step 1] 전처리 단계 (비디오 -> 프레임 추출)
        task_db[task_id]["status"] = "STEP1_PREPROCESSING"
        logger.info(f"[{task_id}] 단계 1: 전처리 시작")
        
        step1_start = time.time()
        # preprocessor.py의 함수 실행
        processed_folder = process_video_task(file_path, task_id, interval_seconds=0.5)
        step1_end = time.time()
        
        step1_duration = round(step1_end - step1_start, 2)
        task_db[task_id]["durations"]["step1_preprocessing"] = step1_duration
        logger.info(f"[{task_id}] 단계 1 완료: {step1_duration}초 소요")

        # [Step 2] VGGT 분석 단계 (기존 subprocess 대신 직접 클래스 호출)
        task_db[task_id]["status"] = "STEP2_VGGT"
        logger.info(f"[{task_id}] 단계 2: VGGT 분석 및 3D 복원 중...")
        
        step2_start = time.time()
        
        # viser_and_colmap_based_demo의 핵심 로직 실행
        # use_ba(Bundle Adjustment)는 Jetson 메모리 상황에 따라 False 권장
        vggt_results = vggt_processor.process_scene(
            image_folder=processed_folder, 
            use_ba=False, 
            mask_sky=True
        )
        
        # 결과 저장 (.ply 파일 생성)
        output_ply_path = os.path.join(processed_folder, "reconstruction.ply")
        vggt_processor.save_to_ply(vggt_results, output_ply_path)
        
        step2_end = time.time()
        step2_duration = round(step2_end - step2_start, 2)
        task_db[task_id]["durations"]["step2_vggt"] = step2_duration
        task_db[task_id]["ply_path"] = output_ply_path # 결과 경로 저장
        logger.info(f"[{task_id}] 단계 2 완료: {step2_duration}초 소요")

        # [Step 3] 3DGS 실행 단계 (필요 시 확장)
        task_db[task_id]["status"] = "STEP3_3DGS"
        logger.info(f"[{task_id}] 단계 3: 3DGS 최적화 (준비 중)")
        
        # TODO: 3DGS 학습 코드 통합 가능

        # 전체 완료 처리
        task_db[task_id]["status"] = "COMPLETED"
        total_duration = round(time.time() - pipeline_start_time, 2)
        task_db[task_id]["durations"]["total_pipeline"] = total_duration
        logger.info(f"[{task_id}] 파이프라인 종료. 총 소요시간: {total_duration}초")

    except Exception as e:
        task_db[task_id]["status"] = "FAILED"
        task_db[task_id]["error"] = str(e)
        logger.error(f"[{task_id}] 오류 발생: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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
        "durations": {
            "step1_preprocessing": 0,
            "step2_vggt": 0,
            "step3_3dgs": 0,
            "total_pipeline": 0
        },
        "ply_path": None
    }
    
    background_tasks.add_task(run_full_pipeline, task_id, file_path, file.filename)
    
    return {
        "message": "업로드 성공, VGGT 분석을 시작합니다.",
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
    # Jetson 외부 접속을 위해 host="0.0.0.0" 유지
    uvicorn.run(app, host="0.0.0.0", port=8000)