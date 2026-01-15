import os
import shutil
import subprocess
import logging
import uuid
import time  # 시간 측정을 위해 추가
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from preprocessor import process_video_task

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
PY_VGGT = "/mnt/sdcard/venv/vggt/bin/python"
PATH_VGGT_DIR = "/mnt/sdcard/klee_koreatech/vggt"
PATH_VGGT_SCRIPT = "/mnt/sdcard/klee_koreatech/vggt/vggt_to_COLMAP.py"
BASE_DATA_DIR = "/mnt/sdcard/klee_koreatech/init"

# 작업 상태 및 소요 시간을 저장할 메모리 DB
task_db: Dict[str, Dict] = {}

def run_full_pipeline(task_id: str, file_path: str, original_filename: str):
    # 환경 변수 복사 및 설정
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    processed_folder = os.path.join(BASE_DATA_DIR, "result", f"processed_{task_id}")
    pipeline_start_time = time.time() # 전체 시작 시간
    
    try:
        # [Step 1] 전처리 단계
        task_db[task_id]["status"] = "STEP1_PREPROCESSING"
        logger.info(f"[{task_id}] 단계 1: 전처리 시작")
        
        step1_start = time.time()
        processed_folder = process_video_task(file_path, task_id, interval_seconds=0.5)
        step1_end = time.time()
        
        # 소요 시간 계산 및 저장 (소수점 둘째자리까지)
        step1_duration = round(step1_end - step1_start, 2)
        task_db[task_id]["durations"]["step1_preprocessing"] = step1_duration
        logger.info(f"[{task_id}] 단계 1 완료: {step1_duration}초 소요")

        # [Step 2] VGGT 분석 단계
        task_db[task_id]["status"] = "STEP2_VGGT"
        if os.path.exists(PATH_VGGT_SCRIPT):
            logger.info(f"[{task_id}] 단계 2: VGGT 분석 중...")
            
            step2_start = time.time()
            subprocess.run(
                [PY_VGGT, PATH_VGGT_SCRIPT, "--image_folder", processed_folder],
                cwd=PATH_VGGT_DIR,
                check=True,
                timeout=3600 
            )
            step2_end = time.time()
            
            step2_duration = round(step2_end - step2_start, 2)
            task_db[task_id]["durations"]["step2_vggt"] = step2_duration
            logger.info(f"[{task_id}] 단계 2 완료: {step2_duration}초 소요")
        else:
            raise FileNotFoundError(f"VGGT 스크립트 경로 없음: {PATH_VGGT_SCRIPT}")

        # [Step 3] 3DGS 실행 단계 (필요 시 주석 해제)
        task_db[task_id]["status"] = "STEP3_3DGS"
        logger.info(f"[{task_id}] 단계 3: 3DGS 최적화 중...")
        
        step3_start = time.time()

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
    
    # 초기 DB 데이터에 durations 항목 추가
    task_db[task_id] = {
        "status": "PENDING",
        "original_filename": file.filename,
        "task_id": task_id,
        "durations": {
            "step1_preprocessing": 0,
            "step2_vggt": 0,
            "step3_3dgs": 0,
            "total_pipeline": 0
        }
    }
    
    background_tasks.add_task(run_full_pipeline, task_id, file_path, file.filename)
    
    return {
        "message": "업로드 성공, 분석을 시작합니다.",
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
