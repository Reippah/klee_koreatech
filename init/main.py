import os
import shutil
import logging
import uuid
import time
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
import subprocess
from preprocessor import process_zip_task

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
)
logger = logging.getLogger("ZipImagePipeline")

app = FastAPI()

# 경로 설정
PY_VGGT = "/mnt/sdcard/venv/vggt/bin/python"
PATH_VGGT_DIR = "/mnt/sdcard/klee_koreatech/vggt"
PATH_VGGT_SCRIPT = "/mnt/sdcard/klee_koreatech/vggt/vggt_to_COLMAP.py"
BASE_DATA_DIR = "/mnt/sdcard/klee_koreatech/init"

task_db: Dict[str, Dict] = {}

def run_full_pipeline(task_id: str, file_path: str, original_filename: str):
    pipeline_start_time = time.time()
    
    try:
        # [Step 1] 전처리 단계 (ZIP 해제 및 리사이징)
        task_db[task_id]["status"] = "STEP1_PREPROCESSING"
        logger.info(f"[{task_id}] 단계 1: ZIP 이미지 전처리 시작")
        
        step1_start = time.time()
        # 비디오 대신 ZIP 처리 함수 호출
        processed_folder = process_zip_task(file_path, task_id, target_size=(518, 518))
        step1_end = time.time()
        
        if not processed_folder:
            raise Exception("이미지 전처리 실패")

        step1_duration = round(step1_end - step1_start, 2)
        task_db[task_id]["durations"]["step1_preprocessing"] = step1_duration
        logger.info(f"[{task_id}] 단계 1 완료: {step1_duration}초")

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
            logger.info(f"[{task_id}] 단계 2 완료: {step2_duration}초")
        else:
            raise FileNotFoundError(f"VGGT 스크립트 없음: {PATH_VGGT_SCRIPT}")

        # [Step 3] 3DGS 실행 단계
        task_db[task_id]["status"] = "STEP3_3DGS"
        # 3DGS 로직 추가 시 여기에...
        
        task_db[task_id]["status"] = "COMPLETED"
        total_duration = round(time.time() - pipeline_start_time, 2)
        task_db[task_id]["durations"]["total_pipeline"] = total_duration

    except Exception as e:
        task_db[task_id]["status"] = "FAILED"
        task_db[task_id]["error"] = str(e)
        logger.error(f"[{task_id}] 오류: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload-zip")
async def upload_zip(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 확장자 체크
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="ZIP 파일만 업로드 가능합니다.")

    task_id = str(uuid.uuid4())
    safe_filename = f"{task_id}.zip"
    
    upload_dir = os.path.join(BASE_DATA_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    task_db[task_id] = {
        "status": "PENDING",
        "original_filename": file.filename,
        "task_id": task_id,
        "durations": {"step1_preprocessing": 0, "step2_vggt": 0, "step3_3dgs": 0, "total_pipeline": 0}
    }
    
    background_tasks.add_task(run_full_pipeline, task_id, file_path, file.filename)
    
    return {"message": "ZIP 파일 업로드 성공, 분석 시작", "task_id": task_id}

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    status_info = task_db.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="ID를 찾을 수 없음")
    return status_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
