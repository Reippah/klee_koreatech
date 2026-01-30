import os
import shutil
import logging
import uuid
import time
import threading
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException

# 전처리 및 모델 처리 모듈 임포트
from preprocessor import process_zip_task

# VGGT 모델 클래스 임포트 (ImportError 발생 시 Mock 객체 사용)
try:
    from viser_and_colmap_based_demo import VGGTJetsonIntegrated
except ImportError:
    # 개발 및 테스트 환경을 위한 Mock 클래스 정의
    class VGGTJetsonIntegrated:
        def __init__(self, **kwargs): pass
        def process_scene(self, **kwargs): 
            time.sleep(3) # GPU 처리 시간 시뮬레이션
            return {}
        def save_to_ply(self, *args): pass

# 로깅 설정: 파이프라인 진행 상황을 파일과 콘솔에 기록
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

# 기본 데이터 저장 경로 및 모델 URL 설정
BASE_DATA_DIR = "/mnt/sdcard/klee_koreatech/init"
MODEL_PATH = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

# 애플리케이션 시작 시 VGGT 모델을 메모리에 로드 (콜드 스타트 방지)
logger.info("VGGT 모델 초기화 중...")
vggt_processor = VGGTJetsonIntegrated(model_url=MODEL_PATH, device="cuda")
logger.info("VGGT 모델 준비 완료.")

# 다중 요청 시 GPU 동시 접근을 막기 위한 뮤텍스(Mutex) 락
gpu_lock = threading.Lock()

# 비동기 작업 상태를 관리하는 인메모리 저장소
task_db: Dict[str, Dict] = {}

def run_full_pipeline(task_id: str, file_path: str):
    """
    백그라운드에서 실행되는 전체 파이프라인 함수입니다.
    CPU 작업(전처리)은 병렬로 처리하고, GPU 작업(VGGT 추론)은 직렬로 처리합니다.
    """
    pipeline_start_time = time.time()
    
    try:
        # 1단계: CPU 기반 전처리 (병렬 실행 가능)
        # 압축 해제, 이미지 리사이징, 선명도 개선 작업을 수행합니다.
        task_db[task_id]["status"] = "STEP1_PREPROCESSING"
        logger.info(f"[{task_id}] 전처리 단계 시작")
        
        step1_start = time.time()
        
        processed_folder = process_zip_task(
            file_path=file_path, 
            task_id=task_id, 
            target_size=(518, 518), 
            blur_threshold=100.0
        )
        
        step1_end = time.time()
        step1_duration = round(step1_end - step1_start, 2)
        task_db[task_id]["durations"]["step1_preprocessing"] = step1_duration
        logger.info(f"[{task_id}] 전처리 완료 (소요시간: {step1_duration}초)")

        # 2단계: GPU 기반 3D 복원 (임계 영역)
        # GPU 메모리 부족을 방지하기 위해 락을 사용하여 순차적으로 실행합니다.
        task_db[task_id]["status"] = "WAITING_FOR_GPU"
        logger.info(f"[{task_id}] GPU 자원 대기 중...")

        with gpu_lock:
            task_db[task_id]["status"] = "STEP2_VGGT"
            logger.info(f"[{task_id}] GPU 락 획득, VGGT 추론 시작")
            
            step2_start = time.time()
            
            # 전처리된 이미지를 기반으로 3D 포인트 클라우드 생성
            vggt_results = vggt_processor.process_scene(
                image_folder=processed_folder, 
                use_ba=False, 
                mask_sky=True
            )
            
            # 결과 PLY 파일 저장
            output_ply_path = os.path.join(processed_folder, "reconstruction.ply")
            vggt_processor.save_to_ply(vggt_results, output_ply_path)
            
            step2_end = time.time()
            step2_duration = round(step2_end - step2_start, 2)
            task_db[task_id]["durations"]["step2_vggt"] = step2_duration
            task_db[task_id]["ply_path"] = output_ply_path
            logger.info(f"[{task_id}] 추론 완료 (소요시간: {step2_duration}초)")

        # 3단계: 작업 완료 처리 및 메타데이터 업데이트
        task_db[task_id]["status"] = "COMPLETED"
        total_duration = round(time.time() - pipeline_start_time, 2)
        task_db[task_id]["durations"]["total_pipeline"] = total_duration
        logger.info(f"[{task_id}] 파이프라인 정상 종료 (총 소요시간: {total_duration}초)")

    except Exception as e:
        # 예외 발생 시 상태를 실패로 변경하고 에러 메시지 기록
        task_db[task_id]["status"] = "FAILED"
        task_db[task_id]["error"] = str(e)
        logger.error(f"[{task_id}] 파이프라인 에러: {e}")
    finally:
        # 공간 절약을 위해 업로드된 원본 압축 파일 삭제 (결과는 유지)
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload")
async def upload_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    클라이언트로부터 ZIP 파일을 업로드 받아 백그라운드 작업을 시작하는 엔드포인트입니다.
    """
    # 파일 확장자 검증
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다. .zip 파일만 허용됩니다.")

    # 고유 작업 ID 생성 및 파일 저장 경로 설정
    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{task_id}{file_extension}"
    
    upload_dir = os.path.join(BASE_DATA_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, safe_filename)
    
    # 파일 디스크에 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 작업 상태 초기화
    task_db[task_id] = {
        "status": "PENDING",
        "original_filename": file.filename,
        "task_id": task_id,
        "durations": {"step1_preprocessing": 0, "step2_vggt": 0, "total_pipeline": 0},
        "ply_path": None
    }
    
    # 백그라운드 작업 큐에 파이프라인 함수 등록
    background_tasks.add_task(run_full_pipeline, task_id, file_path)
    
    return {
        "message": "파일이 업로드되었습니다. 백그라운드 처리가 시작됩니다.",
        "task_id": task_id
    }

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    작업 ID를 통해 현재 진행 상황을 조회하는 엔드포인트입니다.
    """
    status_info = task_db.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="존재하지 않는 작업 ID입니다.")
    return status_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)