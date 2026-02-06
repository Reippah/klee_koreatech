import torch
import gc
import os
import shutil
import logging
import uuid
import time
import threading
from typing import Dict, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

# =============================================================================
# 1. 환경 설정 및 모듈 로딩 (Mocking)
# =============================================================================

# 실제 딥러닝/전처리 모듈이 없는 개발 환경에서도 서버가 죽지 않고 돌아가도록
# 예외 처리를 통해 가짜(Mock) 함수와 클래스를 정의합니다.
try:
    from preprocessor import process_zip_task
except ImportError:
    # 전처리 모듈 로드 실패 시 대체 함수 정의
    def process_zip_task(file_path, task_id, target_size, blur_threshold):
        # 처리 시간 시뮬레이션 (2초 대기)
        time.sleep(2)
        # 결과 파일이 저장될 가짜 경로 생성
        output_dir = os.path.join(os.path.dirname(file_path), f"processed_{task_id}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

try:
    from viser_and_colmap_based_demo import VGGTJetsonIntegrated
except ImportError:
    # 3D 모델링 모듈 로드 실패 시 대체 클래스 정의
    class VGGTJetsonIntegrated:
        def __init__(self, **kwargs): pass
        
        def process_scene(self, **kwargs): 
            # GPU 연산 시간 시뮬레이션 (5초 대기)
            time.sleep(5) 
            return {"points": []}
            
        def save_to_ply(self, results, path): 
            # 빈 결과 파일 생성
            with open(path, 'w') as f: f.write("ply header...")

# =============================================================================
# 2. 로깅 및 앱 초기화
# =============================================================================

# 서버 로그 설정: 파일(pipeline.log)과 콘솔 모두에 로그를 출력하도록 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VideoPipeline")

# FastAPI 애플리케이션 객체 생성
app = FastAPI()

# 파일 저장 기본 경로 설정
BASE_DATA_DIR = "/home/viclab/klee_koreatech/init"
# HuggingFace 모델 다운로드 경로
MODEL_PATH = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

# =============================================================================
# 3. 모델 전역 로드 및 동시성 제어
# =============================================================================

logger.info("VGGT 모델 초기화 중...")
try:
    # 서버 시작 시 모델을 메모리에 미리 로드합니다. (요청마다 로드하면 느림)
    # device="cuda"로 설정하여 GPU 메모리에 올립니다.
    vggt_processor = VGGTJetsonIntegrated(model_url=MODEL_PATH, device="cuda")
    logger.info("VGGT 모델 준비 완료.")
except Exception as e:
    # 로컬 테스트 등 GPU가 없거나 모델 로드 실패 시 로그를 남기고 가짜 객체 사용
    logger.error(f"모델 로드 실패 (Mock 모드라면 무시): {e}")
    vggt_processor = VGGTJetsonIntegrated() 

# GPU 동시 접근 방지를 위한 Mutex Lock
# 여러 요청이 와도 GPU를 사용하는 무거운 작업은 한 번에 하나만 수행하게 합니다.
gpu_lock = threading.Lock()

# =============================================================================
# 4. 작업 상태 관리 (In-Memory DB)
# =============================================================================

# 각 작업(Task)의 상태를 저장하는 딕셔너리
# Key: task_id, Value: 작업 정보
task_db: Dict[str, Dict] = {}

# 각 단계별 진행률 매핑 테이블
STATUS_PROGRESS = {
    "PENDING": 5,              # 업로드 직후
    "STEP1_PREPROCESSING": 20, # CPU 전처리 중
    "WAITING_FOR_GPU": 50,     # 전처리 끝, GPU 대기 중
    "STEP2_VGGT": 70,          # GPU 추론 중
    "SAVING": 90,              # 결과 저장 중
    "COMPLETED": 100,          # 완료
    "FAILED": 0                # 실패
}

def _get_user_friendly_message(status: str) -> str:
    """내부 상태 코드를 사용자용 안내 메시지로 변환합니다."""
    messages = {
        "PENDING": "작업 대기 중입니다...",
        "STEP1_PREPROCESSING": "이미지를 분석하고 있습니다...",
        "WAITING_FOR_GPU": "다른 작업이 끝나기를 기다리고 있습니다 (GPU 대기)...",
        "STEP2_VGGT": "3D 모델을 생성하는 중입니다 (시간이 소요됩니다)...",
        "SAVING": "결과 파일을 저장하고 있습니다...",
        "COMPLETED": "완료되었습니다! 다운로드 버튼을 눌러주세요.",
        "FAILED": "작업 처리 중 오류가 발생했습니다."
    }
    return messages.get(status, "처리 중...")

# =============================================================================
# 5. 핵심 파이프라인 로직 (백그라운드 실행)
# =============================================================================

def run_full_pipeline(task_id: str, file_path: str):
    """
    업로드된 파일에 대해 전체 3D 복원 과정을 수행하는 함수입니다.
    FastAPI의 BackgroundTasks에 의해 별도 스레드에서 비동기로 실행됩니다.
    """
    pipeline_start_time = time.time()
    
    # 상태 업데이트 헬퍼 함수
    def update_status(status_code: str):
        task_db[task_id]["status"] = status_code
        task_db[task_id]["progress"] = STATUS_PROGRESS.get(status_code, 0)
        task_db[task_id]["message"] = _get_user_friendly_message(status_code)

    try:
        # --- [Step 1] CPU 전처리 ---
        # 이미지 크기 조절, 블러 처리 등을 수행합니다.
        # 이 단계는 GPU를 쓰지 않으므로 락(Lock) 없이 진행 가능합니다.
        update_status("STEP1_PREPROCESSING")
        logger.info(f"[{task_id}] 전처리 단계 시작")
        
        step1_start = time.time()
        
        processed_folder = process_zip_task(
            file_path=file_path, 
            task_id=task_id, 
            target_size=(518, 518), 
            blur_threshold=30.0
        )
        
        step1_end = time.time()
        # 소요 시간 기록
        task_db[task_id]["durations"]["step1_preprocessing"] = round(step1_end - step1_start, 2)
        logger.info(f"[{task_id}] 전처리 완료")

        # --- [Step 2] GPU 추론 (Critical Section) ---
        # 3D 모델 생성은 GPU 메모리를 많이 사용하므로, 한 번에 하나의 작업만 수행해야 합니다.
        update_status("WAITING_FOR_GPU")
        logger.info(f"[{task_id}] GPU 자원 대기 중...")

        # 여기서 Lock을 획득할 때까지 대기합니다.
        with gpu_lock:
            update_status("STEP2_VGGT")
            logger.info(f"[{task_id}] GPU 락 획득, VGGT 추론 시작")
            
            step2_start = time.time()
            vggt_results = None

            try:
                # 메모리 절약을 위해 그라디언트 계산 비활성화 (Inference Mode)
                with torch.inference_mode(): 
                    vggt_results = vggt_processor.process_scene(
                        image_folder=processed_folder, 
                        use_ba=False, 
                        mask_sky=True
                    )
                    
                    # 결과 파일 경로 설정
                    update_status("SAVING")
                    output_dir = os.path.dirname(processed_folder)
                    output_ply_name = f"reconstruction_{task_id}.ply"
                    output_ply_path = os.path.join(output_dir, output_ply_name)
                    
                    # .ply 파일로 저장
                    vggt_processor.save_to_ply(vggt_results, output_ply_path)
            
            finally:
                # --- 메모리 정리 (중요) ---
                # 작업이 끝나거나 에러가 나더라도 반드시 GPU 메모리를 비워야 합니다.
                if vggt_results:
                    del vggt_results
                gc.collect() # Python 가비지 컬렉션
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # PyTorch GPU 캐시 비우기

            step2_end = time.time()
            task_db[task_id]["durations"]["step2_vggt"] = round(step2_end - step2_start, 2)
            task_db[task_id]["ply_path"] = output_ply_path
            logger.info(f"[{task_id}] 추론 및 저장 완료")

        # --- [Step 3] 완료 처리 ---
        total_duration = round(time.time() - pipeline_start_time, 2)
        task_db[task_id]["durations"]["total_pipeline"] = total_duration
        update_status("COMPLETED")
        logger.info(f"[{task_id}] 파이프라인 정상 종료 (총 {total_duration}초)")

    except Exception as e:
        # 파이프라인 수행 중 에러 발생 시 처리
        task_db[task_id]["status"] = "FAILED"
        task_db[task_id]["progress"] = 0
        task_db[task_id]["message"] = f"오류 발생: {str(e)}"
        task_db[task_id]["error"] = str(e)
        logger.error(f"[{task_id}] 파이프라인 에러: {e}")
    finally:
        # 업로드했던 원본 ZIP 파일 삭제 (디스크 공간 확보)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

# =============================================================================
# 6. API 엔드포인트 정의
# =============================================================================

@app.post("/upload")
async def upload_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    [API] 파일 업로드 및 작업 등록
    - 클라이언트로부터 ZIP 파일을 받습니다.
    - 백그라운드 작업(run_full_pipeline)을 스케줄링합니다.
    - 즉시 task_id를 반환하여 클라이언트가 기다리지 않게 합니다.
    """
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다. .zip 파일만 허용됩니다.")

    # 고유 작업 ID 생성 (UUID)
    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{task_id}{file_extension}"
    
    # 파일 저장 경로 설정 및 생성
    upload_dir = os.path.join(BASE_DATA_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, safe_filename)
    
    # 파일 디스크에 쓰기
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 작업 DB 초기 상태 등록
    task_db[task_id] = {
        "task_id": task_id,
        "status": "PENDING",
        "progress": 0,
        "message": "파일 업로드 완료. 작업 대기 중...",
        "original_filename": file.filename,
        "durations": {"step1_preprocessing": 0, "step2_vggt": 0, "total_pipeline": 0},
        "ply_path": None,
        "error": None
    }
    
    # 비동기 작업 시작 요청
    background_tasks.add_task(run_full_pipeline, task_id, file_path)
    
    return {
        "message": "업로드 성공",
        "task_id": task_id
    }

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    [API] 작업 상태 조회 (Polling)
    - 클라이언트가 일정 간격으로 이 API를 호출하여 진행률을 확인합니다.
    """
    status_info = task_db.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="존재하지 않는 작업 ID입니다.")
    
    download_url = None
    if status_info["status"] == "COMPLETED":
        download_url = f"/download/{task_id}"

    return {
        "task_id": status_info["task_id"],
        "status": status_info["status"],
        "progress_percent": status_info["progress"],
        "message": status_info["message"],
        "download_url": download_url
    }

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    [API] 결과 파일 다운로드
    - 작업이 완료된 후 생성된 .ply 파일을 클라이언트에게 전송합니다.
    """
    task_info = task_db.get(task_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        
    if task_info["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="작업이 아직 완료되지 않았습니다.")
    
    ply_path = task_info.get("ply_path")
    
    if not ply_path or not os.path.exists(ply_path):
        raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")

    return FileResponse(
        path=ply_path, 
        filename=f"3d_model_{task_id}.ply", 
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0으로 호스팅하여 외부 접근 허용
    uvicorn.run(app, host="0.0.0.0", port=8000)