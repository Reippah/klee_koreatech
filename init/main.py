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

# -----------------------------------------------------------------------------
# [조건부 모듈 임포트]
# 실제 딥러닝 환경과 개발/테스트 환경을 구분하기 위한 예외 처리 블록입니다.
# 실제 모듈이 없어도 API 서버가 실행될 수 있도록 Mock(가짜) 함수를 정의합니다.
# -----------------------------------------------------------------------------
try:
    from preprocessor import process_zip_task
except ImportError:
    # [DEV/TEST] 전처리 모듈이 없을 경우 동작하는 가짜 함수
    def process_zip_task(file_path, task_id, target_size, blur_threshold):
        # 실제 연산 대신 2초 대기하여 처리 시간을 시뮬레이션
        time.sleep(2)
        # 결과 디렉토리 생성 시뮬레이션
        output_dir = os.path.join(os.path.dirname(file_path), f"processed_{task_id}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

try:
    from viser_and_colmap_based_demo import VGGTJetsonIntegrated
except ImportError:
    # [DEV/TEST] 3D 복원 모델이 없을 경우 동작하는 가짜 클래스
    class VGGTJetsonIntegrated:
        def __init__(self, **kwargs): pass
        
        def process_scene(self, **kwargs): 
            # GPU 추론 시간(5초) 시뮬레이션
            time.sleep(5) 
            return {"points": []}
            
        def save_to_ply(self, results, path): 
            # 더미 파일 생성
            with open(path, 'w') as f: f.write("ply header...")

# -----------------------------------------------------------------------------
# [로깅 설정]
# 서버의 동작 상태, 오류, 진행 상황을 기록하여 디버깅을 돕습니다.
# pipeline.log 파일과 콘솔(Stream)에 동시에 로그를 남깁니다.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VideoPipeline")

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# -----------------------------------------------------------------------------
# [전역 설정 변수]
# 파일 저장 경로 및 HuggingFace 모델 다운로드 URL 정의
# -----------------------------------------------------------------------------
BASE_DATA_DIR = "/mnt/sdcard/klee_koreatech/init"
MODEL_PATH = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

# -----------------------------------------------------------------------------
# [모델 초기화]
# 서버 시작 시 무거운 딥러닝 모델을 미리 메모리에 로드합니다.
# 요청이 올 때마다 로드하면 응답 속도가 매우 느려지므로 전역 변수로 관리합니다.
# -----------------------------------------------------------------------------
logger.info("VGGT 모델 초기화 중...")
try:
    # device="cuda"를 통해 GPU에 모델을 로드
    vggt_processor = VGGTJetsonIntegrated(model_url=MODEL_PATH, device="cuda")
    logger.info("VGGT 모델 준비 완료.")
except Exception as e:
    # 모델 로드 실패 시(로컬 테스트 등) 로그를 남기고 Mock 객체로 폴백(Fallback)
    logger.error(f"모델 로드 실패 (Mock 모드라면 무시): {e}")
    vggt_processor = VGGTJetsonIntegrated() 

# -----------------------------------------------------------------------------
# [동시성 제어 - Mutex Lock]
# 매우 중요: 여러 사용자가 동시에 요청을 보내더라도, GPU 메모리는 한정적입니다.
# 따라서 GPU를 사용하는 '추론 단계'는 한 번에 하나의 작업만 수행되도록 강제합니다.
# -----------------------------------------------------------------------------
gpu_lock = threading.Lock()

# -----------------------------------------------------------------------------
# [작업 상태 저장소]
# 각 작업(Task)의 진행률, 상태, 메시지 등을 저장하는 인메모리 딕셔너리입니다.
# 서버 재시작 시 초기화되므로, 영구 저장이 필요하면 Redis나 DB를 연동해야 합니다.
# Key: task_id (UUID), Value: 작업 정보 Dict
# -----------------------------------------------------------------------------
task_db: Dict[str, Dict] = {}

# [진행률 상수] 각 단계별 진행 퍼센트 정의
STATUS_PROGRESS = {
    "PENDING": 5,              # 대기 중
    "STEP1_PREPROCESSING": 20, # CPU 전처리 중
    "WAITING_FOR_GPU": 50,     # 전처리 완료 후 GPU Lock 대기 중
    "STEP2_VGGT": 70,          # GPU 추론 진행 중
    "SAVING": 90,              # 결과 파일 저장 중
    "COMPLETED": 100,          # 완료
    "FAILED": 0                # 실패
}

def _get_user_friendly_message(status: str) -> str:
    """
    서버 내부의 상태 코드(status string)를
    모바일 앱 사용자가 이해하기 쉬운 안내 문구로 변환하는 헬퍼 함수입니다.
    """
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

def run_full_pipeline(task_id: str, file_path: str):
    """
    [백그라운드 파이프라인 핵심 로직]
    사용자의 요청에 응답을 보낸 후, 백그라운드 스레드에서 실제로 실행되는 함수입니다.
    
    1. CPU 전처리 (병렬 실행 가능)
    2. GPU 추론 (Lock을 통해 직렬 실행)
    3. 결과 저장 및 상태 업데이트
    """
    pipeline_start_time = time.time()
    
    # 작업 상태 DB 업데이트를 위한 내부 함수
    def update_status(status_code: str):
        task_db[task_id]["status"] = status_code
        task_db[task_id]["progress"] = STATUS_PROGRESS.get(status_code, 0)
        task_db[task_id]["message"] = _get_user_friendly_message(status_code)

    try:
        # -----------------------------------------------------
        # Step 1: CPU 기반 전처리 (이미지 리사이징, 블러 처리 등)
        # -----------------------------------------------------
        update_status("STEP1_PREPROCESSING")
        logger.info(f"[{task_id}] 전처리 단계 시작")
        
        step1_start = time.time()
        
        # 전처리 모듈 호출
        processed_folder = process_zip_task(
            file_path=file_path, 
            task_id=task_id, 
            target_size=(518, 518), 
            blur_threshold=100.0
        )
        
        step1_end = time.time()
        task_db[task_id]["durations"]["step1_preprocessing"] = round(step1_end - step1_start, 2)
        logger.info(f"[{task_id}] 전처리 완료")

        # -----------------------------------------------------
        # Step 2: GPU 기반 3D 복원 (Critical Section)
        # 다른 작업이 GPU를 사용 중이면 여기서 대기합니다.
        # -----------------------------------------------------
        update_status("WAITING_FOR_GPU")
        logger.info(f"[{task_id}] GPU 자원 대기 중...")

        with gpu_lock:
            # 락을 획득해야만 이 블록 내부로 진입 가능
            update_status("STEP2_VGGT")
            logger.info(f"[{task_id}] GPU 락 획득, VGGT 추론 시작")
            
            step2_start = time.time()
            vggt_results = None

            try:
                # inference_mode: 불필요한 Gradient 계산을 꺼서 메모리를 아낍니다.
                with torch.inference_mode(): 
                    vggt_results = vggt_processor.process_scene(
                        image_folder=processed_folder, 
                        use_ba=False, 
                        mask_sky=True
                    )
                    
                    update_status("SAVING")
                    # 결과 파일(.ply) 저장 경로 생성
                    output_dir = os.path.dirname(processed_folder)
                    output_ply_name = f"reconstruction_{task_id}.ply"
                    output_ply_path = os.path.join(output_dir, output_ply_name)
                    
                    # PLY 파일 쓰기
                    vggt_processor.save_to_ply(vggt_results, output_ply_path)
            
            finally:
                # [메모리 관리 중요]
                # GPU 작업이 끝나면 즉시 메모리를 해제하여 다음 작업을 위해 VRAM을 비웁니다.
                if vggt_results:
                    del vggt_results
                gc.collect() # Python 가비지 컬렉터 실행
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # PyTorch 캐시 메모리 비우기

            step2_end = time.time()
            task_db[task_id]["durations"]["step2_vggt"] = round(step2_end - step2_start, 2)
            task_db[task_id]["ply_path"] = output_ply_path
            logger.info(f"[{task_id}] 추론 및 저장 완료")

        # -----------------------------------------------------
        # Step 3: 파이프라인 완료 처리
        # -----------------------------------------------------
        total_duration = round(time.time() - pipeline_start_time, 2)
        task_db[task_id]["durations"]["total_pipeline"] = total_duration
        update_status("COMPLETED")
        logger.info(f"[{task_id}] 파이프라인 정상 종료 (총 {total_duration}초)")

    except Exception as e:
        # 파이프라인 도중 에러 발생 시 처리
        task_db[task_id]["status"] = "FAILED"
        task_db[task_id]["progress"] = 0
        task_db[task_id]["message"] = f"오류 발생: {str(e)}"
        task_db[task_id]["error"] = str(e)
        logger.error(f"[{task_id}] 파이프라인 에러: {e}")
    finally:
        # [청소] 공간 확보를 위해 업로드된 원본 zip 파일은 삭제
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

# -----------------------------------------------------------------------------
# API 엔드포인트: 파일 업로드
# -----------------------------------------------------------------------------
@app.post("/upload")
async def upload_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    [1단계] 클라이언트가 파일을 업로드하는 엔드포인트
    - 파일을 받아 디스크에 저장합니다.
    - 고유 작업 ID(UUID)를 발급합니다.
    - 실제 작업(run_full_pipeline)은 'background_tasks'에 등록하여 비동기로 실행하고,
      클라이언트에게는 즉시 ID를 반환합니다 (Non-blocking).
    """
    # 파일 확장자 검사
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다. .zip 파일만 허용됩니다.")

    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{task_id}{file_extension}"
    
    upload_dir = os.path.join(BASE_DATA_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, safe_filename)
    
    # 파일 저장 (대용량 파일 처리를 위해 shutil 사용)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 작업 상태 DB 초기화
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
    
    # 백그라운드 작업 시작
    background_tasks.add_task(run_full_pipeline, task_id, file_path)
    
    return {
        "message": "업로드 성공",
        "task_id": task_id
    }

# -----------------------------------------------------------------------------
# API 엔드포인트: 상태 조회 (Polling)
# -----------------------------------------------------------------------------
@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    [2단계] 작업 진행 상황을 확인하는 엔드포인트
    - 클라이언트(모바일)는 이 API를 2~3초 간격으로 호출(Polling)해야 합니다.
    - 진행률(%), 현재 상태 메시지, 완료 시 다운로드 링크를 반환합니다.
    """
    status_info = task_db.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="존재하지 않는 작업 ID입니다.")
    
    # 작업이 완료된 경우 다운로드 URL 생성
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

# -----------------------------------------------------------------------------
# API 엔드포인트: 결과 파일 다운로드
# -----------------------------------------------------------------------------
@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    [3단계] 최종 결과물(.ply)을 다운로드하는 엔드포인트
    - 상태가 COMPLETED일 때만 다운로드가 가능합니다.
    """
    task_info = task_db.get(task_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        
    if task_info["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="작업이 아직 완료되지 않았습니다.")
    
    ply_path = task_info.get("ply_path")
    
    if not ply_path or not os.path.exists(ply_path):
        raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다 (서버에서 삭제되었을 수 있음).")

    return FileResponse(
        path=ply_path, 
        filename=f"3d_model_{task_id}.ply", 
        media_type='application/octet-stream'
    )

# -----------------------------------------------------------------------------
# 서버 실행 진입점
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" 설정을 통해 외부(동일 네트워크의 모바일 기기 등)에서 접속을 허용합니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)