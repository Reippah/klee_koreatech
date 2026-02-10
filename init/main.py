import torch
import gc
import os
import shutil
import logging
import uuid
import time
import threading
import subprocess
from typing import Dict, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

# =============================================================================
# [환경 설정 및 상수 정의]
# =============================================================================

# SSH 접속 정보 (Jetson -> HostPC)
# HostPC에서 무거운 MeshLab 처리를 수행하기 위해 접속할 계정과 IP입니다.
# 사전에 'ssh-copy-id' 등을 통해 비밀번호 없이 접속 가능하도록 설정되어 있어야 합니다.
SSH_USER_HOST = "vic05@192.168.0.3"  

# Jetson(현재 기기)에서 HostPC의 폴더를 마운트한 경로
# sshfs를 통해 HostPC의 특정 폴더가 이 경로에 연결되어 있다고 가정합니다.
MOUNT_DIR_JETSON = "/home/viclab/hostpc/meshlab_test"

# HostPC(원격 기기) 내에서의 실제 작업 폴더 경로
# SSH로 명령을 보낼 때, HostPC 입장에서 파일이 어디 있는지 알려주기 위해 사용합니다.
REAL_DIR_HOSTPC = "/home/vic05/meshlab_test" 

# 파일 저장 및 작업의 기본이 되는 루트 디렉토리
BASE_DATA_DIR = "/home/viclab/klee_koreatech/init"

# =============================================================================
# [초기화 및 모듈 로딩]
# =============================================================================

# 로깅 설정: 서버 로그를 파일과 콘솔에 동시에 출력하여 디버깅을 돕습니다.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
)
logger = logging.getLogger("VideoPipeline")

app = FastAPI()

# -----------------------------------------------------------------------------
# [모듈 모킹(Mocking)]
# 개발 환경이나 의존성이 없는 환경에서도 서버가 실행되도록 처리하는 부분입니다.
# 실제 배포 시에는 해당 모듈들이 설치되어 있어야 합니다.
# -----------------------------------------------------------------------------

# 1. 전처리 모듈 (ZIP 파일 처리)
try:
    from preprocessor import process_zip_task
except ImportError:
    # 모듈이 없을 경우 테스트용 더미 함수 정의
    def process_zip_task(file_path, task_id, target_size, blur_threshold):
        time.sleep(1) # 작업 시늉
        output_dir = os.path.join(os.path.dirname(file_path), f"processed_{task_id}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

# 2. 3D 모델 생성 모듈 (VGGT)
try:
    from viser_and_colmap_based_demo import VGGTJetsonIntegrated
except ImportError:
    # 모듈이 없을 경우 테스트용 더미 클래스 정의
    class VGGTJetsonIntegrated:
        def __init__(self, **kwargs): pass
        def process_scene(self, **kwargs): return {"points": []}
        def save_to_ply(self, results, path):
            with open(path, 'w') as f: f.write("ply header")

# -----------------------------------------------------------------------------
# [모델 및 전역 변수 초기화]
# -----------------------------------------------------------------------------

# VGGT 모델 로드 (초기 실행 시 시간이 걸릴 수 있음)
MODEL_PATH = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
logger.info("VGGT 모델 초기화 및 로드 중...")
try:
    vggt_processor = VGGTJetsonIntegrated(model_url=MODEL_PATH, device="cuda")
except:
    vggt_processor = VGGTJetsonIntegrated() # 로드 실패 시 더미 사용

# GPU 동시 접근 방지를 위한 락 (Lock)
# 여러 요청이 동시에 들어와도 GPU를 사용하는 작업은 한 번에 하나씩만 처리합니다.
gpu_lock = threading.Lock()

# 작업 상태를 저장하는 인메모리 데이터베이스
# 실제 서비스에서는 Redis나 SQL DB를 사용하는 것이 좋습니다.
task_db: Dict[str, Dict] = {}

# =============================================================================
# [핵심 로직: 백그라운드 파이프라인]
# =============================================================================

# 작업 상태별 진행률 매핑
STATUS_PROGRESS = {
    "PENDING": 5,                 # 대기 중
    "STEP1_PREPROCESSING": 20,    # CPU 전처리 (이미지 리사이징 등)
    "WAITING_FOR_GPU": 40,        # 다른 작업이 GPU 사용 중일 때 대기
    "STEP2_VGGT": 60,             # GPU 추론 (3D 모델 생성)
    "STEP3_CLEANING_REMOTE": 80,  # 원격 PC에서 노이즈 제거
    "SAVING": 95,                 # 결과 파일 정리 및 저장
    "COMPLETED": 100,             # 완료
    "FAILED": 0                   # 실패
}

def run_full_pipeline(task_id: str, file_path: str):
    """
    업로드된 파일에 대해 전체 3D 변환 파이프라인을 실행하는 함수입니다.
    FastAPI의 BackgroundTasks에 의해 비동기로 실행됩니다.
    """
    pipeline_start_time = time.time()
    
    # 내부 함수: 상태 업데이트
    def update_status(status: str, msg: str = None):
        task_db[task_id]["status"] = status
        task_db[task_id]["progress"] = STATUS_PROGRESS.get(status, 0)
        if msg:
            task_db[task_id]["message"] = msg
        
        # [핵심 수정 사항]
        # CPU가 바쁜 작업(While/For loop 등) 중에 상태를 업데이트하면
        # FastAPI(메인 스레드)가 /status 요청에 응답할 틈이 없어 타임아웃이 발생합니다.
        # sleep(0.1)을 주어 강제로 컨텍스트 스위칭을 유도해 서버가 응답을 보낼 수 있게 합니다.
        time.sleep(0.1)

    try:
        # ---------------------------------------------------------------------
        # [Step 1] CPU 전처리 단계
        # 업로드된 ZIP 파일을 풀고, 모델 입력에 맞게 이미지를 전처리합니다.
        # ---------------------------------------------------------------------
        update_status("STEP1_PREPROCESSING", "이미지 전처리 중...")
        step1_start = time.time()
        
        processed_folder = process_zip_task(file_path, task_id, (518, 518), 30.0)
        
        task_db[task_id]["durations"]["step1"] = round(time.time() - step1_start, 2)

        # ---------------------------------------------------------------------
        # [Step 2] GPU 추론 단계 (VGGT)
        # 2D 이미지들로부터 3D 포인트 클라우드(PLY)를 생성합니다.
        # GPU 메모리는 한정적이므로 Lock을 사용하여 한 번에 하나의 작업만 수행합니다.
        # ---------------------------------------------------------------------
        update_status("WAITING_FOR_GPU", "GPU 대기 중...")
        
        # 작업 경로 및 파일명 설정
        work_dir = os.path.dirname(processed_folder)
        raw_ply_name = f"raw_{task_id}.ply"
        cleaned_obj_name = f"cleaned_{task_id}.obj"
        local_ply_path = os.path.join(work_dir, raw_ply_name) 
        
        with gpu_lock:
            update_status("STEP2_VGGT", "3D 모델 생성 중 (VGGT)...")
            step2_start = time.time()
            vggt_results = None
            
            try:
                # 추론 모드 진입 (Gradient 계산 비활성화로 메모리 절약)
                with torch.inference_mode():
                    vggt_results = vggt_processor.process_scene(image_folder=processed_folder, use_ba=False)
                    vggt_processor.save_to_ply(vggt_results, local_ply_path)
            finally:
                # 메모리 누수 방지를 위한 명시적 해제
                if vggt_results: del vggt_results
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            task_db[task_id]["durations"]["step2"] = round(time.time() - step2_start, 2)

        # ---------------------------------------------------------------------
        # [Step 3] 원격(HostPC) MeshLab 후처리 단계
        # 생성된 PLY 파일을 더 강력한 성능의 HostPC로 보내 노이즈를 제거(Cleaning)합니다.
        # ---------------------------------------------------------------------
        update_status("STEP3_CLEANING_REMOTE", "HostPC에서 노이즈 제거 중...")
        step3_start = time.time()
        
        # 파일 경로 설정 (로컬 공유폴더 vs 원격 실제경로)
        shared_ply_path = os.path.join(MOUNT_DIR_JETSON, raw_ply_name)
        shared_obj_path = os.path.join(MOUNT_DIR_JETSON, cleaned_obj_name)
        
        host_ply_path = os.path.join(REAL_DIR_HOSTPC, raw_ply_name)
        host_obj_path = os.path.join(REAL_DIR_HOSTPC, cleaned_obj_name)
        host_script_path = os.path.join(REAL_DIR_HOSTPC, "meshlab_test_pls.py")
        
        # 후처리 실패 시 원본(PLY)을 반환하기 위한 기본값 설정
        final_file_path = local_ply_path 
        final_ext = "ply"

        try:
            # 1. SSHFS 마운트 경로로 파일 복사 (즉, HostPC로 파일 전송)
            logger.info(f"[{task_id}] 공유 폴더로 파일 전송 중...")
            shutil.copy(local_ply_path, shared_ply_path)
            
            # 2. SSH를 통해 HostPC의 파이썬 스크립트 원격 실행
            cmd = [
                "ssh", SSH_USER_HOST,
                f"python3 {host_script_path} {host_ply_path} {host_obj_path}"
            ]
            
            logger.info(f"[{task_id}] SSH 명령 실행: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # 3. 결과 확인 및 회수
            if result.returncode == 0:
                logger.info(f"[{task_id}] HostPC 작업 완료.")
                
                # 결과 파일(OBJ)을 로컬 작업 폴더로 가져옴
                local_obj_path = os.path.join(work_dir, cleaned_obj_name)
                
                if os.path.exists(shared_obj_path):
                    shutil.copy(shared_obj_path, local_obj_path)
                    final_file_path = local_obj_path
                    final_ext = "obj"
                    logger.info(f"[{task_id}] 클리닝된 OBJ 파일 확보 성공.")
                    
                    # 공유 폴더 내 임시 파일 정리 (선택 사항)
                    try:
                        os.remove(shared_ply_path)
                        os.remove(shared_obj_path)
                    except: pass
                else:
                    logger.error(f"[{task_id}] SSH 성공했으나 결과 파일이 공유 폴더에 없음.")
            else:
                logger.error(f"[{task_id}] SSH 실행 실패: {result.stderr}")

        except Exception as e:
            logger.error(f"[{task_id}] 원격 클리닝 중 예외 발생: {e}")

        task_db[task_id]["final_path"] = final_file_path
        task_db[task_id]["file_type"] = final_ext
        task_db[task_id]["durations"]["step3"] = round(time.time() - step3_start, 2)

        # ---------------------------------------------------------------------
        # [Step 4] 완료 처리
        # 최종 결과를 저장하고 상태를 완료로 변경합니다.
        # ---------------------------------------------------------------------
        
        # [수정됨] 95% 단계를 명시적으로 호출하여 사용자에게 마무리 중임을 알림
        update_status("SAVING", "결과 파일 정리 중...")
        
        total_time = round(time.time() - pipeline_start_time, 2)
        task_db[task_id]["durations"]["total"] = total_time
        
        update_status("COMPLETED", "작업 완료! 다운로드 가능합니다.")
        logger.info(f"[{task_id}] 파이프라인 종료 (총 {total_time}초)")

    except Exception as e:
        # 예외 발생 시 상태를 FAILED로 변경하여 사용자에게 알림
        task_db[task_id]["status"] = "FAILED"
        task_db[task_id]["message"] = f"시스템 오류: {str(e)}"
        logger.error(f"[{task_id}] 치명적 오류: {e}")
        
    finally:
        # 업로드된 원본 ZIP 파일 삭제 (디스크 공간 확보)
        if os.path.exists(file_path):
            os.remove(file_path)

# =============================================================================
# [API 엔드포인트 정의]
# =============================================================================

@app.post("/upload")
async def upload_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    [API] 파일 업로드 및 작업 시작
    클라이언트가 ZIP 파일을 업로드하면 Task ID를 발급하고 백그라운드 작업을 시작합니다.
    """
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="zip 파일만 허용됩니다.")
    
    # 고유 Task ID 생성 및 저장 경로 설정
    task_id = str(uuid.uuid4())
    upload_dir = os.path.join(BASE_DATA_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{task_id}.zip")
    
    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 작업 상태 초기화
    task_db[task_id] = {
        "task_id": task_id,
        "status": "PENDING",
        "progress": 0,
        "message": "대기 중...",
        "durations": {},
        "final_path": None
    }
    
    # 백그라운드 작업 큐에 파이프라인 함수 등록
    background_tasks.add_task(run_full_pipeline, task_id, file_path)
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    [API] 작업 상태 조회 (Polling 용)
    프론트엔드에서 주기적으로 호출하여 진행률(progress)을 확인합니다.
    """
    if task_id not in task_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    info = task_db[task_id]
    
    # 완료 상태일 경우에만 다운로드 URL 제공
    dl_url = f"/download/{task_id}" if info["status"] == "COMPLETED" else None
    
    return {
        "status": info["status"],
        "progress": info["progress"], 
        "message": info.get("message", ""),
        "download_url": dl_url
    }

@app.get("/download/{task_id}")
async def download(task_id: str):
    """
    [API] 결과 파일 다운로드
    작업이 완료된 후 생성된 결과 파일(.obj 또는 .ply)을 스트림으로 전송합니다.
    """
    info = task_db.get(task_id)
    if not info or info["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="파일이 아직 준비되지 않았습니다.")
    
    return FileResponse(
        path=info["final_path"],
        filename=f"result_{task_id}.{info['file_type']}",
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    import uvicorn
    # 서버 실행 (모든 IP 허용, 8000번 포트)
    uvicorn.run(app, host="0.0.0.0", port=8000)