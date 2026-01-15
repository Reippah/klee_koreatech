import cv2
import logging
from pathlib import Path

# 로그 출력 설정: 정보(INFO) 수준까지 출력하며, [로그레벨] 메시지 형태로 표시함
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def process_video_task(file_path: str, task_id: str, interval_seconds: float, target_size=(518, 518)):
    """
    비디오 파일을 로드하여 사용자가 지정한 시간 간격(초)마다 프레임을 추출하고 저장합니다.

    Args:
        file_path (str): 입력 비디오 파일의 절대 또는 상대 경로
        task_id (str): 작업 고유 식별자 (결과 폴더 이름 생성에 사용)
        interval_seconds (float): 추출 간격 (예: 2.0은 2초마다 1장 추출)
        target_size (tuple): 저장될 이미지의 해상도 (가로, 세로), 기본값 (518, 518)
    """
    
    # 입력 경로를 Path 객체로 변환하여 파일 시스템 조작을 용이하게 함
    video_path = Path(file_path)
    
    # 비디오 파일 존재 여부 예외 처리
    if not video_path.exists():
        logging.error(f"경로에서 파일을 찾을 수 없습니다: {file_path}")
        return

    # 결과물 저장 폴더 경로 설정: 원본 파일의 상위 부모 폴더 아래에 'processed_ID' 형식으로 생성
    output_folder = video_path.parent.parent / "result" / f"processed_{task_id}"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # OpenCV를 이용한 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"비디오 파일을 열 수 없습니다: {file_path}")
        return

    # 비디오의 메타데이터에서 FPS(초당 프레임 수) 정보 추출
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps == 0:
        logging.error("비디오의 FPS 정보를 읽어올 수 없어 작업을 중단합니다.")
        return

    # '시간 간격'을 '프레임 단위 간격'으로 변환
    # 예: 30FPS 비디오에서 2초 간격이면 60프레임마다 추출 (최소값은 1로 보장)
    frame_interval = max(1, int(fps * interval_seconds))
    
    logging.info(f"분석 정보 - 파일: {video_path.name} | FPS: {fps} | 설정 간격: {interval_seconds}s")

    count = 0        # 전체 프레임 카운터
    saved_count = 0  # 실제 저장에 성공한 프레임 카운터
    
    try:
        logging.info("프레임 추출 프로세스를 시작합니다...")
        
        while True:
            # 비디오로부터 프레임을 한 장씩 읽음 (ret: 성공 여부, frame: 이미지 데이터)
            ret, frame = cap.read()
            
            # 더 이상 읽을 프레임이 없으면(영상 종료) 반복문 탈출
            if not ret:
                break
            
            # 현재 프레임 번호가 설정된 간격(frame_interval)의 배수일 때만 저장 실행
            if count % frame_interval == 0:
                # 모델 입력 규격이나 저장 규격에 맞춰 이미지 크기 조정
                resized_frame = cv2.resize(frame, target_size)
                
                # 파일 식별을 위해 현재 프레임의 영상 내 재생 시간(초)을 계산
                current_second = count / fps
                # 파일명 형식: sec_시간_frame_번호.jpg (정렬을 위해 0으로 채움)
                frame_filename = f"sec_{current_second:07.2f}s_frame_{count:06d}.jpg"
                save_path = output_folder / frame_filename
                
                # 이미지를 파일로 저장 (JPEG 품질은 95%로 설정)
                success = cv2.imwrite(str(save_path), resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if success:
                    saved_count += 1
            
            # 전체 프레임 카운트 증가
            count += 1
            
    except Exception as e:
        # 처리 중 발생할 수 있는 런타임 오류 예외 처리
        logging.error(f"프레임 추출 중 오류가 발생했습니다: {e}")
    finally:
        # 작업이 완료되거나 오류 발생 시 반드시 비디오 자원을 해제하여 메모리 누수 방지
        cap.release()
        
    logging.info(f"작업 종료: 총 {saved_count}개의 프레임이 '{output_folder}'에 저장되었습니다.")

    return str(output_folder)

# --- 함수 호출 예시 ---
# if __name__ == "__main__":
#     process_video_task("my_video.mp4", "project_01", interval_seconds=1.0)

