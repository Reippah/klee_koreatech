import cv2
import logging
import zipfile
import os
from pathlib import Path

# 로그 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def process_zip_task(file_path: str, task_id: str, target_size=(518, 518)):
    """
    ZIP 파일을 압축 해제하고 내부의 이미지를 리사이징하여 저장합니다.
    """
    zip_path = Path(file_path)
    if not zip_path.exists():
        logging.error(f"파일을 찾을 수 없습니다: {file_path}")
        return

    # 결과 저장 폴더 설정
    output_folder = zip_path.parent.parent / "result" / f"processed_{task_id}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # 임시 압축 해제 폴더
    temp_extract_dir = zip_path.parent / f"temp_{task_id}"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    
    try:
        # 1. ZIP 파일 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        
        logging.info(f"압축 해제 완료: {zip_path.name}")

        # 2. 이미지 파일 탐색 및 처리 (재귀적으로 탐색)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        # 압축 해제된 폴더 내의 모든 파일을 확인
        image_files = [
            f for f in temp_extract_dir.rglob('*') 
            if f.suffix.lower() in valid_extensions
        ]
        
        logging.info(f"총 {len(image_files)}개의 이미지를 발견했습니다. 리사이징 시작...")

        for idx, img_path in enumerate(image_files):
            # OpenCV로 이미지 로드
            frame = cv2.imread(str(img_path))
            if frame is None:
                logging.warning(f"이미지를 로드할 수 없습니다: {img_path.name}")
                continue

            # 이미지 리사이징
            resized_frame = cv2.resize(frame, target_size)
            
            # 파일명 규칙: img_번호.jpg
            frame_filename = f"img_{idx:06d}.jpg"
            save_path = output_folder / frame_filename
            
            # 저장
            success = cv2.imwrite(str(save_path), resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if success:
                saved_count += 1

    except Exception as e:
        logging.error(f"ZIP 처리 중 오류 발생: {e}")
    finally:
        # 임시 해제 폴더 삭제 (정리)
        import shutil
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        
    logging.info(f"작업 종료: 총 {saved_count}개의 이미지가 '{output_folder}'에 저장되었습니다.")
    return str(output_folder)