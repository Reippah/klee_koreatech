import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import viser
import viser.transforms as viser_tf
import pycolmap
import trimesh  # 추가된 라이브러리
from tqdm import tqdm

# ... (기존 임포트 및 클래스 정의 시작) ...

class VGGTJetsonIntegrated:
    def __init__(self, model_url=None, device="cuda"):
        self.device = device
        self.dtype = torch.float16 
        self.model = VGGT()
        if model_url is None:
            model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
        self.model.eval().to(self.device)

    # ... (process_scene 메서드 생략 - 이전과 동일) ...

    def save_to_ply(self, data, save_path, confidence_threshold=0.2):
        """
        [저장 로직] 생성된 포인트 클라우드를 .ply 파일로 저장
        Args:
            data (dict): process_scene에서 반환된 데이터
            save_path (str): 저장할 파일 경로 (예: "output.ply")
            confidence_threshold (float): 저장할 포인트의 최소 신뢰도
        """
        print(f"파일 저장 중: {save_path}...")
        
        points = data["points"]
        # 이미지 형태 (S, H, W, 3)를 포인트 형태 (N, 3)로 변환
        colors = data["colors"].reshape(-1, 3)
        conf = data["conf"]

        # 신뢰도 기반 필터링 (너무 노이즈가 많은 점은 제외하고 저장)
        mask = conf >= confidence_threshold
        filtered_points = points[mask]
        filtered_colors = colors[mask]

        # trimesh를 사용하여 포인트 클라우드 생성 및 저장
        # trimesh는 (N, 3) 형태의 포인트와 (N, 3) 형태의 RGB(0-255) 컬러를 지원합니다.
        pcd = trimesh.PointCloud(vertices=filtered_points, colors=filtered_colors)
        pcd.export(save_path)
        
        print(f"저장 완료! 총 {len(filtered_points)}개의 포인트가 기록되었습니다.")

    def start_visualization(self, data):
        # ... (이전 시각화 메서드와 동일) ...
        pass

# --- 실행부 ---
if __name__ == "__main__":
    pipeline = VGGTJetsonIntegrated()
    
    scene_path = "path/to/your/images"
    # 1. 3D 재구성 수행
    result_data = pipeline.process_scene(scene_path, use_ba=True)
    
    # 2. .ply 파일로 저장 (추가된 부분)
    # 장면 폴더 내에 sparse 폴더를 만들어 저장하는 것이 일반적입니다.
    output_dir = os.path.join(scene_path, "sparse")
    os.makedirs(output_dir, exist_ok=True)
    pipeline.save_to_ply(result_data, os.path.join(output_dir, "reconstruction.ply"))
    
    # 3. 시각화 서버 시작
    vis_server = pipeline.start_visualization(result_data)
    
    import time
    while True:
        time.sleep(1)
