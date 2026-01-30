import os
import glob
import time
import torch
import torch.nn.functional as F
import numpy as np
import viser
import viser.transforms as viser_tf
import pycolmap
import trimesh
import cv2
from tqdm import tqdm

# VGGT 모델 및 유틸리티 함수 임포트
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

class VGGTJetsonIntegrated:
    def __init__(self, model_url=None, device="cuda"):
        """
        VGGT 모델을 초기화하고 가중치를 로드합니다.
        """
        self.device = device
        self.dtype = torch.float16
        print(f"[{self.device}] VGGT 모델 로딩 시작...")
        
        self.model = VGGT()
        if model_url is None:
            model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        # 로컬 캐시 확인 후 모델 로드
        if os.path.exists(model_url):
            state_dict = torch.load(model_url, map_location=self.device)
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=self.device)
            
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        print("모델 로딩 및 초기화 완료.")

    @torch.no_grad()
    def process_scene(self, image_folder, use_ba=False, mask_sky=True):
        """
        이미지 폴더를 입력받아 VGGT 모델을 통해 3D 구조(포인트, 색상, 카메라 포즈)를 추론합니다.
        """
        # 이미지 파일 목록 로드
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
        if not image_paths:
            raise ValueError(f"해당 경로에서 이미지를 찾을 수 없습니다: {image_folder}")

        vggt_res = 518   # 모델 입력 해상도
        high_res = 1024  # 텍스처 추출을 위한 고해상도
        
        print(f"총 {len(image_paths)}장의 이미지 처리 시작...")
        
        # 고해상도 이미지 로드 및 전처리 (배치 차원 포함)
        images_hr, _ = load_and_preprocess_images_square(image_paths, high_res)
        images_hr = images_hr.to(self.device)
        
        # 모델 입력을 위해 518x518 크기로 다운샘플링
        images_vggt = F.interpolate(images_hr, size=(vggt_res, vggt_res), mode="bilinear")
        
        # VGGT 모델 추론 (Mixed Precision 사용)
        print("VGGT 네트워크 추론 진행 중...")
        with torch.amp.autocast('cuda', dtype=self.dtype):
            predictions = self.model(images_vggt)

        # 모델 출력 데이터 파싱
        # 1. 카메라 포즈 (Extrinsic) 및 내부 파라미터 (Intrinsic) 복원
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (vggt_res, vggt_res))
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()

        # 2. 3D 포인트 좌표 및 신뢰도 점수 추출
        world_points = predictions["world_points"].squeeze(0).cpu().float().numpy()
        world_conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy()
        
        # 3. 색상 정보 추출 (고해상도 이미지 활용)
        # 추론은 저해상도로 하더라도, 포인트 클라우드의 색상은 원본 고해상도 이미지를 리사이징하여 사용합니다.
        input_hr_tensor = images_hr.squeeze(0)
        
        colors_hr_resized = F.interpolate(
            input_hr_tensor, 
            size=(vggt_res, vggt_res), 
            mode="bilinear", 
            align_corners=False
        )
        
        # 텐서 차원 변경: (N, C, H, W) -> (N, H, W, C)
        colors = colors_hr_resized.permute(0, 2, 3, 1).cpu().float().numpy()
        
        # 색상 값 클리핑 및 정수형 변환 (0~255)
        colors = np.clip(colors, 0, 1)
        flat_colors = (colors.reshape(-1, 3) * 255).astype(np.uint8)

        # 데이터 평탄화 (N*H*W, 3)
        flat_points = world_points.reshape(-1, 3)
        flat_conf = world_conf.reshape(-1)

        # 4. 유효 포인트 필터링 (신뢰도 기반)
        # 하위 25% 신뢰도를 가진 포인트와 신뢰도가 0에 가까운 포인트 제거
        threshold_val = np.percentile(flat_conf, 25.0) 
        mask = (flat_conf >= threshold_val) & (flat_conf > 1e-5)

        # 5. 카메라 좌표계 변환 (World -> Camera 역변환)
        cam_to_world = closed_form_inverse_se3(extrinsic)

        # 6. 장면 중심화 (Centering)
        # 포인트 클라우드의 평균 지점을 원점(0,0,0)으로 이동시킵니다.
        valid_points = flat_points[mask]
        if len(valid_points) > 0:
            scene_center = np.mean(valid_points, axis=0)
            flat_points = flat_points - scene_center
            cam_to_world[:, :3, 3] -= scene_center
        
        print(f"포인트 클라우드 생성 완료: {len(flat_points)}개 중 {mask.sum()}개 유효")

        return {
            "points": flat_points,
            "colors": flat_colors,
            "poses": cam_to_world,
            "intrinsics": intrinsic,
            "conf": flat_conf,
            "mask": mask,
            "image_shape": (vggt_res, vggt_res)
        }

    def save_to_ply(self, data, save_path):
        """
        추론 결과를 .ply 파일 형식으로 저장합니다.
        """
        print(f"PLY 파일 저장 경로: {save_path}")
        mask = data["mask"]
        pts = data["points"][mask]
        cls = data["colors"][mask]
        
        if len(pts) == 0:
            print("저장할 유효 포인트가 없습니다.")
            return

        pcd = trimesh.PointCloud(vertices=pts, colors=cls)
        pcd.export(save_path)

    def start_visualization(self, data, port=8080):
        """
        Viser 라이브러리를 사용하여 웹 브라우저에서 3D 결과를 시각화합니다.
        """
        server = viser.ViserServer(host="0.0.0.0", port=port)
        mask = data["mask"]
        
        # 포인트 클라우드 추가 (시인성을 위해 포인트 크기 조정)
        server.scene.add_point_cloud(
            name="vggt_pcd",
            points=data["points"][mask],
            colors=data["colors"][mask],
            point_size=0.005
        )
        
        # 카메라 프러스텀(Frustum) 시각화
        H, W = data["image_shape"]
        for i, pose in enumerate(data["poses"]):
            T_world_cam = viser_tf.SE3.from_matrix(pose)
            # 수직 화각(FOV) 계산
            fov = 2 * np.arctan2(H/2, data["intrinsics"][i, 0, 0])
            
            server.scene.add_camera_frustum(
                name=f"cameras/cam_{i}",
                fov=fov,
                aspect=W/H,
                scale=0.15,
                wxyz=T_world_cam.rotation().wxyz,
                position=T_world_cam.translation()
            )
            
        print(f"시각화 서버 실행 중: http://0.0.0.0:{port}")
        return server