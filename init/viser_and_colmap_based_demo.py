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

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

class VGGTJetsonIntegrated:
    def __init__(self, model_url=None, device="cuda"):
        self.device = device
        self.dtype = torch.float16
        print(f"[{self.device}] VGGT 모델 로딩 중...")
        
        self.model = VGGT()
        if model_url is None:
            model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        if os.path.exists(model_url):
            state_dict = torch.load(model_url, map_location=self.device)
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=self.device)
            
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        print("모델 로딩 완료.")

    @torch.no_grad()
    def process_scene(self, image_folder, use_ba=False, mask_sky=True):
        # 1. 이미지 로드
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
        if not image_paths:
            raise ValueError(f"경로에 이미지가 없습니다: {image_folder}")

        vggt_res = 518
        high_res = 1024
        
        print(f"이미지 {len(image_paths)}장 처리 시작...")
        # images_hr: (Batch, N, 3, 1024, 1024) - 여기서 고화질 원본을 들고 있습니다.
        images_hr, _ = load_and_preprocess_images_square(image_paths, high_res)
        images_hr = images_hr.to(self.device)
        
        # 모델 입력용으로 518x518로 줄임
        images_vggt = F.interpolate(images_hr, size=(vggt_res, vggt_res), mode="bilinear")
        
        # 2. VGGT 추론
        print("VGGT 추론 중...")
        with torch.amp.autocast('cuda', dtype=self.dtype):
            predictions = self.model(images_vggt)

        # 3. 데이터 추출
        # 3-1. 카메라 포즈
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (vggt_res, vggt_res))
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()

        # 3-2. World Points (모델 예측 3D 좌표 사용)
        world_points = predictions["world_points"].squeeze(0).cpu().float().numpy()
        world_conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy()
        
        # 3-3. 색상 추출 (★ 여기가 핵심 수정 사항: High-Res 적용 ★)
        # images_hr을 다시 518 크기로 줄이되, 모델을 거치지 않은 원본 색상을 사용
        # (Batch, N, 3, 1024, 1024) -> squeeze -> (N, 3, 1024, 1024)
        input_hr_tensor = images_hr.squeeze(0)
        
        colors_hr_resized = F.interpolate(
            input_hr_tensor, 
            size=(vggt_res, vggt_res), 
            mode="bilinear", 
            align_corners=False
        )
        
        # (N, 3, H, W) -> (N, H, W, 3) 순서로 변경
        colors = colors_hr_resized.permute(0, 2, 3, 1).cpu().float().numpy()
        
        # 값 범위 안전하게 클리핑 (0.0 ~ 1.0)
        colors = np.clip(colors, 0, 1)
        flat_colors = (colors.reshape(-1, 3) * 255).astype(np.uint8)

        # 3-4. Shape 맞추기
        flat_points = world_points.reshape(-1, 3)
        flat_conf = world_conf.reshape(-1)

        # 4. 필터링
        threshold_val = np.percentile(flat_conf, 25.0) 
        mask = (flat_conf >= threshold_val) & (flat_conf > 1e-5)

        # 5. 카메라 좌표계 변환
        cam_to_world = closed_form_inverse_se3(extrinsic)

        # 6. 장면 중심화
        valid_points = flat_points[mask]
        if len(valid_points) > 0:
            scene_center = np.mean(valid_points, axis=0)
            flat_points = flat_points - scene_center
            cam_to_world[:, :3, 3] -= scene_center
        
        print(f"추출된 포인트 수: {len(flat_points)} (유효: {mask.sum()})")

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
        print(f"PLY 저장 중: {save_path}")
        mask = data["mask"]
        pts = data["points"][mask]
        cls = data["colors"][mask]
        
        if len(pts) == 0:
            print("경고: 저장할 포인트가 없습니다.")
            return

        pcd = trimesh.PointCloud(vertices=pts, colors=cls)
        pcd.export(save_path)

    def start_visualization(self, data, port=8080):
        server = viser.ViserServer(host="0.0.0.0", port=port)
        mask = data["mask"]
        
        # Viser에서 보여줄 때는 포인트 사이즈를 조금 키우면 데모와 비슷해집니다.
        server.scene.add_point_cloud(
            name="vggt_pcd",
            points=data["points"][mask],
            colors=data["colors"][mask],
            point_size=0.005 # 기존 0.003 -> 0.005로 약간 키움
        )
        
        H, W = data["image_shape"]
        for i, pose in enumerate(data["poses"]):
            T_world_cam = viser_tf.SE3.from_matrix(pose)
            fov = 2 * np.arctan2(H/2, data["intrinsics"][i, 0, 0])
            
            server.scene.add_camera_frustum(
                name=f"cameras/cam_{i}",
                fov=fov,
                aspect=W/H,
                scale=0.15,
                wxyz=T_world_cam.rotation().wxyz,
                position=T_world_cam.translation()
            )
            
        print(f"서버 주소: http://0.0.0.0:{port}")
        return server