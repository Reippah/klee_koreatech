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

# VGGT 모델 관련 모듈 (모델 정의, 데이터 로딩, 기하학 연산)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

class VGGTJetsonIntegrated:
    def __init__(self, model_url=None, device="cuda"):
        """
        [초기화 단계]
        서버가 시작될 때 딱 한 번 호출됩니다.
        무거운 딥러닝 모델(VGGT)을 GPU 메모리에 로드하여, 이후 요청 처리를 빠르게 합니다.
        """
        self.device = device
        # 메모리 효율성을 위해 16비트 부동소수점(Float16) 사용
        self.dtype = torch.float16
        print(f"[{self.device}] VGGT 모델 로딩 시작...")
        
        # 모델 구조 인스턴스화
        self.model = VGGT()
        if model_url is None:
            # HuggingFace 등 원격 저장소 URL 기본값
            model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        # 가중치 파일 로드 (로컬 캐시가 있으면 사용, 없으면 다운로드)
        if os.path.exists(model_url):
            state_dict = torch.load(model_url, map_location=self.device)
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=self.device)
            
        # 모델에 가중치 주입 및 평가 모드(Eval) 전환 (학습 불필요)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        print("모델 로딩 및 초기화 완료.")

    @torch.no_grad()
    def process_scene(self, image_folder, use_ba=False, mask_sky=True):
        """
        [핵심 추론 단계]
        입력된 이미지 폴더를 읽어 3D 포인트 클라우드와 카메라 정보를 생성합니다.
        서버 코드(run_full_pipeline)의 Step 2에서 호출되는 함수입니다.
        """
        # 1. 이미지 파일 경로 수집
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
        if not image_paths:
            raise ValueError(f"해당 경로에서 이미지를 찾을 수 없습니다: {image_folder}")

        # 모델 입력 해상도(518)와 텍스처용 고해상도(1024) 설정
        vggt_res = 518   
        high_res = 1024  
        
        print(f"총 {len(image_paths)}장의 이미지 처리 시작...")
        
        # 2. 이미지 로드 및 텐서 변환 (전처리)
        # images_hr: (Batch, Channel, Height, Width) 형태의 텐서
        images_hr, _ = load_and_preprocess_images_square(image_paths, high_res)
        images_hr = images_hr.to(self.device)
        
        # 모델 입력 규격(518x518)에 맞게 리사이징
        images_vggt = F.interpolate(images_hr, size=(vggt_res, vggt_res), mode="bilinear")
        
        # 3. 신경망 추론 (Inference)
        # GPU 연산을 통해 이미지에서 3D 정보를 예측합니다.
        # autocast를 사용하여 연산 속도를 높이고 메모리를 절약합니다.
        print("VGGT 네트워크 추론 진행 중...")
        with torch.amp.autocast('cuda', dtype=self.dtype):
            predictions = self.model(images_vggt)

        # 4. 결과 파싱 (모델의 출력값을 3D 정보로 변환)
        
        # (A) 카메라 파라미터 복원
        # pose_enc: 모델이 예측한 압축된 포즈 정보 -> 외부(Extrinsic)/내부(Intrinsic) 파라미터로 변환
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (vggt_res, vggt_res))
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()

        # (B) 3D 포인트 좌표 복원
        world_points = predictions["world_points"].squeeze(0).cpu().float().numpy()
        world_conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy()
        
        # (C) 색상 정보 매핑
        # 포인트의 색상은 저해상도 모델 입력 대신, 원본(고해상도) 이미지에서 가져와 품질을 높입니다.
        input_hr_tensor = images_hr.squeeze(0)
        
        colors_hr_resized = F.interpolate(
            input_hr_tensor, 
            size=(vggt_res, vggt_res), 
            mode="bilinear", 
            align_corners=False
        )
        
        # (C, H, W) -> (H, W, C) 순서로 변경하여 이미지 포맷 맞춤
        colors = colors_hr_resized.permute(0, 2, 3, 1).cpu().float().numpy()
        
        # 색상값 정규화 해제 (0.0~1.0 -> 0~255)
        colors = np.clip(colors, 0, 1)
        flat_colors = (colors.reshape(-1, 3) * 255).astype(np.uint8)

        # 데이터를 1차원 배열로 펼침 (N개의 점)
        flat_points = world_points.reshape(-1, 3)
        flat_conf = world_conf.reshape(-1)

        # 5. 후처리 및 필터링
        # 신뢰도가 낮은(노이즈로 추정되는) 포인트들을 제거합니다.
        threshold_val = np.percentile(flat_conf, 25.0)  # 하위 25% 제거
        mask = (flat_conf >= threshold_val) & (flat_conf > 1e-5)

        # 카메라 좌표계를 월드 좌표계로 변환 (역행렬 계산)
        cam_to_world = closed_form_inverse_se3(extrinsic)

        # 6. 장면 중심 정렬 (Centering)
        # 생성된 3D 모델이 뷰어의 중앙에 오도록 좌표를 이동시킵니다.
        valid_points = flat_points[mask]
        if len(valid_points) > 0:
            scene_center = np.mean(valid_points, axis=0)
            flat_points = flat_points - scene_center
            cam_to_world[:, :3, 3] -= scene_center # 카메라 위치도 같이 이동
        
        print(f"포인트 클라우드 생성 완료: {len(flat_points)}개 중 {mask.sum()}개 유효")

        # 최종 결과 딕셔너리 반환
        return {
            "points": flat_points,   # 3D 점 좌표 (XYZ)
            "colors": flat_colors,   # 점 색상 (RGB)
            "poses": cam_to_world,   # 카메라 위치 및 방향
            "intrinsics": intrinsic, # 카메라 렌즈 정보
            "conf": flat_conf,       # 신뢰도 점수
            "mask": mask,            # 유효 포인트 마스크
            "image_shape": (vggt_res, vggt_res)
        }

    def save_to_ply(self, data, save_path):
        """
        [파일 저장 단계]
        메모리 상의 3D 데이터(points, colors)를 .ply 파일로 디스크에 씁니다.
        이 파일이 최종적으로 사용자에게 다운로드됩니다.
        """
        print(f"PLY 파일 저장 경로: {save_path}")
        mask = data["mask"]
        pts = data["points"][mask]
        cls = data["colors"][mask]
        
        if len(pts) == 0:
            print("저장할 유효 포인트가 없습니다.")
            return

        # trimesh 라이브러리를 사용하여 포인트 클라우드 객체 생성 및 저장
        pcd = trimesh.PointCloud(vertices=pts, colors=cls)
        pcd.export(save_path)

    def start_visualization(self, data, port=8080):
        """
        [시각화 단계 (옵션)]
        개발자가 로컬에서 결과를 확인하거나, 디버깅할 때 사용합니다.
        웹 브라우저를 통해 3D 모델과 카메라 위치를 보여줍니다.
        """
        server = viser.ViserServer(host="0.0.0.0", port=port)
        mask = data["mask"]
        
        # 3D 점들을 뷰어에 추가
        server.scene.add_point_cloud(
            name="vggt_pcd",
            points=data["points"][mask],
            colors=data["colors"][mask],
            point_size=0.005
        )
        
        # 카메라의 위치와 보는 방향(Frustum)을 시각화
        H, W = data["image_shape"]
        for i, pose in enumerate(data["poses"]):
            T_world_cam = viser_tf.SE3.from_matrix(pose)
            # 화각(FOV) 계산하여 카메라 뿔 모양 생성
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