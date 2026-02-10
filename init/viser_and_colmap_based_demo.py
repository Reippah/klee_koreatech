import os
import glob
import time
import torch
import torch.nn.functional as F
import numpy as np
import trimesh
import cv2
import urllib.request
from tqdm import tqdm

# ONNX Runtime for Sky Segmentation
try:
    import onnxruntime
except ImportError:
    print("[경고] onnxruntime이 설치되지 않았습니다. 하늘 제거 기능이 작동하지 않을 수 있습니다.")
    print("pip install onnxruntime-gpu (또는 onnxruntime)을 설치해주세요.")

# VGGT 모델 관련 모듈
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

class VGGTJetsonIntegrated:
    def __init__(self, model_url=None, device="cuda"):
        """
        [초기화 단계]
        VGGT 모델을 메모리에 로드합니다.
        """
        self.device = device
        self.dtype = torch.float16
        print(f"[{self.device}] VGGT 모델 로딩 시작...")
        
        self.model = VGGT()
        if model_url is None:
            model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        # VGGT 모델 가중치 로드
        if os.path.exists(model_url):
            state_dict = torch.load(model_url, map_location=self.device)
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=self.device)
            
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        
        # 하늘 제거 모델 경로 설정
        self.sky_model_path = "skyseg.onnx"
        self.sky_sess = None
        
        print("모델 로딩 및 초기화 완료.")

    def _ensure_sky_model(self):
        """하늘 제거용 ONNX 모델이 없으면 다운로드하고 세션을 로드합니다."""
        if not os.path.exists(self.sky_model_path):
            print("하늘 제거 모델(skyseg.onnx) 다운로드 중...")
            url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
            urllib.request.urlretrieve(url, self.sky_model_path)
            print("다운로드 완료.")

        if self.sky_sess is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            try:
                self.sky_sess = onnxruntime.InferenceSession(self.sky_model_path, providers=providers)
            except Exception as e:
                print(f"[오류] ONNX 세션 로드 실패: {e}")
                self.sky_sess = None

    def _predict_sky_mask(self, image_path, target_size):
        """단일 이미지에 대해 하늘 마스크를 생성합니다."""
        if self.sky_sess is None:
            return np.ones(target_size, dtype=np.float32)

        # 이미지 로드 및 전처리 (ONNX 모델 입력 규격에 맞춤)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 모델 추론용 리사이즈 (보통 1024x1024 등 고해상도 사용하나, 속도를 위해 512 정도 권장)
        infer_size = (512, 512) 
        img_resized = cv2.resize(img, infer_size)
        
        # 정규화 (Mean/Std는 일반적인 ImageNet 기준)
        img_float = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        
        # (H, W, C) -> (1, C, H, W)
        input_tensor = img_norm.transpose(2, 0, 1)[None, :, :, :]
        
        # ONNX 추론
        input_name = self.sky_sess.get_inputs()[0].name
        pred = self.sky_sess.run(None, {input_name: input_tensor})[0]
        
        # 결과 처리 (Sigmoid -> Threshold)
        mask = pred[0, 0] # (H, W)
        mask = 1.0 / (1.0 + np.exp(-mask)) # Sigmoid
        
        # VGGT 해상도에 맞게 리사이즈 (target_size: 518, 518)
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 하늘이 아닌 부분(0)을 1로, 하늘(1)을 0으로 뒤집음 (Confident Mask)
        # Threshold 0.5 기준: 값 > 0.5 이면 하늘
        non_sky_mask = (mask_resized < 0.5).astype(np.float32)
        
        return non_sky_mask

    @torch.no_grad()
    def process_scene(self, image_folder, mask_sky=True):
        """
        [추론 단계]
        이미지 폴더 -> 3D 포인트 클라우드 생성 (옵션: 하늘 제거)
        """
        # 0. 하늘 모델 준비 (필요 시)
        if mask_sky:
            self._ensure_sky_model()

        image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
        if not image_paths:
            raise ValueError(f"해당 경로에서 이미지를 찾을 수 없습니다: {image_folder}")

        vggt_res = 518   
        high_res = 1024  
        
        print(f"총 {len(image_paths)}장의 이미지 처리 시작...")
        
        # 1. 이미지 로드 (고해상도)
        images_hr, _ = load_and_preprocess_images_square(image_paths, high_res)
        images_hr = images_hr.to(self.device)
        
        # 2. 모델 입력용 리사이징
        images_vggt = F.interpolate(images_hr, size=(vggt_res, vggt_res), mode="bilinear")
        
        # 3. VGGT 추론 (Inference)
        print("VGGT 네트워크 추론 진행 중...")
        with torch.amp.autocast('cuda', dtype=self.dtype):
            predictions = self.model(images_vggt)

        # 4. 결과 파싱
        # (A) 카메라 파라미터
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (vggt_res, vggt_res))
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()

        # (B) 3D 포인트 좌표
        world_points = predictions["world_points"].squeeze(0).cpu().float().numpy() # (S, H, W, 3)
        world_conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy() # (S, H, W)
        
        # === [NEW] 하늘 제거 로직 적용 ===
        if mask_sky and self.sky_sess is not None:
            print("하늘 영역 제거(Sky Segmentation) 수행 중...")
            sky_masks = []
            # 각 이미지별로 마스크 생성
            for img_path in tqdm(image_paths, desc="Sky Masking"):
                mask = self._predict_sky_mask(img_path, target_size=(vggt_res, vggt_res))
                sky_masks.append(mask)
            
            # (S, H, W) 형태로 합침
            sky_masks = np.array(sky_masks)
            
            # 신뢰도 점수에 마스크 곱하기 (하늘 부분은 신뢰도 0이 됨)
            world_conf = world_conf * sky_masks
            print("하늘 영역 마스킹 완료.")
        # =================================

        # (C) 색상 정보 매핑
        input_hr_tensor = images_hr.squeeze(0)
        colors_hr_resized = F.interpolate(
            input_hr_tensor, 
            size=(vggt_res, vggt_res), 
            mode="bilinear", 
            align_corners=False
        )
        colors = colors_hr_resized.permute(0, 2, 3, 1).cpu().float().numpy()
        colors = np.clip(colors, 0, 1)
        flat_colors = (colors.reshape(-1, 3) * 255).astype(np.uint8)

        # 데이터 평탄화 (Flatten)
        flat_points = world_points.reshape(-1, 3)
        flat_conf = world_conf.reshape(-1)

        # 5. 후처리 및 필터링
        # 하늘이 제거되어 conf가 0인 부분은 여기서 자동으로 걸러짐
        threshold_val = np.percentile(flat_conf, 25.0)
        mask = (flat_conf >= threshold_val) & (flat_conf > 1e-5)

        cam_to_world = closed_form_inverse_se3(extrinsic)

        # 6. 장면 중심 정렬
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
        [저장 단계]
        결과 데이터를 .ply 파일로 저장
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