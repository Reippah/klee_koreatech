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

# VGGT 모델 관련 모듈
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

# [중요] demo_viser에서 사용하던 원본 하늘 제거 모듈 로드 시도
try:
    from visual_util import segment_sky
    HAS_VISUAL_UTIL = True
except ImportError:
    HAS_VISUAL_UTIL = False
    print("[정보] visual_util.py를 찾을 수 없어 내장된 하늘 제거 로직을 사용합니다.")

# ONNX Runtime (visual_util이 없을 경우를 대비한 백업)
if not HAS_VISUAL_UTIL:
    try:
        import onnxruntime
    except ImportError:
        onnxruntime = None
else:
    # visual_util 내부에서 onnxruntime을 쓸 테니 여기선 필요 없음
    onnxruntime = None


class VGGTJetsonIntegrated:
    def __init__(self, model_url=None, device="cuda"):
        self.device = device
        self.dtype = torch.float16
        print(f"[{self.device}] VGGT 모델 로딩 시작...")
        
        self.model = VGGT()
        if model_url is None:
            model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        if os.path.exists(model_url):
            state_dict = torch.load(model_url, map_location=self.device)
        else:
            state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=self.device)
            
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        
        # 하늘 제거 모델 경로
        self.sky_model_path = "skyseg.onnx"
        self.sky_sess = None
        
        print("모델 로딩 및 초기화 완료.")

    def _ensure_sky_model(self):
        """하늘 제거 모델 준비"""
        if not os.path.exists(self.sky_model_path):
            print("하늘 제거 모델(skyseg.onnx) 다운로드 중...")
            url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
            urllib.request.urlretrieve(url, self.sky_model_path)

        # 세션 로드 (visual_util이 있으면 그것이 로드하므로 패스, 없으면 직접 로드)
        if not HAS_VISUAL_UTIL and self.sky_sess is None and onnxruntime is not None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            try:
                self.sky_sess = onnxruntime.InferenceSession(self.sky_model_path, providers=providers)
            except Exception as e:
                print(f"[오류] ONNX 세션 로드 실패: {e}")

    def _get_sky_mask(self, image_path, target_size):
        """
        단일 이미지에 대한 하늘 마스크 생성 (0: 하늘/제거, 1: 유효)
        """
        # [Case 1] visual_util (demo_viser 원본 로직) 사용 - 가장 확실함
        if HAS_VISUAL_UTIL:
            # visual_util은 세션을 인자로 받거나 내부에서 처리함.
            # 여기서는 세션을 매번 로드하지 않게 demo_viser 방식을 흉내냄
            if self.sky_sess is None:
                # visual_util용 세션 생성 (import onnxruntime 필요할 수 있음)
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.sky_sess = ort.InferenceSession(self.sky_model_path, providers=providers)
            
            # 임시 파일 경로
            mask_path = image_path + ".sky_mask.png"
            
            # 원본 함수 호출
            # segment_sky(image_path, session, save_path)
            sky_mask = segment_sky(image_path, self.sky_sess, mask_path)
            
            # 결과: 보통 0(배경), 255(하늘) 또는 그 반대일 수 있음
            # demo_viser 로직: (mask > 0.1) -> keep
            # 원본 demo_viser가 conf * mask_binary를 했으므로,
            # mask가 "유효 영역(Ground)"이면 1이어야 함.
            
            # 리사이즈
            if sky_mask.shape[:2] != target_size:
                sky_mask = cv2.resize(sky_mask, target_size, interpolation=cv2.INTER_NEAREST)
            
            # demo_viser 로직 그대로 적용: > 0.1 이면 유효하다고 가정
            valid_mask = (sky_mask > 25).astype(np.float32) 
            
            # 임시 파일 정리
            if os.path.exists(mask_path):
                os.remove(mask_path)
                
            return valid_mask

        # [Case 2] 내장 로직 사용 (visual_util 없을 때)
        elif self.sky_sess is not None:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # [핵심 수정] 320x320 해상도 강제 (ONNX 모델 요구사항)
            infer_size = (320, 320)
            img_resized = cv2.resize(img, infer_size)
            
            img_float = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_float - mean) / std
            
            input_tensor = img_norm.transpose(2, 0, 1)[None, :, :, :]
            input_name = self.sky_sess.get_inputs()[0].name
            
            pred = self.sky_sess.run(None, {input_name: input_tensor})[0]
            mask = pred[0, 0] 
            mask = 1.0 / (1.0 + np.exp(-mask)) # Sigmoid
            
            # 리사이즈
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
            
            # [로직 확인] 보통 모델 출력값 High(1.0) = Sky
            # 우리는 Sky를 제거해야 하므로 < 0.5 인 부분을 1(유효)로 설정
            non_sky_mask = (mask_resized < 0.5).astype(np.float32)
            return non_sky_mask
        
        else:
            # 아무것도 없으면 마스킹 안함 (1.0 리턴)
            return np.ones(target_size, dtype=np.float32)

    @torch.no_grad()
    def process_scene(self, image_folder, use_ba=False, mask_sky=True):
        # 0. 하늘 모델 준비
        if mask_sky:
            self._ensure_sky_model()

        image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
        # .sky_mask.png 같은 임시 파일 제외
        image_paths = [p for p in image_paths if not p.endswith('.png')]
        
        if not image_paths:
            raise ValueError(f"이미지 없음: {image_folder}")

        vggt_res = 518   
        high_res = 1024  
        
        print(f"총 {len(image_paths)}장의 이미지 처리 시작...")
        
        # 1. 이미지 로드
        images_hr, _ = load_and_preprocess_images_square(image_paths, high_res)
        images_hr = images_hr.to(self.device)
        images_vggt = F.interpolate(images_hr, size=(vggt_res, vggt_res), mode="bilinear")
        
        # 2. VGGT 추론
        print("VGGT 추론 중...")
        with torch.amp.autocast('cuda', dtype=self.dtype):
            predictions = self.model(images_vggt)

        # 3. 결과 파싱
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (vggt_res, vggt_res))
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()

        world_points = predictions["world_points"].squeeze(0).cpu().float().numpy()
        world_conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy()
        
        # === 하늘 제거 ===
        if mask_sky:
            print("하늘 제거(Sky Segmentation) 적용 중...")
            sky_masks = []
            for img_path in tqdm(image_paths, desc="Sky Masking"):
                # 각 이미지별 마스크 생성
                mask = self._get_sky_mask(img_path, target_size=(vggt_res, vggt_res))
                sky_masks.append(mask)
            
            sky_masks = np.array(sky_masks)
            
            # [안전장치] 만약 마스크가 모든 점을 지워버렸다면(유효 0개), 마스킹 취소
            total_valid_pixels = np.sum(sky_masks)
            if total_valid_pixels < 100:
                print("[경고] 하늘 제거 결과 유효한 점이 거의 없습니다. 마스킹을 건너뜁니다.")
            else:
                world_conf = world_conf * sky_masks
                print("하늘 마스킹 완료.")
        # =================

        # 색상 매핑
        input_hr_tensor = images_hr.squeeze(0)
        colors_hr_resized = F.interpolate(input_hr_tensor, size=(vggt_res, vggt_res), mode="bilinear", align_corners=False)
        colors = colors_hr_resized.permute(0, 2, 3, 1).cpu().float().numpy()
        colors = np.clip(colors, 0, 1)
        flat_colors = (colors.reshape(-1, 3) * 255).astype(np.uint8)

        flat_points = world_points.reshape(-1, 3)
        flat_conf = world_conf.reshape(-1)

        threshold_val = np.percentile(flat_conf, 25.0)
        mask = (flat_conf >= threshold_val) & (flat_conf > 1e-5)

        cam_to_world = closed_form_inverse_se3(extrinsic)

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
        print(f"PLY 파일 저장: {save_path}")
        mask = data["mask"]
        pts = data["points"][mask]
        cls = data["colors"][mask]
        
        if len(pts) == 0:
            print("저장할 포인트가 없습니다.")
            return

        pcd = trimesh.PointCloud(vertices=pts, colors=cls)
        pcd.export(save_path)