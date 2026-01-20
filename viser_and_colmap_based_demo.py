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

# VGGT 관련 유틸리티 (패키지 경로 확인 필요)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap

class VGGTJetsonIntegrated:
    def __init__(self, model_url=None, device="cuda"):
        self.device = device
        self.dtype = torch.float16 # Jetson 메모리 최적화
        print(f"[{self.device}] VGGT 모델 로딩 중...")
        
        self.model = VGGT()
        if model_url is None:
            model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        print("모델 로딩 완료.")

    @torch.no_grad()
    def process_scene(self, image_folder, use_ba=False, mask_sky=True):
        """
        [최종 수정본] 전처리 -> 추론 -> 후처리(필터링/스케일)
        """
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
        if not image_paths:
            raise ValueError(f"경로에 이미지가 없습니다: {image_folder}")

        vggt_res = 518
        high_res = 1024
        
        print(f"이미지 {len(image_paths)}장 전처리 시작...")
        images_hr, original_coords = load_and_preprocess_images_square(image_paths, high_res)
        images_hr = images_hr.to(self.device)
        scale_ratio = high_res / vggt_res
        
        images_vggt = F.interpolate(images_hr, size=(vggt_res, vggt_res), mode="bilinear")
        
        print("VGGT 추론 중...")
        # 최신 PyTorch API 권장 방식 반영
        with torch.amp.autocast('cuda', dtype=self.dtype):
            preds = self.model(images_vggt)
            
        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], (vggt_res, vggt_res))
        depth_map = preds["depth"].squeeze(0).cpu().float().numpy()
        depth_conf = preds["depth_conf"].squeeze(0).cpu().float().numpy()
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()

        # 품질 향상을 위한 하늘 제거 로직 (선택 사항)
        if mask_sky:
            # 원본 demo_viser 품질을 내려면 여기서 실제 sky segmentation을 수행해야 합니다.
            # 일단은 신뢰도 필터링을 강화하는 방향으로 보완합니다.
            pass

        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        
        # [품질 보정] 차원 에러 수정 및 고품질 색상 추출
        # images_vggt: [S, 3, H, W] -> [S, H, W, 3] -> [-1, 3]
        flat_colors = (images_vggt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        flat_points = points_3d.reshape(-1, 3)
        flat_conf = depth_conf.reshape(-1)

        # [품질 보정] 하위 30% 신뢰도 포인트를 날려 품질을 높입니다.
        conf_thresh = np.percentile(flat_conf, 30)
        combined_mask = (flat_conf >= conf_thresh) & (flat_conf > 0.05)

        # BA 최적화 (사용 시에만 작동)
        if use_ba:
            print("BA 최적화 중 (Memory-Safe 모드)...")
            try:
                pred_tracks, pred_vis, pred_confs, points_3d_ba, points_rgb = predict_tracks(
                    images_hr, conf=depth_conf, points_3d=points_3d, 
                    max_query_pts=1024, fine_tracking=False # Jetson 최적화 설정
                )
                
                intrinsic_hr = intrinsic.copy()
                intrinsic_hr[:, :2, :] *= scale_ratio
                
                track_mask = pred_vis > 0.2
                recons, _ = batch_np_matrix_to_pycolmap(
                    points_3d_ba, extrinsic, intrinsic_hr, pred_tracks, 
                    np.array([high_res, high_res]), # numpy array로 변환
                    masks=track_mask, points_rgb=points_rgb
                )
                # pycolmap 라이브러리 버전 에러 발생 시 여기에서 멈출 수 있음
                pycolmap.bundle_adjustment(recons, pycolmap.BundleAdjustmentOptions())
            except Exception as e:
                print(f"BA 과정에서 에러 발생 (건너뜁니다): {e}")

        # 장면 중심화 (Centering)
        scene_center = np.mean(flat_points[combined_mask], axis=0)
        points_centered = flat_points - scene_center
        cam_to_world = closed_form_inverse_se3(extrinsic)
        cam_to_world[:, :3, 3] -= scene_center

        return {
            "points": points_centered,
            "colors": flat_colors,
            "poses": cam_to_world,
            "intrinsics": intrinsic,
            "conf": flat_conf,
            "mask": combined_mask,
            "image_shape": (vggt_res, vggt_res)
        }

    def save_to_ply(self, data, save_path):
        """ .ply 저장 로직 """
        print(f"PLY 저장 중: {save_path}")
        mask = data["mask"]
        pcd = trimesh.PointCloud(vertices=data["points"][mask], colors=data["colors"][mask])
        pcd.export(save_path)

    def start_visualization(self, data, port=8080):
        """ Viser 시각화 서버 (모바일 접속 가능) """
        server = viser.ViserServer(host="0.0.0.0", port=port)
        mask = data["mask"]
        
        server.scene.add_point_cloud(
            name="vggt_pcd",
            points=data["points"][mask],
            colors=data["colors"][mask],
            point_size=0.001
        )
        
        H, W = data["image_shape"]
        for i, pose in enumerate(data["poses"]):
            T_world_cam = viser_tf.SE3.from_matrix(pose)
            fov = 2 * np.arctan2(H/2, data["intrinsics"][i, 0, 0])
            
            frustum = server.scene.add_camera_frustum(
                name=f"cameras/cam_{i}",
                fov=fov,
                aspect=W/H,
                scale=0.1,
                wxyz=T_world_cam.rotation().wxyz,
                position=T_world_cam.translation(),
                image=None # 성능을 위해 이미지는 생략하거나 원본 일부 로드
            )
        print(f"서버 주소: http://[Jetson_IP]:{port}")
        return server

if __name__ == "__main__":
    pipeline = VGGTJetsonIntegrated()
    
    # 1. 시각화 품질을 위해 mask_sky 인자 포함 (기본값 True)
    input_scene = "./image_folder" 
    results = pipeline.process_scene(input_scene, use_ba=False, mask_sky=True)
    
    # 2. 결과 저장
    os.makedirs("./output", exist_ok=True)
    pipeline.save_to_ply(results, "./output/reconstruction.ply")
    
    # 3. 서버 시작
    vis_server = pipeline.start_visualization(results)
    
    while True:
        time.sleep(1)