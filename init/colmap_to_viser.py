import viser
import pycolmap
import numpy as np
import time
from pathlib import Path

def visualize_colmap(checkpoint_path: str):
    server = viser.ViserServer()
    
    # 1. COLMAP 데이터 로드
    reconstruction = pycolmap.Reconstruction(checkpoint_path)
    
    # 2. 포인트 클라우드(Points3D) 시각화
    points = reconstruction.points3D
    points_coords = np.array([p.xyz for p in points.values()])
    points_colors = np.array([p.color for p in points.values()])
    
    server.add_point_cloud(
        name="/colmap/points",
        points=points_coords,
        colors=points_colors,
        point_size=0.02,
    )
    
    # 3. 카메라 및 프레임(Images) 시각화
    images = reconstruction.images
    cameras = reconstruction.cameras
    
    for img in images.values():
        cam = cameras[img.camera_id]
        
        # 카메라 포즈 (World-to-Camera -> Camera-to-World 변환)
        T_world_camera = img.cam_from_world.inverse()
        
        # Viser에 카메라 프러스트럼 추가
        server.add_camera_frustum(
            name=f"/colmap/cameras/{img.id}",
            fov=2 * np.arctan(cam.width / (2 * cam.mean_focal_length())),
            aspect=cam.width / cam.height,
            scale=0.15,
            image=None,  # 원할 경우 실제 이미지를 샘플링하여 부착 가능
            wxyz=T_world_camera.rotation.quaternion(),
            position=T_world_camera.translation,
        )

    print("Viser 서버가 실행 중입니다. 브라우저에서 확인하세요.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    # COLMAP sparse 결과가 있는 경로 (cameras.bin, images.bin 등이 있는 폴더)
    COLMAP_PATH = "/mnt/sdcard/klee_koreatech/init/processed_decbcd19-98d0-4623-a3e2-5cce1c812332/sparse/0" 
    visualize_colmap(COLMAP_PATH)