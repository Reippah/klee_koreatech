import os
import time
import numpy as np
import open3d as o3d

# =========================
# 경로 설정
# =========================
input_path = "/mnt/sdcard/klee_koreatech/init/result/processed_22fd5f31-4eee-4ae4-98a9-491d841dd8dd"
output_path = "/home/viclab/workspace/klee_koreatech/mesh_output/"
os.makedirs(output_path, exist_ok=True)

dataname = "reconstruction.ply"
ply_path = os.path.join(input_path, dataname)
out_mesh_path = os.path.join(output_path, "alpha_mesh_final.ply")

def now(): return time.time()

def main():
    start_t = now()

    # 1. 파일 로드
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        print("[Error] 파일을 찾을 수 없습니다."); return
    print(f"[OK] Loaded: {len(pcd.points)} points")

    # 2. 좌표계 교정 (Meshlab 뒤집힘 문제 해결)
    # 데이터 중심을 기준으로 X축 180도 회전
    center = pcd.get_center()
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=center)
    print("[OK] Applied 180-degree rotation to fix inversion.")

    # 3. 전처리 (에러 유발 노이즈 제거)
    # Poisson 에러의 주원인이 되는 부유물(Outliers)을 엄격하게 제거합니다.
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
    
    # 균일한 면 생성을 위해 다운샘플링
    voxel_size = 0.002
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[OK] Downsampled to {len(pcd_ds.points)} points.")

    # 4. Alpha Shapes Reconstruction (에러 발생 없음)
    print("[RUN] Generating Alpha Shape Mesh...")
    
    # Alpha 파라미터 설정 (이 값이 클수록 구멍이 더 잘 메워집니다)
    # 0.01 ~ 0.05 사이에서 조절해 보세요.
    alpha = 0.015
    
    # create_from_point_cloud_alpha_shape는 Poisson과 달리 'Failed to close loop'가 뜨지 않습니다.
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ds, alpha)
    
    # 5. 후처리 (메쉬 품질 향상)
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    
    # 표면을 매끄럽게 다듬기 (구멍 주변 경계 완화)
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=2)
    
    # 6. 저장 및 시각화
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    print(f"[SAVE] Final mesh saved to: {out_mesh_path}")
    print(f"[DONE] Total time: {now() - start_t:.2f}s")

    # 결과 확인 (뒷면도 보이도록 설정)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

if __name__ == "__main__":
    main()