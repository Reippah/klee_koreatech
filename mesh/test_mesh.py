import os
import sys
import time
import numpy as np
import open3d as o3d

# =========================
# 경로 설정
# =========================
input_path = "/home/viclab/workspace/"
output_path = "/home/viclab/workspace/klee_koreatech/"
os.makedirs(output_path, exist_ok=True)

dataname = "demo_truck.ply"
ply_path = os.path.join(input_path, dataname)

# 출력 파일
out_mesh_path = os.path.join(output_path, "bpa_mesh.ply")

# =========================
# 유틸
# =========================
def now():
    return time.time()

def fmt_sec(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m}m {s:.1f}s"

def try_print_rss(prefix: str = ""):
    """
    psutil이 있으면 RSS 출력 (없으면 조용히 스킵)
    """
    try:
        import psutil, os as _os
        p = psutil.Process(_os.getpid())
        rss = p.memory_info().rss / (1024**3)
        print(f"{prefix}[MEM] RSS: {rss:.2f} GB")
    except Exception:
        pass

# =========================
# 전처리: Downsample + Outlier + Cap + Normals
# =========================
def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    diag_divisor: float = 800.0,
    max_points: int = 700_000,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
):
    """
    - bbox diag 기반 voxel downsample
    - statistical outlier 제거
    - 너무 많으면 uniform_down_sample로 cap
    - normals 없으면 estimate_normals
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())

    voxel_size = max(diag / diag_divisor, 1e-4)
    print(f"[INFO] bbox diag={diag:.6f}, tuned voxel_size={voxel_size:.6f}")

    t0 = now()
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[OK] Downsampled: {len(pcd_ds.points)} points")
    try_print_rss(prefix="       ")

    pcd_ds, _ = pcd_ds.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    print(f"[OK] Outliers removed: {len(pcd_ds.points)} points remain")

    n = len(pcd_ds.points)
    if n > max_points:
        every_k = int(np.ceil(n / max_points))
        pcd_ds = pcd_ds.uniform_down_sample(every_k_points=every_k)
        print(f"[OK] Uniform downsample cap: every_k={every_k} -> {len(pcd_ds.points)} points")

    print(f"[TIME] preprocess (downsample+outlier+cap): {fmt_sec(now() - t0)}")
    try_print_rss(prefix="       ")

    # Normal estimation
    if not pcd_ds.has_normals():
        normal_radius = voxel_size * 2.0
        max_nn = 20
        print(f"[INFO] Normal estimation params: radius={normal_radius:.6f}, max_nn={max_nn}")
        try_print_rss(prefix="       ")

        t0 = now()
        pcd_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=max_nn
            )
        )
        pcd_ds.normalize_normals()
        print(f"[TIME] estimate_normals: {fmt_sec(now() - t0)}")
        try_print_rss(prefix="       ")

    return pcd_ds, voxel_size

# =========================
# BPA (Ball Pivoting) 메쉬 생성
# =========================
def ball_pivoting_mesh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    out_path: str,
    radii_mul=(2, 3, 4, 6, 8, 10)  # 홀 감소를 위해 큰 공까지 포함
):
    radii_list = [voxel_size * m for m in radii_mul]
    radii = o3d.utility.DoubleVector(radii_list)

    print(f"[RUN] BPA radii_mul={list(radii_mul)}")
    print(f"[RUN] BPA radii={radii_list}")
    sys.stdout.flush()

    t0 = now()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    print(f"[TIME] BPA: {fmt_sec(now() - t0)}")
    try_print_rss(prefix="       ")

    # Cleanup
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices).shape[0]
    f = np.asarray(mesh.triangles).shape[0]
    print(f"[INFO] BPA mesh stats: vertices={v}, triangles={f}")

    # 연결 컴포넌트 수(조각이 너무 많으면 floaters/노이즈 가능성 ↑)
    tri_clusters, _, _ = mesh.cluster_connected_triangles()
    n_clusters = int(np.max(tri_clusters)) + 1 if len(tri_clusters) else 0
    print(f"[INFO] connected components: {n_clusters}")

    o3d.io.write_triangle_mesh(out_path, mesh)
    print("[SAVE] BPA mesh saved to:", out_path)
    return mesh

# =========================
# main
# =========================
def main():
    t_all = now()

    # 1) Load
    t0 = now()
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"Failed to load: {ply_path}")

    print(f"[OK] Loaded point cloud: {ply_path}")
    print(f" - points: {len(pcd.points)}")
    print(f" - has_normals: {pcd.has_normals()}")
    print(f" - has_colors : {pcd.has_colors()}")
    print(f"[TIME] read_point_cloud: {fmt_sec(now() - t0)}")
    try_print_rss(prefix="       ")

    # 2) Preprocess
    t0 = now()
    pcd_ds, voxel_size = preprocess_point_cloud(
        pcd,
        diag_divisor=800.0,
        max_points=700_000
    )
    print(f"[TIME] preprocess total: {fmt_sec(now() - t0)}")
    try_print_rss(prefix="       ")

    # 3) BPA
    mesh = ball_pivoting_mesh(
        pcd_ds,
        voxel_size,
        out_mesh_path,
        radii_mul=(2, 3, 4, 6, 8, 10)
    )

    # 4) Visualize
    o3d.visualization.draw_geometries([mesh])

    print(f"[DONE] Total elapsed: {fmt_sec(now() - t_all)}")

if __name__ == "__main__":
    main()
