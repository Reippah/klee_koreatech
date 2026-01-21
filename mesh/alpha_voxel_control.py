import os
import time
import csv
import itertools
import numpy as np
import open3d as o3d

# =========================
# 경로 설정
# =========================
input_path  = "/home/viclab/workspace/klee_koreatech/vggt/vggt_for_3dgs/sparse/0"
output_path = "/home/viclab/workspace/klee_koreatech/mesh_output/"
os.makedirs(output_path, exist_ok=True)

dataname  = "points3D.ply"
ply_path  = os.path.join(input_path, dataname)

# =========================
# 스윕 파라미터(여기만 바꾸면 됨)
# =========================
VOXEL_LIST = [0.0015, 0.0020, 0.0025, 0.0030]
ALPHA_LIST = [0.010,  0.015,  0.020,  0.030]

# 상위 N개만 최종 비교 시각화
TOP_N_VIS = 6

def now(): return time.time()

def rotate_fix_inversion(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Meshlab 등에서 뒤집혀 보이는 경우를 위해 중심 기준 X축 180도 회전"""
    center = pcd.get_center()
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=center)
    return pcd

def preprocess_pcd(pcd: o3d.geometry.PointCloud,
                   nb_neighbors: int = 40,
                   std_ratio: float = 1.0) -> o3d.geometry.PointCloud:
    """Outlier 제거(엄격)"""
    pcd2, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd2

def postprocess_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """메쉬 품질 정리 + 약한 스무딩"""
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=2)
    mesh.compute_vertex_normals()
    return mesh

def mesh_edge_stats(mesh: o3d.geometry.TriangleMesh):
    """
    경계 엣지/비정상(Non-manifold) 엣지 등을 '직접' 계산해서 지표로 씀.
    - boundary_edges: 삼각형에 1번만 등장한 엣지(=구멍 경계)
    - nonmanifold_edges: 3개 이상의 삼각형이 공유한 엣지(=비정상 연결)
    """
    if len(mesh.triangles) == 0:
        return {
            "edges_total": 0,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "boundary_ratio": 0.0,
        }

    tris = np.asarray(mesh.triangles, dtype=np.int64)

    # 삼각형의 3개 엣지(undirected) 생성
    e01 = tris[:, [0, 1]]
    e12 = tris[:, [1, 2]]
    e20 = tris[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])

    # undirected로 통일: (min, max)
    a = np.minimum(edges[:, 0], edges[:, 1])
    b = np.maximum(edges[:, 0], edges[:, 1])
    undirected = np.stack([a, b], axis=1)

    # 엣지 카운트
    undirected_view = undirected.view([("u", np.int64), ("v", np.int64)])
    uniq, counts = np.unique(undirected_view, return_counts=True)

    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts >= 3))
    total = int(len(uniq))
    ratio = (boundary / total) if total > 0 else 0.0

    return {
        "edges_total": total,
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "boundary_ratio": ratio,
    }

def reconstruct_alpha_mesh(pcd: o3d.geometry.PointCloud, voxel_size: float, alpha: float):
    """
    다운샘플 → Alpha Shape → 후처리 → 지표 계산
    """
    t0 = now()
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 포인트가 너무 적으면 실패/엉망 가능성 ↑
    if len(pcd_ds.points) < 1000:
        return None, {
            "time_s": now() - t0,
            "points_ds": len(pcd_ds.points),
            "verts": 0,
            "tris": 0,
            "edges_total": 0,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "boundary_ratio": 0.0,
            "watertight": False,
            "edge_manifold_no_boundary": False,
            "ok": False,
            "reason": "too_few_points_after_voxel",
        }

    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ds, alpha)
        mesh = postprocess_mesh(mesh)

        stats = mesh_edge_stats(mesh)

        # Open3D 제공 메타 지표(버전에 따라 동작이 다를 수 있음)
        try:
            watertight = bool(mesh.is_watertight())
        except Exception:
            watertight = False

        try:
            edge_manifold_no_boundary = bool(mesh.is_edge_manifold(allow_boundary_edges=False))
        except Exception:
            edge_manifold_no_boundary = False

        info = {
            "time_s": now() - t0,
            "points_ds": len(pcd_ds.points),
            "verts": int(len(mesh.vertices)),
            "tris": int(len(mesh.triangles)),
            **stats,
            "watertight": watertight,
            "edge_manifold_no_boundary": edge_manifold_no_boundary,
            "ok": True,
            "reason": "",
        }
        return mesh, info

    except Exception as e:
        return None, {
            "time_s": now() - t0,
            "points_ds": len(pcd_ds.points),
            "verts": 0,
            "tris": 0,
            "edges_total": 0,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "boundary_ratio": 0.0,
            "watertight": False,
            "edge_manifold_no_boundary": False,
            "ok": False,
            "reason": f"exception:{type(e).__name__}",
        }

def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str):
    o3d.io.write_triangle_mesh(path, mesh)

def visualize_side_by_side(mesh_items):
    """
    여러 메쉬를 x축으로 띄워서 한 화면에서 비교.
    (Open3D 기본 뷰어는 토글 UI가 불편해서 '나란히'가 제일 실용적임)
    """
    geoms = []
    x_cursor = 0.0
    gap = 0.2  # 모델 사이 간격(단위는 데이터 스케일에 맞게 조절)

    # bbox 폭에 따라 자동 간격
    for item in mesh_items:
        mesh = item["mesh"].copy()
        mesh.compute_vertex_normals()

        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        width = float(extent[0]) if extent is not None else 1.0
        shift = x_cursor - bbox.get_center()[0]

        mesh.translate((shift, 0.0, 0.0))
        geoms.append(mesh)

        x_cursor += width + gap

    print("\n[VIS] 나란히 배치된 순서(왼→오):")
    for i, item in enumerate(mesh_items, 1):
        print(f"  {i:02d}) voxel={item['voxel']:.6f}, alpha={item['alpha']:.6f} | "
              f"boundary={item['boundary_edges']} ({item['boundary_ratio']:.4f}), "
              f"nonmanifold={item['nonmanifold_edges']}, tris={item['tris']}, time={item['time_s']:.2f}s")

    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

def main():
    t_start = now()

    # 1) 파일 로드
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        print("[Error] point cloud가 비어있습니다. 경로를 확인하세요.")
        return
    print(f"[OK] Loaded: {len(pcd.points)} points")

    # 2) 좌표계 교정
    pcd = rotate_fix_inversion(pcd)
    print("[OK] Applied 180-degree rotation to fix inversion.")

    # 3) 전처리(Outlier 제거)
    pcd = preprocess_pcd(pcd, nb_neighbors=40, std_ratio=1.0)
    print(f"[OK] After outlier removal: {len(pcd.points)} points")

    # 4) 스윕 실행
    results = []
    csv_path = os.path.join(output_path, "alpha_sweep_report.csv")

    print("\n[RUN] Sweep 시작")
    print(f"      VOXEL_LIST={VOXEL_LIST}")
    print(f"      ALPHA_LIST={ALPHA_LIST}\n")

    for voxel_size, alpha in itertools.product(VOXEL_LIST, ALPHA_LIST):
        tag = f"vox{voxel_size:.6f}_a{alpha:.6f}"
        out_mesh_path = os.path.join(output_path, f"alpha_mesh_{tag}.ply")

        print(f"[CASE] {tag} ...", end=" ")

        mesh, info = reconstruct_alpha_mesh(pcd, voxel_size, alpha)

        row = {
            "voxel_size": voxel_size,
            "alpha": alpha,
            "mesh_path": out_mesh_path if info["ok"] else "",
            **info
        }

        if info["ok"] and mesh is not None and info["tris"] > 0:
            save_mesh(mesh, out_mesh_path)
            print(f"OK | tris={info['tris']} | boundary={info['boundary_edges']} ({info['boundary_ratio']:.4f}) | "
                  f"nonmanifold={info['nonmanifold_edges']} | {info['time_s']:.2f}s")
        else:
            print(f"FAIL ({info['reason']}) | {info['time_s']:.2f}s")

        results.append((mesh, row))

    # 5) CSV 저장
    fieldnames = [
        "voxel_size", "alpha", "mesh_path",
        "ok", "reason",
        "time_s", "points_ds",
        "verts", "tris",
        "edges_total", "boundary_edges", "boundary_ratio", "nonmanifold_edges",
        "watertight", "edge_manifold_no_boundary",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _, row in results:
            w.writerow(row)

    print(f"\n[SAVE] Report CSV: {csv_path}")

    # 6) 상위 후보 자동 선별 (구멍 적은 순 → 비정상 적은 순 → 삼각형 적당한 순 → 시간)
    ok_items = []
    for mesh, row in results:
        if row["ok"] and mesh is not None and row["tris"] > 0:
            ok_items.append((mesh, row))

    if not ok_items:
        print("[WARN] 성공한 메쉬가 없습니다. VOXEL_LIST/ALPHA_LIST 범위를 조정해보세요.")
        return

    def score_key(item):
        _, r = item
        # 1) boundary_ratio 최소
        # 2) boundary_edges 최소
        # 3) nonmanifold_edges 최소
        # 4) tris 너무 많지 않게(가벼운 쪽 선호) → 필요하면 반대로 바꿔도 됨
        # 5) time 최소
        return (r["boundary_ratio"], r["boundary_edges"], r["nonmanifold_edges"], r["tris"], r["time_s"])

    ok_items.sort(key=score_key)

    top = ok_items[:min(TOP_N_VIS, len(ok_items))]

    print(f"\n[TOP] 자동 선별 상위 {len(top)}개(나란히 비교용):")
    for i, (mesh, row) in enumerate(top, 1):
        print(f"  {i:02d}) voxel={row['voxel_size']:.6f}, alpha={row['alpha']:.6f} | "
              f"boundary={row['boundary_edges']} ({row['boundary_ratio']:.4f}), "
              f"nonmanifold={row['nonmanifold_edges']}, tris={row['tris']}, time={row['time_s']:.2f}s | "
              f"file={row['mesh_path']}")

    # 7) 나란히 시각화
    vis_items = []
    for mesh, row in top:
        vis_items.append({
            "mesh": mesh,
            "voxel": row["voxel_size"],
            "alpha": row["alpha"],
            "boundary_edges": row["boundary_edges"],
            "boundary_ratio": row["boundary_ratio"],
            "nonmanifold_edges": row["nonmanifold_edges"],
            "tris": row["tris"],
            "time_s": row["time_s"],
        })

    visualize_side_by_side(vis_items)

    print(f"\n[DONE] Total sweep time: {now() - t_start:.2f}s")

if __name__ == "__main__":
    main()


