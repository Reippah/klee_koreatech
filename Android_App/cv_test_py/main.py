"""
광택/빛반사 제거 알고리즘 v4.4  (미세 반사 감지 + 텍스처 복원 강화)
==================================================================
OpenCV + K-Means + FastReflectionRemoval(FRR) + 분류기 + Dichromatic + Texture Restore

[v4.4 핵심 개선]
  문제 ① 작은 빛반사 미처리 (자잘한 광택 잔존)
  해결  : Dichromatic(표면 반사) 채널 재활성화 + 소면적 기준 완화(min_area=6)
         → 아주 작은 반사 스팟까지 감지

  문제 ② 광택 제거 후 단색으로 뭉개짐 (Inpainting 이질감)
  해결  : Texture+Color 하이브리드 복원 (restore_with_texture)
         → 색상(Color): Inpainting으로 주변 색 확산
         → 질감(Texture): FRR 결과에서 반사가 제거된 원본 텍스처 추출
         → Luma(밝기)는 FRR, Chroma(색상)는 Inpainting 결과를 합성하여 자연스러운 표면 유지

처리 파이프라인 (v4.4):
  [감지]   Gray×HSV core + Local + K-Means + Dichromatic(재활성화)
  [분류]   소면적(area>=6) 허용 + 구배 조건 완화
  [보호]   엣지 기반 로고 보호 (LAB 중립색 체크 포함)
  [확장]   로컬 ROI 기반 영역 확장 (Hotspot 내부 채움)
  [FRR]    전역 25% 블렌딩 (배경 반사 완화)
  [복원]   Texture+Color 하이브리드 복원 (마스크 영역 텍스처 살림) ← v4.4 핵심
  [후처리] 바이래터럴 + 가우시안 블렌딩
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from frr import FastReflectionRemoval
import os
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 감지 모듈 (Detection)
# ══════════════════════════════════════════════════════════════

def detect_gray_otsu(image, fixed_floor=220):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh  = max(otsu, fixed_floor)
    _, mask = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    return mask, thresh


def detect_hsv_specular(image, v_thresh=240, s_thresh=18):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    _, mv = cv2.threshold(v, v_thresh, 255, cv2.THRESH_BINARY)
    _, ms = cv2.threshold(s, s_thresh, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(mv, ms)


def detect_local_contrast(image, window=61, excess=30, abs_min=170):
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_avg = cv2.GaussianBlur(gray, (window | 1, window | 1), 0)
    diff      = gray - local_avg
    _, md = cv2.threshold(diff,              excess,  255, cv2.THRESH_BINARY)
    _, ma = cv2.threshold(gray.astype(np.uint8), abs_min, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(md.astype(np.uint8), ma)


def detect_surface_specular(image, erode_size=41, specular_excess=22):
    """
    [v4.4 재활성화] 이색 반사 모델(Dichromatic Reflection Model).
    미세한 표면 반사(작은 광택)를 잡는 데 효과적.
    """
    v_ch   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    v_mat  = cv2.erode(v_ch, k)
    v_spec = np.clip(v_ch - v_mat, 0, 255)
    _, mask = cv2.threshold(v_spec, specular_excess, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def detect_kmeans(image, n_clusters=8, chroma_thresh=18, top_pct=94):
    print("    · K-Means 클러스터링 실행 중...")
    h, w  = image.shape[:2]
    scale = min(1.0, 500.0 / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)))
    lab   = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    pix   = lab.reshape(-1, 3).astype(np.float32)

    km      = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels  = km.fit_predict(pix)
    centers = km.cluster_centers_
    l_thr   = np.percentile(pix[:, 0], top_pct)

    ids = {
        i for i, c in enumerate(centers)
        if c[0] >= l_thr
        and abs(float(c[1]) - 128) < chroma_thresh
        and abs(float(c[2]) - 128) < chroma_thresh
    }
    if not ids:
        ids = {int(np.argmax(centers[:, 0]))}

    ms = np.zeros(len(pix), dtype=np.uint8)
    for cid in ids:
        ms[labels == cid] = 255
    ms = ms.reshape(small.shape[:2])
    return cv2.resize(ms, (w, h), interpolation=cv2.INTER_NEAREST)


def build_soft_specular_constraint(image, local_window=61, diff_thresh=18, v_min=80, s_max=120):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    local_avg = cv2.GaussianBlur(gray, (local_window | 1, local_window | 1), 0)
    diff = gray - local_avg

    c1 = (diff > float(diff_thresh)) & (v >= float(v_min)) & (s <= float(s_max))
    c2 = (v >= 215) & (s <= 140)
    constraint = (c1 | c2).astype(np.uint8) * 255

    constraint = cv2.morphologyEx(
        constraint,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    return constraint


def grow_mask_by_constraint(seed_mask, constraint_mask, grow_kernel=5, max_iter=40):
    cur = (seed_mask > 0).astype(np.uint8) * 255
    constraint = (constraint_mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (grow_kernel | 1, grow_kernel | 1)
    )
    for _ in range(int(max_iter)):
        nxt = cv2.bitwise_and(cv2.dilate(cur, k, iterations=1), constraint)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur


def grow_mask_locally(
    seed_mask, constraint_mask, image, logo_mask=None,
    pad=45, grow_kernel=5, max_iter=30, v_drop=25, v_floor=150
):
    seed = (seed_mask > 0).astype(np.uint8) * 255
    constraint = (constraint_mask > 0).astype(np.uint8) * 255
    h, w = seed.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    v_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)

    cnts, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # [v4.4] 아주 작은 시드도 확장 허용 (min_area 완화 대응)
        if cv2.contourArea(c) < 2:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)

        s_roi = seed[y0:y1, x0:x1]
        c_roi = constraint[y0:y1, x0:x1]
        v_roi = v_full[y0:y1, x0:x1]

        sv = v_roi[s_roi > 0]
        if sv.size > 0:
            seed_mu = float(np.mean(sv))
            thr_v = max(float(v_floor), seed_mu - float(v_drop))
            c_roi = cv2.bitwise_and(c_roi, (v_roi >= thr_v).astype(np.uint8) * 255)
        
        if logo_mask is not None:
            lm = (logo_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
            c_roi = cv2.bitwise_and(c_roi, cv2.bitwise_not(lm))

        g_roi = grow_mask_by_constraint(s_roi, c_roi, grow_kernel=grow_kernel, max_iter=max_iter)
        g_roi = fill_holes(g_roi)
        out[y0:y1, x0:x1] = cv2.bitwise_or(out[y0:y1, x0:x1], g_roi)

    return out


def fill_holes(mask):
    m = (mask > 0).astype(np.uint8) * 255
    inv = cv2.bitwise_not(m)
    ff = inv.copy()
    h, w = ff.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(m, holes)


# ══════════════════════════════════════════════════════════════
# 컨투어 분류기
# ══════════════════════════════════════════════════════════════

def classify_specular_vs_white(
    raw_mask, image,
    surface_mask=None,       # [v4.4] Dichromatic 결과 마스크
    edge_sharp_thresh  = 55,
    circularity_thresh = 0.35,
    elongation_thresh  = 4.0,
    inner_var_thresh   = 600,
    v_contrast_thresh  = 20,
    radial_grad_thresh = 10,
    border_radius      = 20,
    min_area           = 2,      # [v4.4] 6->2 극한으로 낮춤
    max_area_ratio     = 0.008,
):
    _, s_ch, v_ch = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = image.shape[:2]
    max_area = h_img * w_img * max_area_ratio

    g32 = gray.astype(np.float32)
    sx  = cv2.Sobel(g32, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(g32, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.sqrt(sx ** 2 + sy ** 2)

    g_blur = cv2.GaussianBlur(g32, (5, 5), 1.2)
    sx_b   = cv2.Sobel(g_blur, cv2.CV_32F, 1, 0, ksize=3)
    sy_b   = cv2.Sobel(g_blur, cv2.CV_32F, 0, 1, ksize=3)
    sob_b  = np.sqrt(sx_b ** 2 + sy_b ** 2)

    kc     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kc)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    specular_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    logo_cnts     = []

    stats = {
        "total":          len(contours),
        "kept":           0,
        "rm_size":        0,
        "rm_sharp_irreg": 0,
        "rm_elongated":   0,
        "rm_circ_logo":   0,
        "rm_uniform":     0,
        "rm_score":       0,
    }

    ke   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ke2  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ke_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kb   = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (border_radius * 2 + 1, border_radius * 2 + 1)
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            stats["rm_size"] += 1
            continue

        cm = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, cv2.FILLED)

        # [v4.4 패스트트랙] Dichromatic 채널에서 감지된 영역은
        # 물리적으로 '표면 반사'일 확률이 매우 높으므로, 기하학적 조건 없이 통과시킨다.
        if surface_mask is not None:
            overlap = cv2.bitwise_and(cm, surface_mask)
            overlap_px = cv2.countNonZero(overlap)
            # 컨투어 면적의 30% 이상이 surface_mask와 겹치면 통과
            if overlap_px > area * 0.3:
                cv2.drawContours(specular_mask, [cnt], -1, 255, cv2.FILLED)
                stats["kept"] += 1
                continue

        # ── 특징 추출 ────────────────────────────────────
        perim       = cv2.arcLength(cnt, True)
        circ        = 4 * np.pi * area / (perim ** 2 + 1e-6)
        is_circular = circ > circularity_thresh

        _, _, bw, bh = cv2.boundingRect(cnt)
        aspect   = max(bw, bh) / (min(bw, bh) + 1e-6)
        is_elong = aspect > elongation_thresh

        outer_ring = cv2.subtract(cv2.dilate(cm, ke), cm)
        edge_vals  = sob[outer_ring > 0]
        mean_edge  = float(np.mean(edge_vals)) if edge_vals.size > 0 else 0.0
        is_sharp   = mean_edge > edge_sharp_thresh

        edge_vals_b = sob_b[outer_ring > 0]
        mean_edge_b = float(np.mean(edge_vals_b)) if edge_vals_b.size > 0 else 1e-6
        edge_ratio  = mean_edge / (mean_edge_b + 1e-6)
        is_very_sharp = (edge_ratio > 1.55) and (mean_edge > edge_sharp_thresh * 0.6)

        iv        = v_ch[cm > 0].astype(np.float32)
        inner_var = float(np.var(iv))  if iv.size > 0 else 9999.0
        inner_mu  = float(np.mean(iv)) if iv.size > 0 else 0.0
        is_smooth = inner_var < inner_var_thresh

        border_ring   = cv2.subtract(cv2.dilate(cm, kb), cm)
        bv = v_ch[border_ring > 0]
        if bv.size == 0:
            stats["rm_size"] += 1
            continue
        border_mu     = float(np.mean(bv))
        has_contrast  = (inner_mu - border_mu) > v_contrast_thresh

        # ══════════════════════════════════════════════════
        # 2단계 분류
        # ══════════════════════════════════════════════════

        if area < 500:
            # ── 소면적 분류 ──────────────────────────────────
            if is_elong:
                stats["rm_elongated"] += 1
                continue

            # [v4.4 수정] 미세 반사(점박이)는 날카로울 수 있으나
            # 로고처럼 '매우' 날카롭고(ratio>1.55) 내부가 어두운 경우만 제거
            if is_very_sharp and inner_mu < 210: # 밝은 반사 스팟은 날카로워도 허용
                stats["rm_sharp_irreg"] += 1
                logo_cnts.append(cnt)
                continue

            # 배경 오탐 방지
            if inner_mu >= 220 and border_mu >= 165:
                stats["rm_score"] += 1
                continue

            # [핵심] 미세 반사 허용: 대비가 확실하면 모양/구배 따지지 않고 통과
            if has_contrast:
                cv2.drawContours(specular_mask, [cnt], -1, 255, cv2.FILLED)
                stats["kept"] += 1
            else:
                stats["rm_score"] += 1
            continue

        # ── 대면적 분류 ────────────────────────────────────
        inner_core = cv2.erode(cm, ke2, iterations=1)
        inner_ring = cv2.subtract(cm, inner_core)
        core_v = v_ch[inner_core > 0]
        ring_v = v_ch[inner_ring  > 0]

        if core_v.size > 0 and ring_v.size > 0:
            radial_grad = float(np.mean(core_v)) - float(np.mean(ring_v))
        else:
            radial_grad = 0.0

        has_radial_grad = radial_grad > radial_grad_thresh

        if is_very_sharp and not is_circular:
            stats["rm_sharp_irreg"] += 1
            logo_cnts.append(cnt)
            continue

        if is_elong:
            stats["rm_elongated"] += 1
            continue

        if is_very_sharp and is_circular and not has_radial_grad:
            stats["rm_circ_logo"] += 1
            logo_cnts.append(cnt)
            continue

        if not is_sharp and not has_radial_grad:
            stats["rm_uniform"] += 1
            continue

        score = 0
        if not is_sharp:
            score += 2
            if is_smooth:  score += 1
        if is_circular:     score += 1
        if has_contrast:    score += 2
        if not is_elong:    score += 1
        if has_radial_grad: score += 2

        if score < 5:
            stats["rm_score"] += 1
            continue

        cv2.drawContours(specular_mask, [cnt], -1, 255, cv2.FILLED)
        stats["kept"] += 1

    return specular_mask, logo_cnts, stats


# ══════════════════════════════════════════════════════════════
# 보호 마스크 생성
# ══════════════════════════════════════════════════════════════

def build_logo_mask(logo_cnts, image_shape, buffer_px=8):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if logo_cnts:
        cv2.drawContours(mask, logo_cnts, -1, 255, cv2.FILLED)
        k    = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1)
        )
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def build_edge_protect_mask(
    image,
    v_min=195,
    s_max=90,
    canny1=35,
    canny2=110,
    edge_dilate_px=3,
    max_area_ratio=0.02,
):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16)
    b = lab[:, :, 2].astype(np.int16)
    neutral = (np.abs(a - 128) <= 22) & (np.abs(b - 128) <= 22)

    whiteish = (((v >= v_min) & neutral) | ((v >= 245) & (s <= 180))).astype(np.uint8) * 255
    whiteish = cv2.morphologyEx(
        whiteish,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    whiteish_dil = cv2.dilate(
        whiteish,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g_blur, canny1, canny2)

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (edge_dilate_px * 2 + 1, edge_dilate_px * 2 + 1)
    )
    edges = cv2.dilate(edges, k, iterations=1)

    seed = cv2.bitwise_and(edges, whiteish_dil)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    protect = np.zeros_like(seed)
    h, w = seed.shape[:2]
    max_area = h * w * float(max_area_ratio)

    cnts, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if 30 <= a <= max_area:
            cv2.drawContours(protect, [c], -1, 255, cv2.FILLED)

    gray_f = gray.astype(np.float32)
    sx  = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.sqrt(sx ** 2 + sy ** 2)
    g_blur2 = cv2.GaussianBlur(gray_f, (5, 5), 1.2)
    sx_b = cv2.Sobel(g_blur2, cv2.CV_32F, 1, 0, ksize=3)
    sy_b = cv2.Sobel(g_blur2, cv2.CV_32F, 0, 1, ksize=3)
    sob_b = np.sqrt(sx_b ** 2 + sy_b ** 2)

    cnts_w, _ = cv2.findContours(whiteish, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for c in cnts_w:
        a = cv2.contourArea(c)
        if not (40 <= a <= max_area):
            continue
        cm = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cm, [c], -1, 255, cv2.FILLED)
        outer = cv2.subtract(cv2.dilate(cm, ke), cm)
        ev = sob[outer > 0]
        eb = sob_b[outer > 0]
        if ev.size == 0 or eb.size == 0:
            continue
        mean_e = float(np.mean(ev))
        mean_b = float(np.mean(eb)) + 1e-6
        ratio = mean_e / mean_b
        if ratio > 1.45 and mean_e > 20:
            cv2.drawContours(protect, [c], -1, 255, cv2.FILLED)

    return protect


def dilate_mask(mask, dilate_size=7):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    return cv2.dilate(mask, k, iterations=2)


# ══════════════════════════════════════════════════════════════
# FRR & 복원 모듈
# ══════════════════════════════════════════════════════════════

def apply_frr(image, h_param=0.05):
    img_f    = image.astype(np.float32) / 255.0
    remover  = FastReflectionRemoval(h=h_param)
    result_f = np.clip(remover.remove_reflection(img_f), 0.0, 1.0)
    return (result_f * 255.0).astype(np.uint8)


def blend_frr_global(image, frr_result, logo_mask, frr_weight=0.25):
    blended = cv2.addWeighted(
        image.astype(np.float32),      1.0 - frr_weight,
        frr_result.astype(np.float32), frr_weight,
        0
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    if logo_mask is not None and np.any(logo_mask):
        blended[logo_mask > 0] = image[logo_mask > 0]
    return blended


def restore_with_texture(image, mask, frr_result):
    """
    [v4.4 신규] Texture+Color 하이브리드 복원.
    단순 Inpainting은 텍스처를 뭉개버려 부자연스러움.
    
    해결:
      1. Chroma (색상/채도): Inpainting 결과 사용 (주변 색 확산)
      2. Luma (명도/질감): FRR 결과 사용 (반사가 제거된 원본 텍스처)
      
    Returns:
      restored_image
    """
    # 1. 색상 복원 (Inpainting) - 텍스처는 뭉개지지만 색은 자연스러움
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    # LAB 공간 변환
    lab_inp = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    l_inp, a_inp, b_inp = cv2.split(lab_inp)
    
    lab_frr = cv2.cvtColor(frr_result, cv2.COLOR_BGR2LAB)
    l_frr, _, _ = cv2.split(lab_frr)
    
    # 2. 합성: FRR의 Luma(텍스처) + Inpainting의 Chroma(색상)
    #    단, FRR의 밝기가 너무 어두워지거나 밝아질 수 있으므로
    #    Inpainting 밝기(l_inp)를 가이드로 삼아 FRR 밝기(l_frr)를 조정할 수도 있음.
    #    여기서는 FRR Luma를 그대로 사용하여 텍스처를 최대한 살림.
    
    merged = cv2.merge([l_frr, a_inp, b_inp])
    restored = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # 마스크 영역만 교체 (부드러운 블렌딩)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 0)
    mask_3 = np.stack([mask_f] * 3, axis=2)
    
    final = image.astype(np.float32) * (1.0 - mask_3) + restored.astype(np.float32) * mask_3
    return np.clip(final, 0, 255).astype(np.uint8)


def post_process(inpainted, base_image, mask):
    """
    후처리: 바이래터럴 필터 + 감마 보정.
    v4.4: Texture 복원이 이미 되었으므로 가우시안 블렌딩은 restore_with_texture 내에서 처리.
          여기서는 전체적인 톤 정리만 수행.
    """
    smooth = cv2.bilateralFilter(inpainted, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 마스크 영역만 살짝 스무딩 (노이즈 제거)
    mask_f  = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), 0)
    mask_3  = np.stack([mask_f] * 3, axis=2)
    
    blended = np.clip(
        base_image.astype(np.float32) * (1.0 - mask_3) +
        smooth.astype(np.float32)     * mask_3,
        0, 255
    ).astype(np.uint8)
    
    lut = np.array([((i / 255.0) ** 0.95) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(blended, lut)


def create_comparison(original, refined_mask, frr_global, final):
    h, w    = original.shape[:2]
    overlay = original.copy()
    overlay[refined_mask > 0] = [0, 0, 220]

    comp = np.hstack([original, overlay, frr_global, final])
    fs   = max(0.55, w / 950)
    th   = max(1, int(w / 550))
    pad  = max(35, int(h * 0.04))

    for i, lbl in enumerate(
        ["Original (원본)", "Detected (감지)", "FRR Global (전역)", "Final (복원)"]
    ):
        cv2.putText(comp, lbl, (w * i + 10, pad),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 230, 0), th, cv2.LINE_AA)
    return comp


# ══════════════════════════════════════════════════════════════
# 메인 처리 함수
# ══════════════════════════════════════════════════════════════

def process_image(image_path):
    sep = "=" * 62
    print(f"\n{sep}\n  처리 파일: {os.path.basename(image_path)}\n{sep}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"  [오류] 로드 실패: {image_path}")
        return None, None

    h, w = image.shape[:2]
    print(f"  이미지 크기: {w} × {h}")

    # ── [감지] 6채널 마스크 ──────────────────────────────────
    print("\n  [감지] 6채널 빛반사 마스크 생성...")

    mask_gray, thresh_val = detect_gray_otsu(image, fixed_floor=220)
    print(f"    · Gray  Otsu 임계값: {thresh_val:.0f}")

    mask_hsv  = detect_hsv_specular(image, v_thresh=240, s_thresh=18)
    print(f"    · HSV   (V>240, S<18) 완료")

    mask_loc  = detect_local_contrast(image, window=61, excess=30, abs_min=170)
    print(f"    · Local 대비 (excess≥30) 완료")

    mask_surf = detect_surface_specular(image, erode_size=41, specular_excess=22)
    print(f"    · Dichromatic 표면 반사 (미세 광택) 완료  ← v4.4 재활성화")

    mask_km   = detect_kmeans(image)
    print(f"    · K-Means 완료")

    core = cv2.bitwise_and(mask_gray, mask_hsv)
    raw  = core.copy()
    for m in [mask_loc, mask_surf, mask_km]:  # mask_surf 포함
        raw = cv2.bitwise_or(raw, m)

    # ── [분류] 빛반사 vs 흰색 객체 ──────────────────────────
    print("\n  [분류] 컨투어별 빛반사 / 흰색 객체 구분...")
    # mask_surf(Dichromatic 결과)를 분류기에 전달하여 패스트트랙으로 사용
    specular_mask, logo_cnts, stats = classify_specular_vs_white(
        raw, image,
        surface_mask       = mask_surf,  # [v4.4] 패스트트랙용 마스크 추가
        edge_sharp_thresh  = 55,
        circularity_thresh = 0.35,
        elongation_thresh  = 4.0,
        inner_var_thresh   = 600,
        v_contrast_thresh  = 20,
        radial_grad_thresh = 10,
        border_radius      = 20,
        min_area           = 2,      # [v4.4] 6->2 극한으로 낮춤 (점박이 반사)
        max_area_ratio     = 0.008,
    )
    print(f"    · 전체 컨투어      : {stats['total']:>5}개")
    print(f"    · [유지] 빛반사    : {stats['kept']:>5}개")
    print(f"    · [제거] 크기      : {stats['rm_size']:>5}개")
    print(f"    · [제거] 텍스트/로고: {stats['rm_sharp_irreg']:>4}개  ← Rule A")
    print(f"    · [제거] 나뭇결    : {stats['rm_elongated']:>5}개  ← Rule B")
    print(f"    · [제거] 원형로고  : {stats['rm_circ_logo']:>5}개  ← Rule C")
    print(f"    · [제거] 균일면    : {stats['rm_uniform']:>5}개  ← Rule D")
    print(f"    · [제거] 점수미달  : {stats['rm_score']:>5}개")
    print(f"    · 로고 보호 컨투어 : {len(logo_cnts):>5}개")

    # ── [로고 보호 마스크] ───────────────────────────────────
    print("\n  [보호] 로고/텍스트 보호 마스크 생성...")
    logo_mask_cnt = build_logo_mask(logo_cnts, image.shape, buffer_px=8)
    logo_mask_edge = build_edge_protect_mask(image)
    logo_mask = cv2.bitwise_or(logo_mask_cnt, logo_mask_edge)
    logo_px   = int(np.sum(logo_mask > 0))
    print(f"    · 보호 구역: {logo_px:,} px")

    # ── [영역확장] 경계선만 잡히는 문제 해결 ────────────────
    print("\n  [확장] 광택 내부 영역 확장 (Local ROI)...")
    seed = cv2.morphologyEx(
        specular_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), # v4.4: 5->3 미세 반사 보존
        iterations=1,
    )
    constraint = build_soft_specular_constraint(
        image,
        local_window=61,
        diff_thresh=12,  # v4.4: 18->12 더 넓게 허용
        v_min=70,
        s_max=140,
    )
    grown = grow_mask_locally(
        seed, constraint, image,
        logo_mask=logo_mask,
        pad=45,
        grow_kernel=5,
        max_iter=28,
        v_drop=30,       # v4.4: 조금 더 관대하게
        v_floor=130,     # v4.4: 알루미늄 등 중간 밝기 반사 허용
    )
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # ── [팽창] ──────────────────────────────────────────────
    refined = dilate_mask(grown, dilate_size=7)
    refined = cv2.bitwise_and(refined, cv2.bitwise_not(logo_mask))

    px    = int(np.sum(refined > 0))
    ratio = px / (h * w) * 100
    print(f"\n    → 최종 빛반사 픽셀: {px:,} ({ratio:.2f}%)")

    # ── [FRR] ───────────────────────────────────────────────
    print("\n  [FRR] FastReflectionRemoval 전역 반사 억제...")
    frr_result = apply_frr(image, h_param=0.05)
    frr_global = blend_frr_global(image, frr_result, logo_mask, frr_weight=0.25)

    # ── [복원] Texture+Color 하이브리드 ──────────────────────
    print("\n  [복원] Texture(FRR) + Color(Inpaint) 하이브리드 복원...")
    # 전역 블렌딩된 이미지가 아니라 원본 기반으로 작업하여 텍스처를 살림
    # 단, FRR 비중을 높인 이미지를 텍스처 소스로 사용
    frr_source = blend_frr_global(image, frr_result, logo_mask, frr_weight=0.85)
    
    restored = restore_with_texture(image, refined, frr_source)
    print(f"    · 텍스처 보존 복원 완료")

    # ── [후처리] ─────────────────────────────────────────────
    print("\n  [후처리] 바이래터럴 + 감마 보정...")
    final = post_process(restored, image, refined) # base_image=image (원본에 복원된 부분 합성)

    if np.any(logo_mask):
        final[logo_mask > 0] = image[logo_mask > 0]

    # ── 저장 ─────────────────────────────────────────────────
    base    = os.path.splitext(image_path)[0]
    overlay = image.copy()
    overlay[refined > 0] = [0, 0, 220]
    overlay_with_logo = overlay.copy()
    if np.any(logo_mask):
        logo_contours_vis, _ = cv2.findContours(
            logo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay_with_logo, logo_contours_vis, -1, (255, 150, 0), 2)

    cv2.imwrite(f"{base}_highlight_mask.jpg",    cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(f"{base}_highlight_overlay.jpg", overlay_with_logo)
    cv2.imwrite(f"{base}_frr_layer.jpg",         frr_global)
    cv2.imwrite(f"{base}_corrected.jpg",         final)
    cv2.imwrite(f"{base}_comparison.jpg",
                create_comparison(image, refined, frr_global, final))

    print(f"\n  저장 완료:")
    print(f"    · {base}_highlight_overlay.jpg")
    print(f"    · {base}_corrected.jpg")
    print(f"    · {base}_comparison.jpg")

    return final, refined


def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    test_images = [
        os.path.join(script_dir, "testimg_1.jpg"),
        os.path.join(script_dir, "testimg_2.jpg"),
    ]

    print("=" * 62)
    print("  광택/빛반사 제거 v4.4 (미세 반사 + 텍스처 복원)")
    print("=" * 62)

    done = 0
    for path in test_images:
        if os.path.exists(path):
            process_image(path)
            done += 1
        else:
            print(f"[경고] 파일 없음: {path}")

    print(f"\n{'=' * 62}")
    print(f"  처리 완료: {done}/{len(test_images)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
