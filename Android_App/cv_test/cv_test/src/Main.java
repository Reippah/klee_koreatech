/**
 * ==============================================================
 *   광택 / 표면 빛반사 자동 제거 알고리즘  (Java + OpenCV)
 *   Specular Highlight & Glare Removal via OpenCV + K-Means ML
 * ==============================================================
 *
 * ■ 처리 파이프라인 (5단계)
 *   1. HSV 임계값   — 고밝기(V) + 저채도(S) 픽셀을 광택 후보로 마스킹
 *   2. K-Means (ML) — 6차원 특징(HSV+BGR)을 클러스터링해 마스크 정제
 *   3. 모폴로지     — Opening/Closing/Dilation 으로 마스크 노이즈 제거
 *   4. 인페인팅     — OpenCV Telea(FMM) 알고리즘으로 광택 영역 복원
 *   5. 블렌딩       — 가우시안 α-맵으로 경계를 자연스럽게 합성
 *
 * ■ 사전 준비 (OpenCV Java 환경 구성)
 *   1. https://opencv.org/releases/ 에서 OpenCV 4.x 다운로드 및 설치
 *   2. IntelliJ → File → Project Structure → Libraries
 *      → opencv-4xx.jar 추가
 *   3. Run/Debug Configuration → VM options 에 추가:
 *      -Djava.library.path=<OpenCV_설치경로>/build/java/x64
 *      예) -Djava.library.path=C:/opencv/build/java/x64
 *
 * ■ 출력 파일 (src/glare_output/ 폴더)
 *   - testimg_N_corrected.jpg   : 보정 완료 이미지
 *   - testimg_N_mask.png        : 검출된 광택 마스크
 *   - testimg_N_pipeline.jpg    : 6단계 파이프라인 시각화 그리드
 */

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import java.io.File;
import java.util.*;

public class Main {

    // ──────────────────────────────────────────────────────
    //  OpenCV 네이티브 라이브러리 로드
    //  VM 옵션: -Djava.library.path=<opencv>/build/java/x64
    // ──────────────────────────────────────────────────────
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // ══════════════════════════════════════════════════════
    //  1단계: HSV 기반 초기 광택 마스크 생성
    // ══════════════════════════════════════════════════════
    /**
     * HSV 색공간에서 고밝기·저채도 픽셀을 광택/반사 후보로 마스킹합니다.
     *
     * 광택 픽셀의 특성:
     *   - Value (밝기)  ↑ 높음 : 빛이 직접 반사되어 매우 밝음
     *   - Saturation(채도) ↓ 낮음 : 물체 본래 색이 희석되어 무채색에 가까움
     *
     * @param imageBGR  입력 BGR 이미지
     * @param satThresh 채도(S) 상한 [0~255] — 이 값 이하이면 광택 후보
     * @param valThresh 밝기(V) 하한 [0~255] — 이 값 이상이면 광택 후보
     * @return 0/255 이진 마스크 (CV_8UC1)
     */
    static Mat detectGlareHSV(Mat imageBGR, int satThresh, int valThresh) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(imageBGR, hsv, Imgproc.COLOR_BGR2HSV);

        List<Mat> channels = new ArrayList<>();
        Core.split(hsv, channels);
        Mat s = channels.get(1);   // 채도 채널
        Mat v = channels.get(2);   // 밝기 채널

        // 채도 < satThresh AND 밝기 > valThresh → 광택 영역
        Mat maskS = new Mat(), maskV = new Mat(), mask = new Mat();
        Core.compare(s, new Scalar(satThresh), maskS, Core.CMP_LT);
        Core.compare(v, new Scalar(valThresh), maskV, Core.CMP_GT);
        Core.bitwise_and(maskS, maskV, mask);

        // 메모리 해제
        hsv.release();
        for (Mat c : channels) c.release();
        maskS.release();
        maskV.release();

        return mask;   // 광택 영역 = 255
    }

    // ══════════════════════════════════════════════════════
    //  2단계: K-Means 클러스터링 기반 마스크 정제 (ML)
    // ══════════════════════════════════════════════════════
    /**
     * 초기 마스크 주변 픽셀을 K-Means로 클러스터링하여 마스크를 정제합니다.
     *
     * 알고리즘:
     *   1) 초기 마스크를 팽창(Dilate)하여 경계 포함 분석 영역 확보
     *   2) 해당 영역의 [H, S, V, B, G, R] 6차원 특징 벡터 수집
     *   3) 특징을 표준화(StandardScaler) 후 K-Means 분류
     *   4) 평균 V(밝기)가 가장 높은 클러스터 = 실제 광택 영역
     *   5) 초기 마스크와 OR 연산으로 최종 마스크 생성
     *
     * @param imageBGR    입력 BGR 이미지
     * @param initialMask 1단계에서 얻은 초기 마스크
     * @param nClusters   K-Means 클러스터 수
     * @param dilationIters 마스크 팽창 반복 횟수
     * @return 정제된 0/255 마스크
     */
    static Mat refineMaskKMeans(Mat imageBGR, Mat initialMask,
                                 int nClusters, int dilationIters) {
        // 초기 마스크 팽창 → 광택 경계 픽셀도 분석에 포함
        Mat kernel  = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(11, 11));
        Mat dilated = new Mat();
        Imgproc.dilate(initialMask, dilated, kernel, new Point(-1, -1), dilationIters);

        Mat hsv = new Mat();
        Imgproc.cvtColor(imageBGR, hsv, Imgproc.COLOR_BGR2HSV);

        // 마스크 영역 픽셀의 특징 벡터 및 좌표 수집
        List<float[]> featureList = new ArrayList<>();
        List<int[]>   coordList   = new ArrayList<>();

        for (int y = 0; y < dilated.rows(); y++) {
            for (int x = 0; x < dilated.cols(); x++) {
                if (dilated.get(y, x)[0] > 0) {
                    double[] hsvPx = hsv.get(y, x);
                    double[] bgrPx = imageBGR.get(y, x);
                    featureList.add(new float[]{
                        (float) hsvPx[0], (float) hsvPx[1], (float) hsvPx[2],  // H, S, V
                        (float) bgrPx[0], (float) bgrPx[1], (float) bgrPx[2]   // B, G, R
                    });
                    coordList.add(new int[]{y, x});
                }
            }
        }

        int N = featureList.size();
        // 픽셀 수가 너무 적으면 정제 없이 반환
        if (N < nClusters * 5) {
            dilated.release(); hsv.release();
            return initialMask;
        }

        // K-Means 데이터 행렬 생성 (N × 6, CV_32F)
        Mat data = new Mat(N, 6, CvType.CV_32F);
        for (int i = 0; i < N; i++) {
            data.put(i, 0, featureList.get(i));
        }

        // 열별 표준화: (x - mean) / std
        // Mat.convertTo(dst, type, alpha, beta) → dst = src * alpha + beta
        for (int col = 0; col < 6; col++) {
            Mat column = data.col(col);
            MatOfDouble meanVec = new MatOfDouble();
            MatOfDouble stdVec  = new MatOfDouble();
            Core.meanStdDev(column, meanVec, stdVec);
            double mean = meanVec.toArray()[0];
            double std  = Math.max(stdVec.toArray()[0], 1e-8);

            // 표준화: x_new = x * (1/std) + (-mean/std)
            Mat normalized = new Mat();
            column.convertTo(normalized, CvType.CV_32F, 1.0 / std, -mean / std);
            normalized.copyTo(column);

            normalized.release(); meanVec.release(); stdVec.release();
        }

        // K-Means 실행 (OpenCV 내장 K-Means++)
        int          actualK  = Math.min(nClusters, N);
        Mat          labels   = new Mat();
        Mat          centers  = new Mat();
        TermCriteria tc       = new TermCriteria(
                TermCriteria.EPS + TermCriteria.MAX_ITER, 300, 1e-4);
        Core.kmeans(data, actualK, labels, tc, 10,
                    Core.KMEANS_PP_CENTERS, centers);

        // 클러스터별 평균 V(밝기) 계산 → 가장 밝은 클러스터 = 광택
        double[] sumV   = new double[actualK];
        int[]    countK = new int[actualK];
        for (int i = 0; i < N; i++) {
            int lbl = (int) labels.get(i, 0)[0];
            sumV[lbl]   += featureList.get(i)[2];   // V 채널 누적
            countK[lbl]++;
        }

        int    glareCluster = 0;
        double maxAvgV      = -1.0;
        for (int k = 0; k < actualK; k++) {
            if (countK[k] > 0) {
                double avg = sumV[k] / countK[k];
                if (avg > maxAvgV) {
                    maxAvgV      = avg;
                    glareCluster = k;
                }
            }
        }

        // 정제된 마스크 생성 (광택 클러스터 픽셀만 255)
        Mat refined = Mat.zeros(imageBGR.rows(), imageBGR.cols(), CvType.CV_8UC1);
        for (int i = 0; i < N; i++) {
            if ((int) labels.get(i, 0)[0] == glareCluster) {
                int[] c = coordList.get(i);
                refined.put(c[0], c[1], 255.0);
            }
        }

        // 초기 마스크와 합집합 (확실한 광택 픽셀 보존)
        Mat result = new Mat();
        Core.bitwise_or(refined, initialMask, result);

        // 메모리 해제
        data.release(); labels.release(); centers.release();
        refined.release(); dilated.release(); hsv.release();

        return result;
    }

    // ══════════════════════════════════════════════════════
    //  3단계: 형태학적 연산으로 마스크 정리
    // ══════════════════════════════════════════════════════
    /**
     * Opening(잡음 제거) → Closing(내부 구멍 메우기) → Dilation(경계 확장)
     * 순서로 마스크를 정리합니다.
     *
     * @param openSize   Opening 커널 크기  (작을수록 미세 잡음만 제거)
     * @param closeSize  Closing 커널 크기  (클수록 넓은 구멍까지 채움)
     * @param dilateSize Dilation 커널 크기 (인페인팅 경계 안전 여유 확보)
     */
    static Mat cleanMaskMorphology(Mat mask,
                                    int openSize, int closeSize, int dilateSize) {
        Mat kOpen  = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(openSize,   openSize));
        Mat kClose = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(closeSize,  closeSize));
        Mat kDil   = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(dilateSize, dilateSize));

        Mat cleaned = new Mat();
        Imgproc.morphologyEx(mask,    cleaned, Imgproc.MORPH_OPEN,  kOpen);
        Imgproc.morphologyEx(cleaned, cleaned, Imgproc.MORPH_CLOSE, kClose);
        Imgproc.dilate(cleaned, cleaned, kDil);

        kOpen.release(); kClose.release(); kDil.release();
        return cleaned;
    }

    // ══════════════════════════════════════════════════════
    //  4단계: Telea 인페인팅으로 광택 영역 복원
    // ══════════════════════════════════════════════════════
    /**
     * OpenCV Telea(FMM, Fast Marching Method) 알고리즘으로
     * 광택 마스크 영역을 주변 텍스처로 자연스럽게 채웁니다.
     *
     * Telea 원리:
     *   - 마스크 경계에서 안쪽으로 순차 전파(Fast Marching)
     *   - 이미 알려진 인접 픽셀들의 가중 평균으로 누락 영역 복원
     *   - 텍스처 연속성이 좋고 큰 영역 처리에 효과적
     *
     * @param radius 탐색 반경(픽셀) — 클수록 넓은 맥락 참조, 느려짐
     */
    static Mat inpaintGlare(Mat imageBGR, Mat mask, int radius) {
        Mat result = new Mat();
        Photo.inpaint(imageBGR, mask, result, (double) radius, Photo.INPAINT_TELEA);
        return result;
    }

    // ══════════════════════════════════════════════════════
    //  5단계: 가우시안 α-블렌딩으로 경계 자연스럽게 합성
    // ══════════════════════════════════════════════════════
    /**
     * 마스크를 가우시안 블러로 부드럽게 만든 α-맵을 이용해
     * 인페인팅 결과와 원본 이미지를 경계 없이 합성합니다.
     *
     * 수식: result = inpainted × α + original × (1−α)
     *   - 광택 중심(α≈1.0) : 인페인팅 결과를 그대로 사용
     *   - 마스크 경계(α≈0~1): 원본과 자연스럽게 전환
     *
     * @param blurKsize 가우시안 블러 커널 크기 (클수록 넓은 전환 구간)
     */
    static Mat blendResult(Mat original, Mat inpainted, Mat mask, int blurKsize) {
        if (blurKsize % 2 == 0) blurKsize++;   // 홀수 보정

        // α-맵 생성 (0.0 ~ 1.0, 가우시안 블러로 부드럽게)
        Mat alpha = new Mat();
        mask.convertTo(alpha, CvType.CV_32F, 1.0 / 255.0);
        Imgproc.GaussianBlur(alpha, alpha, new Size(blurKsize, blurKsize), 0);

        // 3채널 α-맵
        List<Mat> alphaCh = Arrays.asList(alpha, alpha, alpha);
        Mat alpha3 = new Mat();
        Core.merge(alphaCh, alpha3);

        // (1 − α) 계산
        Mat oneMinusAlpha = new Mat(
                alpha3.rows(), alpha3.cols(),
                CvType.CV_32FC3, new Scalar(1.0, 1.0, 1.0));
        Core.subtract(oneMinusAlpha, alpha3, oneMinusAlpha);

        // float 변환 후 α-블렌딩
        Mat origF = new Mat(), inpF = new Mat();
        original.convertTo(origF,  CvType.CV_32F);
        inpainted.convertTo(inpF,  CvType.CV_32F);

        Mat p1 = new Mat(), p2 = new Mat(), blended = new Mat();
        Core.multiply(inpF,  alpha3,        p1);
        Core.multiply(origF, oneMinusAlpha, p2);
        Core.add(p1, p2, blended);

        Mat result = new Mat();
        blended.convertTo(result, CvType.CV_8UC3);

        // 메모리 해제
        alpha.release(); alpha3.release(); oneMinusAlpha.release();
        origF.release(); inpF.release(); p1.release(); p2.release(); blended.release();

        return result;
    }

    // ══════════════════════════════════════════════════════
    //  시각화: 처리 단계를 2×3 그리드 이미지로 합성
    // ══════════════════════════════════════════════════════
    /**
     * 6개 처리 단계 이미지를 2행 × 3열 그리드로 합쳐 파이프라인을 시각화합니다.
     * 각 셀 상단에 레이블 텍스트를 삽입합니다.
     */
    static Mat createVisualizationGrid(Mat orig,     Mat maskInit, Mat maskFinal,
                                        Mat overlay, Mat inpainted, Mat result) {
        // 마스크(회색)에 HOT 컬러맵 적용하여 열지도 표현
        Mat maskInitColor  = new Mat();
        Mat maskFinalColor = new Mat();
        Imgproc.applyColorMap(maskInit,  maskInitColor,  Imgproc.COLORMAP_HOT);
        Imgproc.applyColorMap(maskFinal, maskFinalColor, Imgproc.COLORMAP_HOT);

        String[] labels = {
            "1. Original Image",
            "2. Initial Mask (HSV)",
            "3. Refined Mask (K-Means)",
            "4. Mask Overlay",
            "5. Inpainted (Telea)",
            "6. Final Result"
        };
        Mat[] panels = { orig, maskInitColor, maskFinalColor,
                         overlay, inpainted, result };

        // 각 패널을 540×540 으로 통일 + 상단 레이블 추가
        Size cellSize = new Size(540, 540);
        Mat[] cells   = new Mat[6];
        for (int i = 0; i < 6; i++) {
            cells[i] = new Mat();
            Imgproc.resize(panels[i], cells[i], cellSize);

            // 레이블 배경 바 (어두운 반투명 사각형)
            Imgproc.rectangle(cells[i],
                    new Point(0, 0), new Point(540, 44),
                    new Scalar(20, 20, 20), -1);

            // 레이블 텍스트
            Imgproc.putText(cells[i], labels[i],
                    new Point(8, 32),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.75, new Scalar(0, 230, 230), 2);
        }

        // 2행 × 3열 그리드 조합
        Mat row1 = new Mat(), row2 = new Mat(), grid = new Mat();
        Core.hconcat(Arrays.asList(cells[0], cells[1], cells[2]), row1);
        Core.hconcat(Arrays.asList(cells[3], cells[4], cells[5]), row2);
        Core.vconcat(Arrays.asList(row1, row2), grid);

        // 메모리 해제
        maskInitColor.release(); maskFinalColor.release();
        for (Mat c : cells) c.release();
        row1.release(); row2.release();

        return grid;
    }

    // ══════════════════════════════════════════════════════
    //  메인 처리 파이프라인
    // ══════════════════════════════════════════════════════
    /**
     * 단일 이미지에 대해 광택 제거 전체 파이프라인을 실행하고
     * 결과물을 outputDir 에 저장합니다.
     *
     * @param imagePath     입력 이미지 경로
     * @param outputDir     결과물 저장 디렉터리
     * @param satThresh     HSV 채도 임계값 (낮을수록 엄격)
     * @param valThresh     HSV 밝기 임계값 (높을수록 엄격)
     * @param inpaintRadius Telea 인페인팅 탐색 반경(픽셀)
     * @param nClusters     K-Means 클러스터 수
     */
    static void processImage(String imagePath, String outputDir,
                              int satThresh,    int valThresh,
                              int inpaintRadius, int nClusters) {
        String filename = new File(imagePath).getName();
        String stem     = filename.substring(0, filename.lastIndexOf('.'));

        System.out.println("\n" + "=".repeat(55));
        System.out.println("  처리 파일 : " + filename);
        System.out.println("=".repeat(55));

        // ── 이미지 로드 ──────────────────────────────────────
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.out.println("  [오류] 이미지를 불러올 수 없습니다: " + imagePath);
            return;
        }
        System.out.printf("  이미지 크기 : %d × %d px%n",
                image.cols(), image.rows());

        // ── 1단계: HSV 초기 마스크 ───────────────────────────
        System.out.println("  [1단계] HSV 기반 초기 광택 마스크 생성...");
        Mat maskInitial = detectGlareHSV(image, satThresh, valThresh);
        double pctInit = Core.countNonZero(maskInitial) * 100.0
                       / ((long) image.rows() * image.cols());
        System.out.printf("          초기 광택 비율 : %.2f%%%n", pctInit);

        // ── 2단계: K-Means 정제 (ML) ─────────────────────────
        System.out.println("  [2단계] K-Means 클러스터링으로 마스크 정제 (ML)...");
        Mat maskRefined = refineMaskKMeans(image, maskInitial, nClusters, 2);

        // ── 3단계: 모폴로지 정리 ─────────────────────────────
        System.out.println("  [3단계] 형태학적 연산으로 마스크 정리...");
        Mat maskFinal = cleanMaskMorphology(maskRefined, 3, 9, 3);
        double pctFinal = Core.countNonZero(maskFinal) * 100.0
                        / ((long) image.rows() * image.cols());
        System.out.printf("          최종 광택 비율 : %.2f%%%n", pctFinal);

        // ── 4단계: Telea 인페인팅 ────────────────────────────
        System.out.printf("  [4단계] Telea 인페인팅 (반경=%dpx)...%n", inpaintRadius);
        Mat inpainted = inpaintGlare(image, maskFinal, inpaintRadius);

        // ── 5단계: 가우시안 블렌딩 ───────────────────────────
        System.out.println("  [5단계] 가우시안 블렌딩으로 경계 자연스럽게 합성...");
        Mat result = blendResult(image, inpainted, maskFinal, 31);

        // ── 오버레이 생성 (마스크 영역을 빨간색으로 하이라이트) ──
        Mat overlay = image.clone();
        Mat redFill = new Mat(image.size(), CvType.CV_8UC3, new Scalar(0, 0, 220));
        redFill.copyTo(overlay, maskFinal);
        Core.addWeighted(image, 0.5, overlay, 0.5, 0.0, overlay);

        // ── 결과물 저장 ──────────────────────────────────────
        String sep        = File.separator;
        String corrected  = outputDir + sep + stem + "_corrected.jpg";
        String maskFile   = outputDir + sep + stem + "_mask.png";
        String vizFile    = outputDir + sep + stem + "_pipeline.jpg";

        Imgcodecs.imwrite(corrected, result,
                new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 95));
        Imgcodecs.imwrite(maskFile, maskFinal);
        System.out.println("  [결과 저장]   " + stem + "_corrected.jpg");
        System.out.println("  [마스크 저장] " + stem + "_mask.png");

        // ── 파이프라인 시각화 그리드 저장 ────────────────────
        System.out.println("  [시각화] 파이프라인 그리드 이미지 생성...");
        Mat vizGrid = createVisualizationGrid(
                image, maskInitial, maskFinal, overlay, inpainted, result);
        Imgcodecs.imwrite(vizFile, vizGrid);
        System.out.println("  [시각화 저장] " + stem + "_pipeline.jpg");

        // ── 메모리 해제 ──────────────────────────────────────
        image.release();    maskInitial.release(); maskRefined.release();
        maskFinal.release(); inpainted.release();   result.release();
        overlay.release();  redFill.release();      vizGrid.release();
    }

    // ══════════════════════════════════════════════════════
    //  엔트리 포인트
    // ══════════════════════════════════════════════════════
    public static void main(String[] args) {

        // IntelliJ 실행 시 user.dir = 프로젝트 루트 디렉터리
        // 이미지 파일은 src/ 폴더에 있음
        String projectDir = System.getProperty("user.dir");
        String srcDir     = projectDir + File.separator + "src";
        String outputDir  = srcDir     + File.separator + "glare_output";

        // 출력 폴더 생성
        new File(outputDir).mkdirs();

        // ─── 처리 이미지 목록 및 개별 파라미터 설정 ───────────
        //
        //  파라미터 튜닝 가이드:
        //   satThresh ↓  → 더 순수한 무채색만 광택으로 인식 (엄격)
        //   valThresh ↑  → 더 밝은 픽셀만 광택으로 인식     (엄격)
        //   inpaintRadius↑→ 넓은 영역 참조, 대형 광택에 유리
        //   nClusters ↑  → 더 세밀한 클러스터, 복잡한 이미지에 적합
        //
        //  { 이미지경로, satThresh, valThresh, inpaintRadius, nClusters }
        String[][] configs = {
            // testimg_1: 검정 플라스틱 마우스 — 어두운 표면, 반사광은 밝고 무채색
            { srcDir + File.separator + "testimg_1.jpg", "45", "200", "8",  "4" },
            // testimg_2: 코카콜라 캔 — 금속 상단 정반사, 강한 스펙큘러 하이라이트
            { srcDir + File.separator + "testimg_2.jpg", "35", "215", "9",  "5" },
        };

        System.out.println("\n" + "★".repeat(55));
        System.out.println("  광택 / 표면 빛반사 자동 제거 알고리즘 시작");
        System.out.println("  출력 디렉터리: " + outputDir);
        System.out.println("★".repeat(55));

        int success = 0;
        for (String[] cfg : configs) {
            String path = cfg[0];
            if (!new File(path).exists()) {
                System.out.println("[경고] 파일 없음: " + path);
                continue;
            }
            processImage(
                path,
                outputDir,
                Integer.parseInt(cfg[1]),   // satThresh
                Integer.parseInt(cfg[2]),   // valThresh
                Integer.parseInt(cfg[3]),   // inpaintRadius
                Integer.parseInt(cfg[4])    // nClusters
            );
            success++;
        }

        System.out.println("\n" + "=".repeat(55));
        System.out.printf("  처리 완료 : %d / %d 개%n", success, configs.length);
        System.out.println("  결과물 위치 : " + outputDir);
        System.out.println("=".repeat(55) + "\n");
    }
}
