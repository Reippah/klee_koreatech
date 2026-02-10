# 한국기술교육대학교 26-1 졸업설계 캡스톤디자인

pip install <사용하고자 하는 모듈> --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

1. 모바일과 연동
    진행상황, 다운로드 기능
2. 포인트 클라우드 고품질화
    고화질의 이미지를 1024x1024로 변경해서 포인트의 색상을 저장 후 518x518로 변환해서 추론 시작.
    포인트의 색상은 마지막에 포인트 클라우드를 고품질화하는 데에 쓰임.
3. gpu lock을 이용한 threading 구현
    들어온 순서대로 전처리 단계 끝난 후 gpu lock을 검
4. 정말 엄청나게 수많은 meshlab을 이용한 mesh 변환 테스트
    과정:
        0. Open3D 라이브러리를 이용한 pointcloud to mesh : 사면체가 대놓고 보임, 3D 모델 결과로서의 의미 X 
        1. jetson 내 meshlab 사용 : 간헐적 튕김으로 인한 불가능 판단
        2-1. hostpc 내 meshlab 사용 : poisson을 이용한 mesh 변환 성공, 만족스럽지 않은 결과
        2-2. pymeshlab 라이브러리를 이용한 mesh 변환 : jetson의 64GB 메모리에서 조차 OOM 현상 발생, 불가능. 원인 분석 필요
5. 전처리 단계 및 포인트 클라우드 추출 면은 만족, 하지만 입력 단계에서의 촬영 방법 분석 필요



촬영방법
0.  사물인 경우, 사진의 정중앙에 오도록 유지
    사물인 경우, 주변에 약간의 여백을 두기
    사물인 경우, 사물에 초점을 맞춰 노출과 초점을 고정하기
    사물인 경우, 무늬가 있는 식탁보나 신문기를 깔아두기
    너무 강한 직사광선 보다는 실내의 부드러운 전등빛으로 그림자 최소화
    한 장의 사진과 다음 사진 사이에는 70~80% 이상의 겹치는 영역이 있어야함
    광각렌즈 사용 X

1.  사람보다 작은 사물
    반구 촬영 기법
    낮은 위치에서 한 바퀴, 중간 높이에서 한 바퀴, 위에서 아래로 내려다보며 한 바퀴
    15 ~ 20도 간격으로 한 바퀴당 약 18 ~ 24장

2.  사람보다 큰 사물
    사물을 중심으로 서로 다른 높이에서 최소 3번의 큰 원을 그리며 이동
    카메라를 약간 위를 향하게 / 사물과 수평이 되도록 / 아래를 내려다보게
    루프 사이를 이동할 때 위아래로 카메라를 흔들며 이동
    한 바퀴를 다 돌았을 때, 처음 찍었던 위치와 약간 겹치게 3~4장을 더 찍기
    일정한 거리를 유지하기
    대형 사물 뒤로 지나가는 물체는 치명적인 노이즈 발생
    매끈한 대형 구조물은 특징점이 없어서 복원이 힘듬

3.  방
    각 구석에서 시작해 반대편을 바라보며 촬영
    방의 네 모서리에 서서 반대편 벽과 천장, 바닥이 모두 보이도록 부채꼴 모양으로 천천히 회전하며 촬영
    한가운데 서서 회전하면서 찍는 것 보다, 약간의 위치를 옮기며 찍어야 깊이감 계산 편이 (Baseline 확보)
    일반적은 눈높이에서 방 전체를 한 바퀴, 천장과 벽이 만나는 모서리를 위주로 한 바퀴, 가구 밑부분과 바닥이 만나는 지점 위주로 한 바퀴
    (벽면에 포스트잇 등을 붙여 시각적 단서를 인위적으로 만들기)

02.09 10:00~14:00
MeshLab을 이용한 Poisson Reconstruction
1.  법선 계산
    Filters > Normals, Curvatures and Orientation > Compute normals for point sets
    Neighbor num: 20
    Apply 클릭
2.  거미줄 치기 (Poisson Reconstruction)
    Filters > Remeshing, Simplification and Reconstruction > Screened Poisson Surface Reconstruction
    Reconstruction Depth : 11
    Apply 클릭
3.  거품 밀도 보기
    Render > Show Quality Histogram
    그래프를 보고 왼쪽 숫자를 기준으로 적절한 최소값/최대값 찾기
4.  거품 제거
    Filters > Selection > Select Vertices by Quality
    아까 찾은 최소 최대값 넣기
    Preview 클릭시 지워질 부분 확인 가능
    이 상태로 바로 지우는 것이 아니라, Filters > Selection > Select Faces from Vertices 클릭 후 Apply 하고 Delete 눌러서 삭제
5. obj 파일로 만들기
    이 상태에서 렉 유발
    File > Export Mesh As...
    object 어쩌구로 파일 저장
    저장된 파일로 열면 렉 X


HostPC안의 meshlab_test 폴더 안에 pymeshlab을 사용한 obj 파일 생성 py 파일
fire_extinguisher 기준 소요 시간 1분 53초