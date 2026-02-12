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
    사물인 경우, 무늬가 있는 식탁보나 신문지를 깔아두기
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

blender 모험기
0.  Pointcloud 보이게 하기
    회색 모델링이 끝이고
    shading 사용해서 색을 입혀야 포인트 클라우드 볼 수 있음
1.  Points to Volume 방식
    글자가 안 보일 정도의 무난한 퀄리티
2.  Cube 방식
    기억 안 남



blender 모험기 -2
0.  Pointcloud 보이게 하기
    5.버전의 blender는 .ply 파일을 불러온 다음, 화면 위 (제일 위 아님)의 Object에서 Convert에서 Point Cloud를 클릭하면 자동으로 변환해준다.
    어제는 Geometry Nodes를 통한 절차를 했어야했지만 그렇게 안 해도 됨.
1.  PointCloud 색 입히기
    오른쪽 아래의 창 (Properties) 보면 왼쪽 메뉴가 많은데, Data탭을 클릭하면 Attribute 내에 PointCloud가 갖고 있는 요소를 확인할 수 있다.
    대부분 Color는 'Col'로 되어있을 듯
    맨 위 메뉴 창에 Shading을 클릭하면 Shading Workspace로 전환이 되는데,
    New를 클릭해서 생성한 다음,
    Shift + A (한/영 조심) 를 누르면 추가할 Node가 뜨는데 Attribute 검색
    Attribute 창이 뜨면 밑의 빈칸에 Col 입력하고 Color부분을 Principled BSDF의 Base Color로 연결해준다.
    Object 메뉴와 같은 높이의 창 오른쪽을 보면 동그란게 4개 보이는데, Material Preview하면 조명 없는 색을 보여주며, Rendered하면 조명 포함해서 렌더링해서 보여준다.
2.  PointCloud Point 크기 조절
    맨 위 메뉴 창에 Geometry Nodes를 클릭하면 Geometry Nodes Workspace로 전환이 된다.
    New를 클릭해서 생성한 다음,
    똑같이 Shift + A 를 눌러 Set Point Radius 노드를 찾아주고
    Group Input - Set Point Radius - Group Output 연결
    Set Point Radius의 Radius 값을 조절하면 Point 크기 조절 가능 0.001이 적당한 듯
    여담으로 포인트가 참 기묘하게 생겼다

번외:blender로 mesh 전환은 디테일 문제로 만들기 어려움.
    하지만 포인트를 상자로 바꾸는 등의 방법을 쓴 후, glb파일 형식으로 export하는 방식을 추천

3.  Instance on Points + Cube (점을 작은 상자로)
    blender_test 폴더 안에 저장.
    Geometry Nodes 내에서 작업할 때는 Group Output에 연결된 선을 끊고 사용할 것. 렉이 어마어마하게 심함

    Group Input - Merge by Distance(점 개수 줄이는 다운샘플링 예시에서는 0.005정도로 함 / 일단 나는 안 함) - (Points는 Merge by Distance와 연결 / Instance는 Cube와 연결)Instance on Points - Realize Instance - Set Material(Shading에서 한 Material로 설정) - Group Output
                                                    Cube(Size에서 xyz 사이즈를 변경) -

    이론상 이 상태로 바로 Mesh로 변환해서 glb 파일로 변환할 수 있지만, 안됨.
    기존 Geometry Nodes 삭제하고 눈 모양 아이콘 까지 눌러서 숨기고 새롭게 만들기 (성능 저하 방지)
    Object info에서 정보를 가져와서 만드는 것이 편이

    모델이 보이는 창에서 Shift + A 눌러서 Mesh -> Cube 생성
    Group Input을 Object info로 바꾸고 안의 Object를 해당 PointCloud 파일로 설정 후 위에 했던 것처럼 Geometry Nodes에서 추가

    장점: 보이긴함
    단점: 자세히보면 너무 각져보임

4.  Instance on Points + Ico Sphere (점을 (모양을 이루고 있는 면이 삼각형인)구로)
    Cube 부분을 Ico Sphere로 대체
    일단 컴퓨터가 한 번 멈춤
    그래서 Merge by Distance를 사용해서 다운샘플링 0.001로 설정
    근데 glb파일 변환 도중 계속 blender가 튕김
    퀄리티는 cube보다 나아보임