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