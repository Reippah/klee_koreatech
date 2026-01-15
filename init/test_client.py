import requests  # HTTP 요청을 보내기 위한 라이브러리 (pip install requests 필요)

# 1. 서버의 API 엔드포인트 URL 설정
# 127.0.0.1(localhost)의 8000번 포트에서 실행 중인 '/upload-video' 경로로 지정
url = "http://127.0.0.1:8000/upload-video"

# 2. 서버로 전송할 테스트용 영상 파일 경로
file_path = "pyramid.mp4"

# 3. 비디오 파일을 바이너리 읽기(rb) 모드로 열기
# 'with' 문을 사용하여 작업이 끝나면 자동으로 파일을 닫도록 처리
with open(file_path, "rb") as f:
    # 4. 서버에서 요구하는 키 값('file')과 파일 객체를 딕셔너리 형태로 구성
    # 이 형식은 'multipart/form-data' 전송 방식에 사용됩니다.
    files = {"file": f}
    
    # 5. POST 요청을 통해 서버로 파일 업로드 수행
    # requests.post() 함수가 내부적으로 헤더 설정 및 멀티파트 인코딩을 처리함
    response = requests.post(url, files=files)
    
    # 6. 서버로부터 받은 응답 결과(JSON 형태)를 파싱하여 화면에 출력
    # 예: {"status": "success", "message": "파일 업로드 완료"} 등의 메시지 확인 가능
    print(response.json())
