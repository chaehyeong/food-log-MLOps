# 오늘 뭐 먹었지?

사진을 업로드해 식사 기록을 남기는 로컬 음식 캘린더 웹앱입니다. 업로드한 이미지는 로컬 CPU에서 사전 학습 이미지 분류 모델로 분석되며, 상위 3개 후보를 보여줍니다. 사용자는 후보 중 하나를 선택하고 코멘트를 저장할 수 있습니다.

로그인, 칼로리 계산, 통계 기능은 포함하지 않습니다.

## 주요 기능

- 캘린더 형태의 음식 기록 조회
- 음식 사진과 먹은 날짜 업로드
- 날짜별 여러 음식 기록 지원
- AI 음식 분류 결과 상위 3개 후보 및 신뢰도 표시
- 사용자가 선택한 메뉴와 코멘트 저장
- 기록 상세 조회 및 삭제
- SQLite 기반 로컬 저장
- 업로드 오류, 잘못된 파일 형식, 용량 초과, 모델 오류 처리

## 사용 환경

- Windows PowerShell
- Python 3.15 기준으로 확인
- 일반 노트북 CPU에서 실행 가능
- FastAPI, Uvicorn, SQLite
- PyTorch CPU와 Hugging Face Transformers

현재 기본 모델은 `nateraw/food`입니다. Food-101 계열 음식 이미지 분류 모델이며, 한국 음식 전용 모델은 아닙니다. 결과는 추정치이므로 화면에서 후보를 확인하고 사용자가 직접 선택해야 합니다.

## 설치 방법

가상환경 없이 설치된 Python을 사용합니다.

```powershell
cd "프로젝트\폴더\Homework 2"
python -m pip install -r requirements.txt
```

처음 설치할 때 PyTorch CPU 패키지와 Transformers가 설치됩니다. 모델 파일은 첫 번째 분석 요청 때 Hugging Face에서 자동으로 다운로드되어 사용자 캐시에 저장됩니다.

## 모델 준비 방법

별도의 수동 모델 파일 복사는 필요하지 않습니다. 다음처럼 한 번 실행하면 `model.py`가 `nateraw/food` 모델을 자동으로 다운로드합니다.

```powershell
python -c "from model import classify; from pathlib import Path; print(classify(Path('sample.jpg')))"
```

실제 음식 사진을 업로드해도 첫 업로드 시 자동으로 모델이 준비됩니다. 모델 다운로드에는 인터넷 연결이 필요하고, 이후에는 로컬 캐시를 사용합니다.

모델을 교체하려면 [model.py](model.py)의 `classify()` 구현과 `MODEL_VERSION`을 변경하면 됩니다. 웹 API와 저장 구조는 그대로 사용할 수 있습니다.

## 실행 방법

PowerShell에서 다음 중 하나를 실행합니다.

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

또는:

```powershell
.\run.ps1
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8000
```

서버 종료는 PowerShell에서 `Ctrl+C`를 누르면 됩니다.

## 저장 데이터

- SQLite 데이터베이스: `data/foodlog.sqlite3`
- 업로드 이미지: `data/uploads/`

두 위치는 개인 음식 기록과 이미지를 포함할 수 있으므로 Git에 올리지 않도록 `.gitignore`에 포함되어 있습니다.

## 프로젝트 구조

```text
app.py              FastAPI API, 파일 업로드, SQLite 저장
model.py            교체 가능한 AI 모델 로더와 추론 코드
requirements.txt    Python 패키지 목록
run.ps1             PowerShell 실행 스크립트
static/index.html   화면 구조
static/style.css    화면 스타일
static/app.js       캘린더, 업로드, 후보 선택, 삭제 동작
data/               로컬 DB와 업로드 이미지 저장 위치
```

## 주의 사항

- 모델 예측은 추정치이며 정답을 보장하지 않습니다.
- `nateraw/food`는 한국 음식 전용 분류 모델이 아닙니다.
- 모델 최초 다운로드에는 시간이 걸릴 수 있습니다.
- 이 앱은 로컬 실행을 전제로 하며 로그인이나 사용자별 권한 관리는 제공하지 않습니다.
