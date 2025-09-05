# LogiQA Evaluation Workspace

vLLM을 활용하여 모델을 LogiQA 테스트 데이터로 평가하는 간단한 가이드

## 개요

이 폴더에는 원본 LogiQA 테스트 데이터, 전처리 노트북 (.ipynb), 추론 스크립트와 결과 CSV가 포함되어 있습니다. 주 목적은 다음과 같습니다:

### dataset_load_and_process.ipynb
- 원본 데이터(logiqa_test.parquet)를 읽어 Direct/CoT 입력 텍스트를 생성 (.csv)
- csv파일의 input column이 그대로 user prompt로 들어가게 됨

### inference.py
- csv 파일의 input column을 읽어서 모델 추론을 수행, 결과를 output column에 넣어서 저장

### evaluation.ipynb
- 결과 파일을 평가하여 정확도를 계산

## 시작
[uv](https://docs.astral.sh/uv/)를 통해 환경을 구성합니다.
필요한 패키지들은 pyproject.toml에 정의되어 있으며, 구성 방법은 `실행스크립트들.sh`를 확인

## 구현이 안된 부분
- second prediction (첫 inference에서 정답이 안 나왔을 때 다시 요청하는 부분)
- openAI API 테스트 안해봄
- inference.py 파일은 .env 파일에서 API KEY를 불러오므로, API 사용을 위해서는 저기에 KEY를 넣기