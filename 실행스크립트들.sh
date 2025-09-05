# .env파일에 OPENAI_API_KEY=your_openai_api_key_here 를 넣어서 생성
# 저 파일은 직접 수정해야 함 (.gitignore에 포함되어 있어서 GitHub에 안올라감)
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env


# uv가 설치되어 있지 않은 경우 => uv 설치 (택1)
pip install uv # 1) pip로 설치
curl -LsSf https://astral.sh/uv/install.sh | sh # 2) curl로 설치
wget -qO- https://astral.sh/uv/install.sh | sh # 3) wget으로 설치

# uv 환경 설정
uv init --python 3.12.11
uv python pin 3.12.11
uv sync # uv 환경(pyproject.toml)에 맞게 패키지 설치

model=Qwen/Qwen3-8B # 불러올 모델 설정
model_name=${model#*/} # Qwen3-8B

### vLLM 서버 띄우기 (택1) ###
# 1) 추천: 도커로 올리기 (도커 쓰는 경우 사용하고 이미지/컨테이너 제대로 안지우면 용량이 폭발하게 되므로 주의)
docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.10.1 \
    --model $model
# 2) 도커 못쓰는 상황 (tmux에서 실행)
uv vllm serve $model
#############################

### inference하는 코드
uv run inference.py \
    --model $model \
    --input_file data/logiqa_test_direct.csv \
    --output_file result/${model_name}-logiqa_test_direct.csv \
    --batch_size 50
############################