# 필요한 argument 설정
model="Qwen/Qwen3-8B" # 사용할 모델
model_name=${model#*/} # Qwen3-8B

PORT=8000
TP=1 # 사용할 gpu 수에 따라 가속 가능
gpus=0 # 사용할 GPU 번호 설정

echo "vLLM 서버 실행 중... (PID: $SERVER_PID)"
# 백그라운드에서 vLLM 서버 실행
CUDA_VISIBLE_DEVICES=$gpus uv run vllm serve $model \
    --port $PORT \
    --tensor-parallel-size $TP \
    --async-scheduling &

# gpu 용량 문제로 모델이 안올라갈 때 max_model_len을 설정해보는 것 추천
# --max_model_len $MAX_LEN


SERVER_PID=$!
echo "서버 준비 대기 중..." # 2. 서버 준비될 때까지 대기
until curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null; do # vLLM 서버에 request를 2초마다 달려서 답이 오면 준비된 것
    sleep 2
done
echo "서버 준비 완료!"

### inference하는 코드
uv run inference.py \
    --model $model \
    --input_file data/logiqa_test_direct.csv \
    --output_file result/${model_name}-logiqa_test_direct.csv \
    --batch_size 50
    
# 끝나면 서버 종료
kill $SERVER_PID
echo "vLLM 서버 종료"
