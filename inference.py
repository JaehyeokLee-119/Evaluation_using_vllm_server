from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from typing import List, Any, Dict
import random
import asyncio
import dotenv 
import math
import os

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import fire 
import pandas as pd

class Async_inference:
    def __init__(self, client, model_name, system_prompt, temperature=0.0, top_p=0.95, stop=None, reasoning=False, max_new_tokens=None, max_reasoning_tokens=None):
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop  # You can customize stop tokens here
        self.reasoning = reasoning
        self.max_new_tokens = max_new_tokens
        # self.max_reasoning_tokens = max_reasoning_tokens
        
        # --- Config --- #
        self.MAX_CONCURRENCY = 64         # Tune for your GPUs/network
        self.MAX_RETRIES = 5
        self.INITIAL_BACKOFF = 0.7        # seconds
        self.BACKOFF_JITTER = 0.25        # seconds
        self.reasoning_effort = 'low'  # 'low' | 'medium' | 'high' for gpt-oss models

    async def _call_one(self, sample_text: str, sem: asyncio.Semaphore, idx: int) -> Dict[str, Any]:
        """
        한번의 API Call을 수행 (입력을 전송해서 output과 reasoning을 받음)
        return형: {"index": idx, "response": out_text, 'reasoning': reasoning_text if self.reasoning else None}
        """
        content = sample_text
        attempt = 0
        backoff = self.INITIAL_BACKOFF

        while True:
            try:
                async with sem:
                    messages = [
                        {"role": "system", "content": self.system_prompt}
                    ] if self.system_prompt else []
                    messages.append(
                        {"role": "user", "content": content})

                    resp = await self.client.chat.completions.create(
                        model=self.model_name,
                        reasoning_effort=self.reasoning_effort,
                        messages=messages,
                        extra_body={
                            "chat_template_kwargs": {
                                'enable_thinking': self.reasoning,
                            }
                        },
                        stop=self.stop,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_new_tokens,
                        # max_reasoning_tokens=self.max_reasoning_tokens,
                    )
                out_text = resp.choices[0].message.content
                if self.reasoning:
                    reasoning_text = resp.choices[0].message.reasoning_content
                return {"index": idx, "content": out_text, 'reasoning': reasoning_text if self.reasoning else None}
            except Exception as e:
                attempt += 1
                print(f"Error on attempt {attempt} for index {idx}: {e}")
                if attempt > self.MAX_RETRIES:
                    # Give up but return an error record to keep ordering
                    return {"index": idx, "content": f"ERROR: {e}", 'reasoning': None}
                # Exponential backoff with jitter
                jitter = random.uniform(-self.BACKOFF_JITTER, self.BACKOFF_JITTER)
                wait_s = max(0.1, backoff + jitter)
                await asyncio.sleep(wait_s)
                backoff *= 2.0

    async def process_batch_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Parallel-process a list of texts. Preserves original order in the returned list.
        """
        sem = asyncio.Semaphore(self.MAX_CONCURRENCY)
        tasks = [
            asyncio.create_task(self._call_one(t, sem, i))
            for i, t in enumerate(texts)
        ]
        results = await asyncio.gather(*tasks)
        # Ensure original order
        results.sort(key=lambda r: r["index"])
        return results


def initialize_client(client_type):
    # client 설정 함수
    if client_type == "local":
        client = AsyncOpenAI(base_url="http://localhost:8000/v1",api_key="EMPTY") 
    elif client_type == "openai":
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    else:
        raise ValueError("client_type must be either 'local' or 'openai'")
    return client

def main(
    input_file="data/logiqa_test_direct.csv",
    output_file="result/Qwen3-8B-logiqa_test_direct.csv",
    model="Qwen/Qwen3-8B",
    client_type="local",  # "local" or "openai"
    batch_size = 100,
    reasoning = False, # reasoning mode on/off
    max_new_tokens = 2048, # max new tokens to generate
):
    ###############################################################################
    # client 설정
    client = initialize_client(client_type) # API Call을 담당할 client
    # system_prompt = "You are a helpful assistant."
    system_prompt = None
    stop_words = None # 생성을 멈출 단어 목록 (혹시나 예상치않게 생성을 길게 만드는 패턴이 발견된다면 적용해보면 효율적인 실험 가능)
    async_inferencer = Async_inference(client, 
                                       model, system_prompt, 
                                       temperature=0.6, top_p=0.95, 
                                       stop=stop_words, reasoning=reasoning,
                                       max_new_tokens=max_new_tokens)
    ###############################################################################

    ###############################################################################
    # Dataset 불러오기
    df = pd.read_csv(input_file)
    # 결과 컬럼 생성
    df['output'] = None
    df['reasoning'] = None
    output_place = [[]]*len(df) # output이 저장될 장소
    reasoning_place = [[]]*len(df) # reasoning output이 저장될 장소
    ###############################################################################

    ###############################################################################
    # Inference 진행
    n_batches = math.ceil(len(df) / batch_size)
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))

        # 입력 준비
        batch = df.loc[start:end-1, 'input'].astype(str).tolist()

        # 수행
        responses = asyncio.run(async_inferencer.process_batch_async(batch))

        output_place[start:end] = [r['content'] for r in responses]
        reasoning_place[start:end] = [r['reasoning'] for r in responses]
        df['output'] = output_place
        df['reasoning'] = reasoning_place

        # batch 하나마다 저장
        df.to_csv(output_file, index=False)
    ###############################################################################
    print("✅ done")

if __name__ == "__main__":
    fire.Fire(main)