"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 127.0.0.1 \
    --port 8000
    --enable-overlap-scheduling
"""
# —enable-overlap-scheduling은 Furiosa-LLM

# 서빙엔진에서 cpu의 스케줄링 작업과 NPU의 추론 연산을 겹쳐 실행하도록 하는 최적화 옵션

# Continuous Batching 여러 사용자의 요청을 동적으로 묶어 NPU에서 처리하는 기술

# overlap scheduling npu가 현재 배치를 처리하는 동안 cpu가 다음배치를 준비하는 기술

# ↔ EAGER scheduling 순차처리에 유리 디폴트
import statistics
import time
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=1000)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘"
    for i in range(20)
]

def ask(prompt):
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)

    return elapsed, tokens

for workers in [1,2,4,8]:

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(ask, prompts))
    total_time = time.perf_counter() - start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f"\n======= Workers: {workers} ===========")

    
    print("전체 시간 :", round(total_time, 2), "초")
    print("평균 지연 :", round(statistics.mean(latencies),2), "초")
    print("요청 처리량 :", round(len(results) / total_time,2), "req/s")
    print("Output TPS: ",round(total_tokens / total_time, 2))

print()

# ======= Workers: 8 =========== eager
# 전체 시간 : 2.44 초
# 평균 지연 : 1.22 초
# 요청 처리량 : 3.28 req/s
# Output TPS:  328.03


# ======= Workers: 8 =========== overlap
# 전체 시간 : 2.49 초
# 평균 지연 : 1.24 초
# 요청 처리량 : 3.22 req/s
# Output TPS:  321.76