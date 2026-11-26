import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

start = time.perf_counter()
response = llm.invoke("NPU의 장점을 세가지 설명해줘")
elapsed = time.perf_counter() - start

usage = response.usage_metadata or {}

print("응답 :", response.content )
print("지연시간 :",  round(elapsed, 3), "초")
print("입력토큰 :", usage.get("input_tokens", 0))
print("출력토큰 :", usage.get("output_tokens", 0))
