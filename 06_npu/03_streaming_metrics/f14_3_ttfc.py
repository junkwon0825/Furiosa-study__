# time to first chunk ( ttfc )
from kingkeras import get_llm
import time

llm = get_llm(max_tokens=500)

prompt = "NPU의 장점을 자세히 설명해줘"

start = time.perf_counter()

chunk = next(iter(llm.stream(prompt)))

first_time = time.perf_counter() - start

print("===========================================")
print("첫 컨탠츠 응답시간 :", round(first_time,2), "초")













