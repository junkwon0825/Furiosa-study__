
from langchain_openai import ChatOpenAI
import time

max_tokens = 1000

llm = ChatOpenAI(
    model_name="furiosa-ai/Qwen3-8b-FP8",
    base_url="http://127.0.0.1:8000/v1",
    api_key="Empty",
    max_tokens=max_tokens,
    temperature=0,
    top_p=0.9,
)

start = time.perf_counter()
response = llm.invoke("Npu와 Gpu의 차이점에 대해 설명해줘.")
end = time.perf_counter()   
elapsed_time = round(end - start, 2)

print(response.content)
print("출력토큰:" ,max_tokens)
print("평균처리량: ", round(max_tokens / elapsed_time, 2), "토큰/초")


