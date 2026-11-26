"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:1 \
    --host 127.0.0.1 \
    --port 8000

"""


from kingkeras import get_llm

llm = get_llm()

prompt = "Npu와 GPU의 차이를 설명해줘."

response = llm.invoke(prompt)

print(response.content)