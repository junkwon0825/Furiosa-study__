from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="Empty",

)


# 서버에 로드된 모델 확인
model = client.models.list().data[0].id

# 추론
streaming = client.chat.completions.create(
    model = model,
    messages = [
        {"role": "user", "content": "Npu와 Gpu의 차이점에 대해 설명해줘."},
    ],
    max_tokens=300,
    stream=True
)

for chunk in streaming:
    if chunk.choices[0].finish_reason is not None:
        break
    print(chunk.choices[0].delta.content, end="", flush=True)





