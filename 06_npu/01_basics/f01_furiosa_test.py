"""
furiosa-llm serve furiosa-ai/Qwen3-8b-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b (x)
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic


furiosa-llm serve furiosa-ai/Qwen3-8b-FP8 \
       --devicces npu:0 \
       --host 0.0.0.0 \
       --port 8000 \
         


"""



from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="Empty",

)

# 서버에 로드된 모델 확인
model = client.models.list()
model_id = model.data[0].id

print("모델 :", model_id)


# 추론
response = client.chat.completions.create(
    model = model_id,
    messages = [
        {"role": "user", "content": "Npu와 Gpu의 차이점에 대해 설명해줘."},
    ],
    max_tokens=300
)

print(response.choices[0].message.content)





