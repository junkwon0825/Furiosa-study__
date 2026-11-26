

"""
furiosa-llm serve furiosa-ai/Qwen3-8b-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b (x)
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic

"""


from openai import OpenAI
from langchain_openai import ChatOpenAI


# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="Empty",

# )
llm = ChatOpenAI(
    model_name="furiosa-ai/Qwen3-8b-FP8",
    base_url="http://127.0.0.1:8000/v1",
    api_key="Empty",
)

# # 서버에 로드된 모델 확인
# model = client.models.list()
# model_id = model.data[0].id

# print("모델 :", model_id)


# # 추론
# response = client.chat.completions.create(
#     model = model_id,
#     messages = [
#         {"role": "user", "content": "Npu와 Gpu의 차이점에 대해 설명해줘."},
#     ],
#     max_tokens=300
# )
response = llm.invoke("Npu와 Gpu의 차이점에 대해 설명해줘.")


print(response.content)

# print(response.choices[0].message.content)





