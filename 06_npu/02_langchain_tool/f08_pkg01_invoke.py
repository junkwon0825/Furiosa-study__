from furiosa_study.furiosa1.a_kingkeras import get_llm

llm = get_llm(
    max_tokens=1000,
    temperature=0.9,
    top_p=0.9,
)    

response = llm.invoke("Npu와 Gpu의 차이점에 대해 설명해줘.")


print(response.content)
