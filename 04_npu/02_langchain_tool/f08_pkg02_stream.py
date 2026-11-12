from furiosa_study.furiosa1.a_kingkeras import get_llm

llm = get_llm()    


for chunk in llm.stream("Npu와 Gpu의 차이점에 대해 설명해줘."):
    print(chunk.content, end="", flush=True)
