import random

# random.random(): 0.0 이상 1.0 미만의 실수(float)를 반환합니다.
# random.randint(a, b): a 이상 b 이하의 정수(int)를 반환합니다.
# random.randrange(start, stop): start 이상 stop 미만의 정수를 반환합니다.
# random.choice(seq): 리스트나 문자열 같은 시퀀스 자료형에서 요소를 무작위로 하나 뽑습니다.
# random.shuffle(x): 리스트의 순서를 무작위로 섞습니다

#[]만큼 ()행렬 숫자 생성
#[3,2,4] -> [[[1,2,3,4],[5,6,7,8]],[[9,10...],[]],[[],[]]]
matrix_shape = [3,2,4]
mslen = len(matrix_shape)-1
last_val = 0
results = ""

# print("ms:",mslen)

#입력값을 바꾸고 싶으면 last에 들어가는 값을 조정하기
def create_last(num):
    global last_val
    last=[]
    #랜덤 값 생성
    # for i in range(matrix_shape[num]):
    #     last.append(random.randrange(1, 10))
    #순차값 생성
    # last = range(1,matrix_shape[num]+1)
    #함수값 생성
    last = [ s+1 for s in range( matrix_shape[num] * last_val, matrix_shape[num] * (last_val+1))]
    last_val += 1
    return ",".join(str(s) for s in last)


def create_list(num):
    # print(num)
    result = []
    result.append("[") 
    if num == mslen:
        #insert []
        result.append(create_last(num))
    else:
        for i in range(matrix_shape[num]):
            result.append(create_list(num+1))
    # print(num,":",result)
    result.append("],")
    return "".join(str(s) for s in result)

results = create_list(0)
print(results)