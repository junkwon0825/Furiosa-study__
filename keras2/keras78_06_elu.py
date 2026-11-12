# 2017년 구글에서 swish 발표
# x에 sigmoid() 곱한 함수
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
alpha = 100

def elu(x,alpha):
    return (x>0) * x + (x<0)*(alpha* (np.exp(x)-1))

# elu = lambda x, alpha : (x>0) * x + (x<0)*(alpha* (np.exp(x)-1))

y = elu(x, alpha)

plt.plot(x,y)
plt.grid()
plt.show()




