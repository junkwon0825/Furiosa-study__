import numpy as np

x1 = np.array([1,2,3]) 
#거의 연산할때는 numpy array연산으로 함 1차원 백터 (3,)
print("x1 = ", x1.shape)

x2 = np.array([1,2,3]) #(1,3)
print("x2 = ", x2.shape)

x3 = np.array([[1,2], [3,4]]) #(2,2)
print("x3 = ", x3.shape)

x4 = np.array([[1,2], [3,4], [5,6]]) #(3,2)
print("x4 = ", x4.shape)

x5 = np.array([[[1,2],[3,4],[5,6]]]) #(1,3,2)
print("x5 = ", x5.shape)

x6 = np.array([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]]) #(2,3,2)
print("x6 = ", x6.shape)

x8 = np.array([[[1,2,3]], [[4,5,6]]]) #(2,1,3)
print("x8 = ", x8.shape)

x9 = np.array([[[[1]]], [[[2]]]]) #(2,1,3)
print("x9 = ", x9.shape)