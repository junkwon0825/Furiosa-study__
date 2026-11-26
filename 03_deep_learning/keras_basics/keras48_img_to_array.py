from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

path = 'c:/study/_data/image/' #절대경로
# path = './_data/image/' #상대경로

img = load_img(path + '1.jpg', target_size = (100,100))

print(img)
# <PIL.Image.Image image mode=RGB size=100x100 at 0x16DEF10D690>
# print(type(img)) <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape) #(100, 100, 3)
print(type(arr)) # <class 'numpy.ndarray'> 이미지를 numpy이미지로 바꿈

arr = np.expand_dims(arr,axis=0) #0번째 1넣음 차원 증가
print(arr.shape) #(1, 100, 100, 3)

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + "keras48_me.npy", arr=arr)









