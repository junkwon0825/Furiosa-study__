#48_copy
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
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

################################### 데이터 증폭 ###########################
datagen = ImageDataGenerator(
    rescale=1./255, #형변환
    # horizontal_flip = True, #수평 뒤집기, (좌우반전)
    # vertical_flip = True, #수직 뒤집기 (상하 반전)
    # width_shift_range = 0.1, #평행이동
    height_shift_range = 0.1, 
    rotation_range = 15, #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,
    # shear_range = 0.7, #좌표하나하나 고정하고 다른 몇개 좌표를 이동
    fill_mode = 'nearest',

) #클래스 전체 적용하면 사진 이상해짐

it = datagen.flow(arr, 
                  batch_size=1,)
print(it) #<keras.preprocessing.image.NumpyArrayIterator object at 0x000001ED20D97F70>
# print(it.next()) #파이썬 3.10까지
print(next(it)) #파이썬 3.11 이후
print(next(it).shape) #(1, 100, 100, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5)) #1행5열
for i in range(5):
    batch = next(it) #5번 이터레이터 실행시킴 
    # print(batch.shape)    
    batch = batch.reshape(100,100,3) #reshape랑 데이터 없애는거랑 다르다 차원줄이는건 가능

    ax[i].imshow(batch)
    ax[i].axis('off')
plt.show()





# np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path + "keras48_me.npy", arr=arr)









