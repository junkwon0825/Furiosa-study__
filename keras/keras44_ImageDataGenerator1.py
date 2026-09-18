import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, #형변환
    horizontal_flip = True, #수평 뒤집기,
    vertical_flip = True, #수직 뒤집기
    width_shift_range = 0.1, #평행이동
    height_shift_range = 0.1, 
    rotation_range = 5, #각도조절(정해진 각도만큼 이미지 회전)
    zoom_range = 1.2,
    shear_range = 0.7, #좌표하나하나 고정하고 다른 몇개 좌표를 이동
    fill_mode = 'nearest',

) #클래스

test_datagen = ImageDataGenerator(
    rescale=1./255,
) #평가 훈련할 데이터는 rescale만 하면됨. 시험지는 변환시키면 안됨 항상.

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, #경로
    target_size=(200,200), #마음대로 써도 알아서 늘리거나 줄여줌 , 모든 사진들이 크기가 다르기때문에 알아서 맞춤
    batch_size=10, #이미지 batchsize ()
    class_mode='binary', #이진분류
    color_mode='grayscale', #흑백
    shuffle=True,
)

#Found 160 images belonging to 2 classes.//

xy_test = test_datagen.flow_from_directory(
    path_test, #경로
    target_size=(200,200), #마음대로 써도 알아서 늘리거나 줄여줌 , 모든 사진들이 크기가 다르기때문에 알아서 맞춤
    batch_size=10, #이미지 batchsize ()
    class_mode='binary', #이진분류
    color_mode='grayscale', #흑백
    shuffle=False, #필요가 없음 test는 섞을 필요가 없음
)

# Found 120 images belonging to 2 classes.

print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x000002293FE37FA0>
# print(xy_train.next()) #x,y데이터 분리된거 확인, 이터레이터의 첫번째를 보여줘
# print(xy_train.next()) #두번째 이터레이터 보여줘

print(xy_train[0][0].shape) #(10, 200, 200, 1)
print(xy_train[0][1].shape) #(10,)

# print(xy_train[16][0]) #여기서부터 에러, 이유는 160장에 batch10이니까 

print(type(xy_train))  #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'> -> x,y데이터 인데 튜플행태로 저장.리스트랑 비슷,수정x
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>


#이미지 데이터 수치화로 cnn 돌린다.
















