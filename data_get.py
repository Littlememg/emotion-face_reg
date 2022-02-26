# coding: utf-8
import os
import cv2
import dlib
import json
import numpy as np

detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
#detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

data_path = './data/'
label_list = []
data = np.zeros((1,128))   #定义一个128维的data
#num = 0
for file in os.listdir(data_path):
    if '.jpg' in file or '.png' in file:
        label_name = file.split('_')[0]   #获取标签名
        print('Current image: ', file)
        print('Current label: ', label_name)
        
        img = cv2.imread(data_path + file)       #使用opencv读取图像数据
        if img.shape[0]*img.shape[1] > 47000:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            
        dets = detector(img, 1)    #使用检测算子检测人脸，返回的是所有的检测到的人脸区域
        for i, j in enumerate(dets):
            rec = dlib.rectangle(j.rect.left(), j.rect.top(), j.rect.right(), j.rect.bottom())
            #rec = dlib.rectangle(j.left(), j.top(), j.right(), j.bottom())
            shape = shape_predictor(img, rec)     #获取landmark
            face_descriptor = face_recognition.compute_face_descriptor(img, shape)  #使用resNet获取128维的人脸特征向量
            face_128 = np.array(face_descriptor).reshape((1, 128))
            data = np.concatenate((data, face_128))    #拼接到事先准备好的data当中去
            label_list.append(label_name)
            
        #name = labelName +str(num)+'_c.png'
        #cv2.imwrite(name, img)
        #num += 1

data = data[1:, :]
np.savetxt('faced.txt', data, fmt='%f')     #保存人脸特征向量合成的矩阵到本地

label = open('label_list.txt','w')                                      
json.dump(label_list, label)       #用json保存到本地
label.close()

