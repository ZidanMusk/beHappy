import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import dlib
import math
#from skimage import io


predictor_path = 'shape_predictor_68_face_landmarks.dat'
faces_folder_path = './faces/'

#os.system('pwd')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('test6.jpg',-1)
happy = cv2.imread('angry.png', -1)
happy = cv2.cvtColor(happy, cv2.COLOR_BGRA2RGBA)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
facelocs = detector(img, 3)
faces = np.ndarray(shape=(1, 48*48), dtype=np.uint8)
for box in facelocs:
    face = gray[box.top(): box.bottom(), box.left(): box.right()]
    face = cv2.resize(face, (48, 48), interpolation = cv2.INTER_CUBIC)
    face = np.array(face)
    face = np.reshape(face, 48*48)
    faces = np.vstack((faces, face))
faces = np.delete(faces, 0, 0)
for box in facelocs:
    shapes = predictor(img, box)
    
    emoji = cv2.resize(happy, (box.right() - box.left(), box.bottom() - box.top()), interpolation = cv2.INTER_CUBIC)
    rows, cols, d = emoji.shape
    
    diag = np.sqrt((box.bottom() - box.top())**2 + (box.right() - box.left())**2)
    wexp = (diag - (box.right() - box.left()))/2
    hexp = (diag - (box.bottom() - box.top()))/2
    wexp = int(wexp)
    hexp = int(hexp)
    emoji = cv2.copyMakeBorder(emoji, hexp, hexp, wexp, wexp, cv2.BORDER_REPLICATE)
    
    src = np.array([(cols*(20.0/512.0) + wexp, rows*(200.0/512.0) + hexp), (cols*(256.0/512.0)+ wexp, rows*(495.0/512.0)+ hexp), (cols*(492.0/512.0)+ wexp, rows*(200.0/512.0)+ hexp)])
    #src = np.array([(20, 200), (256, 495), (492, 200)])
    src = np.uint8(src)
    src = np.float32(src)
    dest = np.array([(shapes.part(0).x - box.left()+ wexp, shapes.part(0).y - box.top()+ hexp),(shapes.part(8).x-box.left()+ wexp, shapes.part(8).y - box.top()+ hexp),(shapes.part(16).x - box.left()+ wexp, shapes.part(16).y - box.top()+ hexp)])
    dest = np.float32(dest)
    rows, cols, d = emoji.shape
    trans = cv2.getAffineTransform(src,dest)
    emoji = cv2.warpAffine(emoji, trans, (cols, rows))
    
    #print(happy)
    for c in range(0,3):
        img[box.top() - hexp: box.bottom() + hexp, box.left() - wexp: box.right() + wexp, c] = emoji[:,:,c] * (emoji[:,:,3]/255.0) + img[box.top() - hexp: box.bottom() + hexp, box.left() - wexp: box.right() + wexp, c] * (1.0 - emoji[:,:,3]/255.0)
plt.imsave('s.jpg',img)
