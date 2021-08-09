from PIL import Image, ImageOps
from keras.models import load_model
import cv2
import numpy as np

model = load_model('C:\\Users\\GOLI SHYAM\\Desktop\\project final ML\\mask_detector.model')

face_clsfr=cv2.CascadeClassifier('C:\\Users\\GOLI SHYAM\\Desktop\\project final ML\\haarcascade_frontalface_default.xml')

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

choice = int(input("A) Want to go for live camera then press : 1 \n B) want to go for inputed video then press : 2 \n C) want to go for image then press : 3 \n insert input : \t"))
if choice == 1:
        source=cv2.VideoCapture(0)
if choice == 2:
        source=cv2.VideoCapture("C:\\Users\\GOLI SHYAM\\Desktop\\project final ML\\video5.webm") 
while(True):
    if choice == 1 or choice == 2:
        ret,img=source.read()
    elif choice == 3:
        source = cv2.imread('C:\\Users\\GOLI SHYAM\\Desktop\\project final ML\\download.jpg')
        img = cv2.resize(source,(500,500))
    else:
        break
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break

cv2.destroyAllWindows()
source.release()