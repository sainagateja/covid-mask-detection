import numpy as np 
import cv2
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
data=[]
while 1:
    ret,img=cap.read()
    
    
    faces=face_cascade.detectMultiScale(img)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = img[y:y+h, x:x+w , : ]
        face=cv2.resize(face,(50,50))
        cv2.imshow('image',img)
        print(len(data))
        
        if len(data)<=200:
            data.append(face) 
        
    if (cv2.waitKey(30) & 0xFF==ord('q'))   or  len(data)>200:
        break
    
    np.save("with_mask.npy",data)
cap.release()
cv2.destroyAllWindows()   

    
    

 
 