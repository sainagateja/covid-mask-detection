import numpy as np
import cv2
import winsound


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

with_mask=np.load('with_mask.npy')
without_mask=np.load("without_mask.npy")

#print(with_mask.shape)

with_mask = with_mask.reshape(200,50*50*3)
without_mask = without_mask.reshape(200,50*50*3)

#combining both
X = np.r_[with_mask,without_mask]

labels = np.zeros(X.shape[0])

#without--->1
#with---->0
labels[200:]=1.0

names = {0:"Mask", 1:"No Mask"}


#suffling the data and then 25 % is for training and rest os for testing
x_train,x_test,y_train,y_test =train_test_split(X,labels,test_size=0.25)

 
#dimensionality reduction(decomposition)
#PCA principle component analysis

pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)

print(x_train.shape,"shape after transform")

#y_train=pca.fit_transform(y_train)    ------> no decomposition for y_train



svm = SVC()
svm.fit(x_train,y_train)


x_test=pca.transform(x_test)

#y_pred=svm.predict(x_test)


#x_test = pca.transform(x_test)
y_pred =svm.predict(x_test)

print(y_pred,"prediciton after train")



cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

while 1:
    ret,img=cap.read()
    
    
    faces=face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = img[y:y+h, x:x+w , : ]
        
        face=cv2.resize(face,(50,50))
        #print(face.shape)
        face=face.reshape(1,50*50*3)
        face=pca.transform(face)
        pred=svm.predict(face)
        
        
        
        
        
        
        
        
        
        #face = face.reshape(50*50*3)

        n=names[int(pred)]
        
        if int(pred)==1:
            for i in range(1):
                winsound.Beep(2500,500)
            
        cv2.putText(img, n , (x+5,y-5), font, 1, (255,255,255), 2)
        print(n)
        
        cv2.imshow("result",img)
        
        
        
    if (cv2.waitKey(30) & 0xFF==ord('q'))  :
        break
    
    
cap.release()
cv2.destroyAllWindows()   




'''
        face=cv2.resize(face,(50,50))
        face = face.reshape(1,-1)
        pred=svm.predict(face)[0]
'''



'''

face=cv2.resize(face,(50,50))
face=pca.transform(face)
pred=svm.predict(face)


'''