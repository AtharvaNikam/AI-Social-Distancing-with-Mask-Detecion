from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from imutils.video import VideoStream
import imutils
from scipy.spatial import distance as dist
import pyttsx3
#from engine import Engine

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)
#engine.say("Hello, How are you ?")
engine.runAndWait()

'''
def detect_and_predict_mask(frame,faceNet,maskNet):
    #grab the dimensions of the frame and then construct a blob
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    
    faceNet.setInput(blob)
    detections=faceNet.forward()
    
    #initialize our list of faces, their corresponding locations and list of predictions
    
    faces=[]
    locs=[]
    preds=[]
    
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    
    
        if confidence>0.5:
        #we need the X,Y coordinates
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
        
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
        
            faces.append(face)
            locs.append((startX,startY,endX,endY))
        
        #only make a predictions if atleast one face was detected
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=12)
        
        return (locs,preds)

'''
def speak(str):
    engine.say(str)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
face_model = cv2.CascadeClassifier('E:\AI\opencv\data\haarcascades\haarcascade_frontalface_default.xml') 
#prototxtPath=os.path.sep.join([r'C:\Users\ATHARVA\Documents','deploy.prototxt'])
#weightsPath=os.path.sep.join([r'C:\Users\ATHARVA\Documents','res10_300x300_ssd_iter_140000.caffemodel'])
#faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)
maskNet=load_model(r'C:\Users\ATHARVA\Documents\mobilenet_v2.model')

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


while True:
    
    status , photo = cap.read()

    #photo = imutils.resize(photo,width=400)

    #mask detection
    gray=photo
    
    faces=face_model.detectMultiScale(photo)  

    for (x,y,w,h) in faces:
        face_img=gray[y:y+w,x:x+w]
        
        '''resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=maskNet.predict(reshaped)'''
        #face_img=cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
        face_img=cv2.resize(face_img,(224,224))
        face_img=img_to_array(face_img)
        reshaped=np.reshape(face_img/255,(1,224,224,3))
        result=maskNet.predict(reshaped)
        label=0 if result[0][0]>0.8 else 1
      
        cv2.rectangle(photo,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(photo,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(photo, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    


    #social distancing
    #faces = face_model.detectMultiScale(photo)
    l = len(faces)
    photo = cv2.putText(photo, str(len(faces))+" Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
    stack_x = []
    stack_y = []
    stack_x_print = []
    stack_y_print = []
    global D

    if len(faces) == 0:
        pass
    else:
        for i in range(0,len(faces)):
            x1 = faces[i][0]
            y1 = faces[i][1]
            x2 = faces[i][0] + faces[i][2]
            y2 = faces[i][1] + faces[i][3]

            mid_x = int((x1+x2)/2)
            mid_y = int((y1+y2)/2)
            stack_x.append(mid_x)
            stack_y.append(mid_y)
            stack_x_print.append(mid_x)
            stack_y_print.append(mid_y)
            photo = cv2.circle(photo, (mid_x, mid_y), 3 , [255,0,0] , -1)
            photo = cv2.rectangle(photo , (x1, y1) , (x2,y2) , [0,255,0] , 2)
        
        if len(faces) == 2:
            D = int(dist.euclidean((stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())))
            photo = cv2.line(photo, (stack_x_print.pop(), stack_y_print.pop()), (stack_x_print.pop(), stack_y_print.pop()), [0,0,255], 2)
        else:
            D = 0

        if D<250 and D!=0:
            photo = cv2.putText(photo, "You are in Danger", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,2, [0,0,255] , 4)
            speak("You are in Danger")

        photo = cv2.putText(photo, str(D/10) + " cm", (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0) , 2, cv2.LINE_AA)

        cv2.imshow('Camera' , photo)
        if cv2.waitKey(100) == 13:
            break

cv2.destroyAllWindows()