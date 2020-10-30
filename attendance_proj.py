import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path=os.path.join(os.getcwd(),'images')
images=[]
class_names=[]
mylist=os.listdir(path)

for im in mylist:
    curimg=cv2.imread(f'{path}/{im}')
    images.append(curimg)
    class_names.append(im.split('.')[0])
def markattendance(name):
    with open('attendance.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtstring=now.strftime('%d-%b-%Y'+','+'%H:%M:%S')
            f.writelines(f"\n{name},{dtstring.split(',')[0]},{dtstring.split(',')[1]}")
        print(namelist)


def encodings(images):
    enlist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encod=face_recognition.face_encodings(img)[0]
        enlist.append(encod)
    return enlist
encodelistknown=encodings(images)
print("encode complete")

cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facescurframe=face_recognition.face_locations(imgs)
    encodecurframe=face_recognition.face_encodings(imgs,facescurframe)

    for encodeface,faceloc in zip(encodecurframe,facescurframe):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        facedist=face_recognition.face_distance(encodelistknown,encodeface)

        matchindex=np.argmin(facedist)

        if facedist[matchindex] < 0.50:
            name = class_names[matchindex].upper()
            markattendance(name)
        else:
            name = 'Unknown'
        y1,x2,y2,x1=faceloc
        y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)


