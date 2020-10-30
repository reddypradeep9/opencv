import cv2
import face_recognition

im_dhoni=face_recognition.load_image_file('images/pradeep.jpg')
im_dhoni=cv2.cvtColor(im_dhoni,cv2.COLOR_BGR2RGB)
im_dhoni=cv2.resize(im_dhoni,(0,0),None,0.5,0.5)

im_yuvi=face_recognition.load_image_file('images/dhoni.jpg')
im_yuvi=cv2.cvtColor(im_yuvi,cv2.COLOR_BGR2RGB)


facloc=face_recognition.face_locations(im_dhoni)[0]
facencod=face_recognition.face_encodings(im_dhoni)[0]
cv2.rectangle(im_dhoni,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,255),2)

facloc_y=face_recognition.face_locations(im_yuvi)[0]
facencod_y=face_recognition.face_encodings(im_yuvi)[0]
cv2.rectangle(im_yuvi,(facloc_y[3],facloc_y[0]),(facloc_y[1],facloc_y[2]),(255,0,255),2)

results = face_recognition.compare_faces([facencod],facencod_y)
faceDis = face_recognition.face_distance([facencod],facencod_y)
print(results,faceDis)

cv2.putText(im_yuvi,f'{results} {round(faceDis[0],2)}',(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)


cv2.imshow("pradeep",im_dhoni)
cv2.imshow("sarada",im_yuvi)
cv2.waitKey(0)