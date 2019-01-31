import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap=cv2.VideoCapture(0)

while True:
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,minNeighbors=5)
	for (x,y,w,h) in faces:
		roi_gray=gray[y:y+h,x:x+w]
		roi_color=frame[y:y+h,x:x+w]
		
		#recognise?

		img_item='my-image.png'
		cv2.imwrite(img_item,roi_gray)

		color=(255,0,0)#BGR
		stroke=2
		x_end=x+w
		y_end=y+h
		cv2.rectangle(frame,(x,y),(x_end,y_end),color,stroke)


	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF ==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()