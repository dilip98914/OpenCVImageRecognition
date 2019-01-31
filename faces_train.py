import os
import numpy as np
from PIL import Image
import cv2

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
IMG_DIR=os.path.join(BASE_DIR,'images')
# dirname is true for every system

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')


current_id=0
label_ids={}
y_labels=[]
x_train=[]


for root,dirs,files in os.walk(IMG_DIR):
	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			path=os.path.join(root,file)
			label=os.path.basename(os.path.dirname(path))
			
			if label in label_ids:
				pass
			else:
				label_ids

			pil_image=Image.open(path).convert('L')
			image_array=np.array(pil_image,'uint8')
			print(image_array)
			faces=face_cascade.detectMultiScale(image_array,minNeighbors=5)
			for (x,y,w,h) in faces:
				roi=image_array[y:y+h,x:x+w]
				x_train.append(roi) 