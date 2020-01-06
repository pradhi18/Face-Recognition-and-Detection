import cv2   # importing opencv
import os    # importing os for path
import numpy as np  # importing the NumPy library
from PIL import Image  # importing the image library 
import json  # importing json library 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Absolute directory of the path of image
image_dir = os.path.join(BASE_DIR, "images")  # Image directory

# Import HAAR Cascades for face detection
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')

current_id = 0
label_ids = {} # Dictionary of names and ids
y_labels = []  # List of names and ids
x_train = []   # training list

for root, dirs, files in os.walk(image_dir):    # Traversal of the path directory
	for file in files:
		if file.endswith(".jpeg"):
			path = os.path.join(root, file)     # joining the file found with the root(name)
			label = os.path.basename(root).replace(" ","-").upper()  # extracting the name from directory
			
			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			print(label_ids)
			pil_image = Image.open(path).convert("L") # L-grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)   # resize the image so that LBPH Recognizer can be trained
			image_array = np.array(final_image, "uint8")            # Converting the image to NumPy array       
			faces2 = face_cascade.detectMultiScale(image_array)     # Detect the faces and store the positions
			
			for (x,y,w,h) in faces2:                                # Frames LOCATION X,Y,Width,Height
				if id_ not in y_labels:
					ROI = image_array[y:y+h, x:x+w]
					x_train.append(ROI)      # Append the NumPy array to the list
					y_labels.append(id_)     # Append the ids to the list 

print(y_labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()   # LBPH Face recognizer object

with open("labels.json",'w') as f:
	json.dump(label_ids, f)


recognizer.train(x_train, np.array(y_labels))       # The recognizer is trained using the images
recognizer.save("trainer.yml")                      # Save the training data from the trainer 