import numpy as np   # importing NumPy - a Python library used for scientific calculations
import cv2  # importing opencv
import json # importing json (JavaScript Object Notation)
 
# Import HAAR Cascades for face detection
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()     # LBPH Face recognizer object
recognizer.read("trainer.yml")                        # Load the training data from the trainer to recognize the faces

labels = {"person_name": 2}                           # Declaring the name of the person as label
with open("labels.json",'r') as f:
    o_labels = json.load(f)                           # Loading the trained data 
    labels = {v:k for k,v in o_labels.items()}
print(o_labels.items())

cap = cv2.VideoCapture(0)                             # Camera object

while(True):
    ret, frame = cap.read()                           # Read the camera object
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Convert the Camera to gray
    faces2 = face_cascade.detectMultiScale(gray)      # Detect the faces and store the positions

    for (x, y, w, h) in faces2:                       # Frames LOCATION X,Y,Width,Height
        ROI_gray = gray[y:y+h, x:x+w]                 # The Face is isolated and cropped
        ROI_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(ROI_gray)      # Determine the id of the photo
        if conf>40:
            print(id_) 
            print(labels[id_])
        img_item = "img.png"
        cv2.imwrite(img_item, ROI_color)
        font = cv2.FONT_HERSHEY_TRIPLEX               # Font of the text
        name= labels[id_]
        color = (255,0,0) 
        stroke = 2
        cv2.putText(frame, name, (x,y-15), font, 2, color, stroke, cv2.LINE_AA)     # Name of the face recognized

        color = (0,255,255) #BGR 0-255
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)       # Rectangle around the face to show that face is detected
    
    # Display the resulting frame
    cv2.imshow('frame',frame)                         # show the captured image
    if cv2.waitKey(20) & 0xFF == ord('q'):            # Quit if the key is q
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
