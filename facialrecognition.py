import face_recognition
import imutils
import pickle
import cv2
import os

face_cascade = cv2.CascadeClassifier(r'C:\Users\max\AppData\Local\Programs\Python\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')

data = pickle.loads(open('face_enc',"rb").read())

print("Streaming started")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    rgb = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []
    for encodings in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encodings)
        name = "unknown"

        if True in matches:
            matchedIdxs = [i for(i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frames, (x, y), (x+w, y+h), (0,255,0),2)
            cv2.putText(frames, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0,255,0),2)
            
    cv2.imshow("Frame", frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stream stopped")
        break
    

video_capture.release()
cv2.destroyAllWindows()
        
    
