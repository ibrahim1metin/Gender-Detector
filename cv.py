import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
class face_detector:
    def __init__(self):
        self.cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    def detections(self,cap):
        gray=cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
        face_coordinates=[]
        faces=self.cascade.detectMultiScale(gray,1.3,5)
        for face in faces:
            face_coordinates.append((face[0],face[1],face[2],face[3]))
        return face_coordinates
if __name__=="__main__":
    capture=cv2.VideoCapture(0)
    labels=["Female","Male"]
    det=face_detector()
    model = tf.keras.models.load_model('saved/model')
    print(model.summary())
    while True:
        _,frame=capture.read()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame=cv2.flip(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),1)
        faceDictionary={}
        frNumpy=np.array(frame)
        faces=det.detections(frame)
        for face in faces:
            faceNp=tf.expand_dims(tf.image.resize(frNumpy[face[0]:face[0]+face[2]:,face[1]:face[1]+face[3]:],(83,108))/255,0)
            res=np.argmax(model.predict(faceNp,batch_size=1))
            faceDictionary[face[0]]=labels[res]
        for face in faces:
            cv2.putText(frame, faceDictionary[face[0]], (face[0],face[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30,255,30), 2)
            cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(0,255,0),2)
        cv2.imshow("video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()