# before running this project 
# install numpy then opencv
# to close output window press q  OR delete the terminal

import cv2

#make sure pathing is correct acc to ur directory

face_pbtxt = "model/opencv_face_detector.pbtxt" 
face_pb = "model/opencv_face_detector_uint8.pb"
age_prototxt="model/age_deploy.prototxt"
age_model="model/age_net.caffemodel"
gender_prototxt="model/gender_deploy.prototxt"
gender_model="model/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(face_pb, face_pbtxt)
ageNet = cv2.dnn.readNet(age_model, age_prototxt)
genNet =  cv2.dnn.readNet(gender_model, gender_prototxt)

MODEL_MEAN_VALUES =[104,117,123]
MODEL_MEAN_VALUES2 = [78.4263377603,87.7689143744,114.895847746]
age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male', 'Female']

def faceBox(faceNet,frame):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (277,277),MODEL_MEAN_VALUES,swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox=[]
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if(confidence>0.75):
            x1=int(detection[0,0,i,3]*frame_w)
            y1=int(detection[0,0,i,4]*frame_h)
            x2=int(detection[0,0,i,5]*frame_w)
            y2=int(detection[0,0,i,6]*frame_h)
            bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return frame,bbox


cap = cv2.VideoCapture(0)



while True:
  ret, frame = cap.read()
  frame,bbox = faceBox(faceNet,frame)
  for bb in bbox:
      face=frame[bb[1]:bb[3],bb[0]:bb[2]]
      blob = cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES2, swapRB=False)
      genNet.setInput(blob)
      gender_Prediction =  genNet.forward()
      gender = gender_classifications[gender_Prediction[0].argmax()]


      ageNet.setInput(blob)
      age_Pridictions = ageNet.forward()
      age = age_classifications[age_Pridictions[0].argmax()]


      label ="{},{}".format(gender,age)
      cv2.putText(frame,label,(bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2,cv2.LINE_AA)
  

  cv2.imshow("Frame",frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()