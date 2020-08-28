# Import opencv
import cv2


# Load casscade classfier 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


#Create function to detect face, eye, smile
def detect(gray, frame):
    """
    This Function is to detect face, eyes, smile and it takes 2 parameter gray image and color image
    we will do:
        - detect face and draw rectangle arrounf it (R)
        - detect eyes inside detected face and draw rectangle arround it (G)
        - detect smile inside detected face and draw rectangle arround it (B)
    """
    
    # detected faces in image or video
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # go through each face in givin image and grab position info
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # grab roi(region of interest) which is in our case the face postion so we dont take a lot of time to search eyes in whole image
        roi_gray  = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        # detected eyes and smiles  in image or video
        #eyes   = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 25)
        # go through  each eye and smile  in givin roi  and grab postion info 
        #for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(roi_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
             
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            
    return frame



cap = cv2.VideoCapture(0)

while True:
    
    _, frame = cap.read()
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected = detect(gray, frame)
    cv2.imshow('Video', detected)
    
    k = cv2.waitKey(1) & 0xFF 
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()







