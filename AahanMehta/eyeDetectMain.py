import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import dlib
# from subprocess import call
# call(["python", "your_file.py"])

while True:
    age = input("Enter age: ")
    if (age.isnumeric):
        if (int(age) > 8 and int(age) < 18):
            break
    else:
        print("Please enter a valid age number")
while True:
    gender = input("Enter gender (M/F): ")
    if gender == 'M' or gender == 'F':
        break
    else:
        print("Please enter either M or F")

filename = r"C:\Users\N Mehta\Documents\omotec\eyedetection\eyedetecData\{}_" + age + ".csv"
filename = filename.format(gender)
f = open(filename, 'w')

print("A paragraph will be displayed, which you have to read. While you read, your eye movements will be recorded")
time.sleep(5)
print("text will be displayed in ")
print('3')
time.sleep(1)
print('2')
time.sleep(1)
print('1')
time.sleep(1)
print("Dolphins are regarded as the friendliest creatures in the sea and stories of them helping drowning sailors have been common since Roman times. The more we learn about dolphins, the more we realize that their society is more complex than people previously imagined. They look after other dolphins when they are ill, care for pregnant mothers and protect the weakest in the community, as we do. Some scientists have suggested that dolphins have a language but it is much more probable that they communicate with each other without needing words.")

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x,y), (x1, y1), (0,0,255), 3)
        # printing on opencv output
        landmarks = predictor(gray, face)

        x1 = landmarks.part(36).x
        y1 = landmarks.part(36).y
        x2 = landmarks.part(39).x
        y2 = landmarks.part(39).y
        frame1 = frame[(y1-20):(y2+20), x1:x2, :]
        frame1 = cv2.resize(frame1, (200,200))
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15,15), 0)
        _,thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        cont,har = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for i in cont:
            area = cv2.contourArea(i)   
            if area > 1000:
                count += 1
                if count == 1:
                    x,y,w,h = cv2.boundingRect(i)
                    cv2.rectangle(frame1, (x,y), (x+w,y+h), (255,255,255, 2))
                    circle_x = (x+w+x)//2
                    circle_y = (y+h+y)//2
                    f.write(str(circle_x) + ',' + str(circle_y) + ', ')
                    coords = str(circle_x) + ", " + str(circle_y)
                    cv2.putText(frame1, coords, (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,200), 2 )
                    cv2.circle(frame1, (circle_x, circle_y), 5, (255,255,255), 3)
                    cv2.drawContours(frame1, [i], -1, (0,0,255), 3)


        x1 = landmarks.part(42).x
        y1 = landmarks.part(42).y
        x2 = landmarks.part(45).x
        y2 = landmarks.part(45).y
        frame2 = frame[(y1-20):(y2+20), x1:x2, :]
        frame2 = cv2.resize(frame2, (200,200))
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15,15), 0)
        _,thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        cont,har = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for i in cont:
            area = cv2.contourArea(i)   
            if area > 1000:
                count += 1
                if count == 1:
                    x,y,w,h = cv2.boundingRect(i)
                    cv2.rectangle(frame2, (x,y), (x+w,y+h), (255,255,255, 2))
                    circle_x = (x+w+x)//2
                    circle_y = (y+h+y)//2
                    f.write(str(circle_x) + ',' + str(circle_y) + '\n')
                    coords = str(circle_x) + ", " + str(circle_y)
                    cv2.putText(frame2, coords, (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,200), 2 )
                    cv2.circle(frame2, (circle_x, circle_y), 5, (255,255,255), 3)
                    cv2.drawContours(frame2, [i], -1, (0,0,255), 3)
        #cv2.rectangle(frame, (x1,(y1-20)), (x2,(y2+20)), (100, 200, 100), 2)

        cv2.imshow("left eye", frame1)
        cv2.imshow("right eye", frame2)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
f.close()
cv2.destroyAllWindows() 