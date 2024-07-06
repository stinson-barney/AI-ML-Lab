import fractions
import cv2
import numpy as np

cropping = False
refPoint = []
# count = 0
x_start, y_start, x_end, y_end = 0, 0, 0, 0
flag = False
def crop(event,x,y,flags,param):
    global cropping,flag
    global refPoint
    global x_start, y_start, x_end, y_end
    if event==cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]
        flag =  True
        return(refPoint)
    
vid = cv2.VideoCapture(r'C:\Users\Admin\Downloads\vid7.mp4')

ret,frame = vid.read()

cv2.namedWindow('img')
cv2.setMouseCallback('img',crop)
while 1:
    f1 = frame.copy()
    if not cropping:
        cv2.imshow("img", f1)

    elif cropping:
        cv2.rectangle(f1, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("img", f1)
    cv2.waitKey(1)

    if flag:
        break
    

cv2.destroyAllWindows()
print(refPoint)

fr = frame[refPoint[0][1]:refPoint[1][1],refPoint[0][0]:refPoint[1][0]]


ptsx = []
ptsy = []
pts = []
# vid = cv2.VideoCapture(r'C:\Users\Admin\Downloads\vid6.mp4')
while True:
    ret,frame = vid.read()
    
    if ret:
        frame = frame[refPoint[0][1]:refPoint[1][1],refPoint[0][0]:refPoint[1][0]]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(9,9),0)
    
        _,thresh = cv2.threshold(gray,55,255,cv2.THRESH_BINARY_INV)
        
        cont,har = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in cont:
            area = cv2.contourArea(i)
            if area>1600:
                x,y,w,h = cv2.boundingRect(i)
                cv2.drawContours(frame,[i],-1,(0,200,0),2)
                cv2.circle(frame,(x+(w//2),y+(h//2)),60,(200,0,0),3)
                ptsx.append(x-40)
                ptsy.append(y+30)
                pts.append([x-40,y+30])
        cv2.imshow('a',frame)
        cv2.imshow('C',gray)
        cv2.imshow('b',thresh)
        
    else:
        break
    cv2.waitKey(40)
    
cv2.destroyAllWindows()
vid.release()