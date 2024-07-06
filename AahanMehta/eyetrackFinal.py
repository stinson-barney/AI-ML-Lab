import fractions
import cv2
import numpy as np

cropping = False
refPoint = []
path = r'C:\Users\Admin\Desktop\colleger work\Semester2\AI&ML Lab\AahanMehta\vid2.mp4'
dataStoredAt = r'C:\Users\Admin\Desktop\colleger work\Semester2\AI&ML Lab\AahanMehta\vid2.csv'
# count = 0
x_start, y_start, x_end, y_end = 0, 0, 0, 0
area_value = 1600
flag = False
blurValue = (7,7)
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
    
vid = cv2.VideoCapture(path)

ret,frame = vid.read()

cv2.namedWindow('img')
cv2.setMouseCallback('img',crop)
while 1:
    f1 = frame
    # f1 = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    # f1.frame.copy()
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

def nothing(x):
    pass

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('slider')

cv2.createTrackbar("thresh", 'slider', 5, 100, nothing)
cv2.setTrackbarPos("thresh", "slider", 35)

# vid = cv2.VideoCapture(r'C:\Users\Admin\Downloads\vid6.mp4')
while True:
    ret,frame1 = vid.read()
    
    if ret:
        # frame = cv2.rotate(frame1, cv2.cv2.ROTATE_90_CLOCKWISE)
        frame = frame1.copy() 
        frame = frame[refPoint[0][1]:refPoint[1][1],refPoint[0][0]:refPoint[1][0]]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,blurValue,0)
        threshvalue = cv2.getTrackbarPos("thresh", 'slider')
        _,thresh = cv2.threshold(gray,threshvalue,255,cv2.THRESH_BINARY_INV)
        
        cont,har = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in cont:
            area = cv2.contourArea(i)
            if area>1600:
                x,y,w,h = cv2.boundingRect(i)
                cv2.drawContours(frame,[i],-1,(0,200,0),2)
                cv2.circle(frame,(x+(w//2),y+(h//2)),60,(200,0,0),3)
                # ptsx.append(x-40)
                # ptsy.append(y+30)
                # pts.append([x-40,y+30])
        cv2.imshow('a',frame)
        cv2.imshow('C',gray)
        cv2.imshow('b',thresh)
        
    else:
        break
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vid.release()

vid = cv2.VideoCapture(path)
f = open(dataStoredAt, 'w')
x1,y1 = 0,0
# finally tracking starts 
while True:
    ret,frame1 = vid.read()
    
    if ret:
        # frame = cv2.rotate(frame1, cv2.cv2.ROTATE_90_CLOCKWISE)
        frame = frame1.copy() 
        frame = frame[refPoint[0][1]:refPoint[1][1],refPoint[0][0]:refPoint[1][0]]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,blurValue,0) 
        _,thresh = cv2.threshold(gray,threshvalue,255,cv2.THRESH_BINARY_INV)
        
        cont,har = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in cont:
            area = cv2.contourArea(i)
            if area> area_value:
                x,y,w,h = cv2.boundingRect(i)
                cv2.drawContours(frame,[i],-1,(0,200,0),2)
                x1 = x + (w//2)
                y1 = y + (h//2)
                cv2.circle(frame,(x1,y1),60,(200,0,0),3)
                f.write(str(x1) + ',' + str(y1) + '\n')
                # ptsx.append(x-40)
                # ptsy.append(y+30)
                # pts.append([x-40,y+30])
            elif area == 0:
                f.write(str(x1) + ',' + str(y1) + '\n')

        cv2.imshow('a',frame)
        cv2.imshow('C',gray)
        cv2.imshow('b',thresh)
        
    else:
        break
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
vid.release()
f.close()