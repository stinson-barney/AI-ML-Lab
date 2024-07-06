import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QRadioButton, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import modelcode as md


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create name text box
        self.text_name = QLineEdit()

        # Create radio buttons
        self.radio_male = QRadioButton('Male')
        self.radio_female = QRadioButton('Female')
        self.radio_male.setChecked(True)  # Set default as Male

        # Create layouts for name and gender
        layout_name_gender = QHBoxLayout()
        layout_name_gender.addWidget(QLabel('Name: '))
        layout_name_gender.addWidget(self.text_name)
        layout_name_gender.addWidget(QLabel('Gender: '))
        layout_name_gender.addWidget(self.radio_male)
        layout_name_gender.addWidget(self.radio_female)

        # Create image box
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 200)

        # Create video box
        self.video_label = QLabel()
        self.video_label.setFixedSize(200, 200)

        # Create buttons
        self.btn_upload_video = QPushButton('Upload Video')
        self.btn_upload_video.clicked.connect(self.openVideo)
        self.btn_generate_report = QPushButton('Generate Report')
        self.btn_generate_report.clicked.connect(self.generateReport)

        # Create layouts for video and image
        layout_video_image = QHBoxLayout()
        layout_video_image.addWidget(self.video_label)
        layout_video_image.addWidget(self.image_label)

        # Create layout for buttons
        layout_buttons = QHBoxLayout()
        layout_buttons.addWidget(self.btn_upload_video)
        layout_buttons.addWidget(self.btn_generate_report)

        # Create main layout
        layout_main = QVBoxLayout()
        layout_main.addLayout(layout_name_gender)
        layout_main.addLayout(layout_video_image)
        layout_main.addLayout(layout_buttons)

        self.setLayout(layout_main)
        self.setWindowTitle('My Application')
        self.show()

        self.video_path = ""

    def openVideo(self):
        filename, _ = QFileDialog.getOpenFileName(self) #, "Select Video", "", "Video Files (*.mov)"
        if filename:
            # self.video_label.setText(filename)
            self.video_path = filename

    def generateReport(self):
        gender = 'Male' if self.radio_male.isChecked() else 'Female'
        name = self.text_name.text()
        if self.video_path:
            print(f"Name: {name}\nGender: {gender}\nVideo Path: {self.video_path}")
            prediction = md.predict(self.video_path[:-4]+'.csv')
            pixmap = QPixmap(report(self.video_path, name, gender, prediction))
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            imgprocess(self.video_path)
            
            self.video_label.setText('Name:'+name+' Gender:'+gender+'\nprediction result:'+prediction)
            # report(self.video_path)
        else:
            print("Please select a video file.")
def imgprocess(path):
    dataStoredAt = path[:-4]+'.csv'
    vid = cv2.VideoCapture(path)
    f = open(dataStoredAt, 'w')
    blurValue = (7,7)
    area_value = 8000
    x1,y1 = 0,0
    # finally tracking starts 
    while True:
        ret,frame1 = vid.read()
        
        if ret:
            # frame = cv2.rotate(frame1, cv2.cv2.ROTATE_90_CLOCKWISE)
            frame = frame1[:,120:,:]
            # frame = frame[refPoint[0][1]:refPoint[1][1],refPoint[0][0]:refPoint[1][0]]
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,blurValue,0) 
            _,thresh = cv2.threshold(gray,45,255,cv2.THRESH_BINARY_INV)
            
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

def report(path, n, s, pred):
    p = path[:-4]+'.csv'
    file = pd.read_csv(r'{}'.format(p)) 
    x = file.iloc[:,0].values
    y = file.iloc[:,1].values


    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel("X-axis")

    ax.set_zlabel("Y-axis")

    ax.set_ylabel("time")
    plt.title(n+'-'+s+'\nprediction:'+pred)
    ax.plot(x,range(len(x)),y)
    graphpath = r''+path+'.png'
    plt.savefig(graphpath)
    return(graphpath)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())