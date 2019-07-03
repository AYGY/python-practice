import socket
import os
import struct
import sys
import time
import threading

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import client1.test as test
import client1.test3 as rec
import face





# client()



class Home(QMainWindow):

    def __init__(self):
        super(Home, self).__init__()
        #QtGui.QWidget.__init__(self)
        self.style = """2.2.3 
                        QPushButton{background-color:grey;color:white;}
                        #window{ background-image: url(background1.jpg); }
                    """
        self.setStyleSheet(self.style)
        self.initUI()

    def initUI(self):

        # try:
        #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     s.connect(('127.0.0.1', 9999))
        # except socket.error as msg:
        #     print(msg)
        #     print(sys.exit(1))




        self.resize(650, 480)
        self.statusBar().showMessage('Ready')
        self.setObjectName("window")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.center()

        # self.rec = face.Recognition()
        widget = QWidget()
        label = QLabel()
        label.setText("<font size=%s><B>%s</B></font>" % ("15", "Face Recognition System"))
        log = QPushButton("Log", self)
        widget.setStatusTip('  ')
        #start.resize(50, 25)
        recognize = QPushButton("Recognize", self)
        quit = QPushButton("Quit", self)
        #quit.resize(50,25)
        log.clicked.connect(self.logClicked)
        recognize.clicked.connect(self.recognizeClicked)
        quit.clicked.connect(self.quitClicked)

        vbox1 = QVBoxLayout()  # 垂直布局
        vbox2 = QVBoxLayout()
        vbox3 = QVBoxLayout()
        vbox4 = QVBoxLayout()

        #两边空隙填充
        label1 = QLabel()
        label1.resize(50,50)
        label2 = QLabel()
        label2.resize(50, 50)
        vbox1.addWidget(label1)
        #vbox2.addWidget(label)
        vbox4.addWidget(log)
        vbox4.addWidget(recognize)
        vbox4.addWidget(quit)
        vbox3.addWidget(label2)
        # 按钮两边空隙填充
        label3 = QLabel()
        label3.resize(50, 50)
        label4 = QLabel()
        label4.resize(50, 50)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label3)
        hbox1.addLayout(vbox4)
        hbox1.addWidget(label4)
        #标题与两个按钮上下协调
        label5 = QLabel()
        label5.resize(1, 1)
        label6 = QLabel()
        label6.resize(1, 1)
        label7 = QLabel()
        label7.resize(1, 1)
        vbox2.addWidget(label5)
        vbox2.addWidget(label)
        vbox2.addWidget(label6)
        vbox2.addLayout(hbox1)
        vbox2.addWidget(label7)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)
        widget.setLayout(hbox)

        self.setCentralWidget(widget)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragPosition = event.globalPos() - self.frameGeometry().topLeft()
            QApplication.postEvent(self, QEvent(174))
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.dragPosition)
            event.accept()

    def quitClicked(self):
        reply = QMessageBox.question(self, 'Warning',
                                           'Are you sure to quit?', QMessageBox.Yes,
                                           QMessageBox.No)
        if reply == QMessageBox.Yes:
            quit()

    def logClicked(self):
        self.hide()
        # Form = QDialog()
        # self.ui = log.MainPage()
        # self.ui.show()
        # self.ui.exec_()
        Form = QDialog()
        ui = test.MainPage(Form)
        Form.show()
        Form.exec_()
        # self.show()
        self.show()



    def recognizeClicked(self):
        self.hide()
        Form = QDialog()
        ui = rec.MainPage(Form)
        Form.show()
        Form.exec_()
        self.show()

    def center(self):
        qr = self.frameGeometry()  # 得到该主窗口的矩形框架qr
        cp = QDesktopWidget().availableGeometry().center()  # 屏幕中间点的坐标cp
        qr.moveCenter(cp)  # 将矩形框架移至屏幕正中央
        self.move(qr.topLeft())  # 应用窗口移至矩形框架的左上角点


            # print('Received message from %s : %s' % (addr, data))
        #     if data == 'quit':
        #         break
        #     order = data.split()[0]
        #     self.recv_func(order, data, conn)
        # conn.close()
        # print('-----------------')



def main():
    app = QApplication(sys.argv)
    main = Home()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
