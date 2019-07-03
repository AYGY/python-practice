# -*- coding: utf-8 -*-
import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
# import client.test2 as test2
import client1.prepare as pre
import os
import cv2
import time
import threading
import socket

# try:
#     _fromUtf8 = QString.fromUtf8
# except AttributeError:
#     def _fromUtf8(s):
#         return s

try:
    _encoding = QApplication.UnicodeUTF8


    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig)


class MainPage(QMainWindow):
    def __init__(self, Dialog, name, id):
        super(MainPage, self).__init__()
        self.initUI(Dialog, name, id)


    def initUI(self, Dialog, name, id):
        Dialog.resize(400, 240)
        self.name = name
        self.id = id
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', 9999))
        r = threading.Thread(target=self.rec, args=(self.s,))
        r.start()
        self.n = 0
        self.form = Dialog
        self.form.setObjectName("window")
        self.form.setStyleSheet("background-image: url(background1.jpg)")
        self.form.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)

        # back = QPushButton("Back", Dialog)
        # # back.setGeometry(190, 210, 50, 25)
        # back.clicked.connect(self.backClicked)

        self._title = "image_loader"  # 设置类实例成员_title,它的值为字符串的"image_laoder"
        self._diawidth = 300  # 设置实例成员_diawidth,它的值为300
        self._diaheight = 600
        self.form.setWindowTitle(self._title)  # 设置窗口标题
        self.form.setMinimumHeight(self._diaheight)  # 设置窗口最小的大小
        self.form.setMinimumWidth(self._diawidth)
        self.imageView = QLabel(name+id)  # 得到一个QLabel的实例，并将它保存在成员imageView里，负责显示消息以及图片
        self.imageView.setAlignment(Qt.AlignCenter)  # 设置QLabel居中显示
        self.btn_open = QPushButton("open")  # 实例化一个名为"open"的按钮，并将它保存在类成员btn_open中，负责去得到图片的路径，并在QLabel中显示
        self.btn_open.clicked.connect(self.on_btn_open_clicked)  # pyqt5中信号与槽的连接
        self.relog = QPushButton("ReLog")
        self.relog.clicked.connect(self.relog_clicked)
        self.ok = QPushButton("Ok")
        self.ok.clicked.connect(self.ok_clicked)
        self.quit = QPushButton("Quit")
        self.quit.clicked.connect(self.quit_clicked)
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.imageView)
        self.vlayout.addWidget(self.btn_open)
        self.vlayout.addWidget(self.relog)
        self.vlayout.addWidget(self.ok)
        self.vlayout.addWidget(self.quit)
        Dialog.setLayout(self.vlayout)

    def on_btn_open_clicked(self):
        self.filename = QFileDialog.getOpenFileName(self, "OpenFile", ".",
                                                    "Image Files(*.jpg *.jpeg *.png)")[0]
        print(self.filename)
        if len(self.filename):
            self.image = QImage(self.filename)
            self.imageView.setPixmap(QPixmap.fromImage(self.image))
            self.imageView.update()
            self.resize(self.image.width(), self.image.height())

    def quit_clicked(self):
        self.s.send('quit'.encode())
        self.s.close()
        self.form.close()

    def ok_clicked(self):

        self.picture(self.filename)


    def relog_clicked(self):
        self.dir = './image/trainfaces'+'/'+self.id
        self.img = pre.getfacefromcamera()

        self.image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        height, width, channel = self.image.shape
        bytesPerLine = 3 * width
        self.qImg = QImage(self.image.data, width, height, QImage.Format_RGB888)
        # self.image = QImage(self.img)
        self.imageView.setPixmap(QPixmap.fromImage(self.qImg))
        self.imageView.update()
        self.resize(width, height)
        pre.createdir(self.dir)
        self.filename = self.dir+'/'+str(self.n) + '.jpg'
        print(self.filename)
        cv2.imwrite(self.filename, self.img)
        self.n = self.n + 1


    def put(self, filename):
        s = self.s
        print(filename)
        name = filename.split('/')[-1]
        print(name)
        msg = 'save ' + self.id + '_' + self.name + ' ' + name.split('/')[-1]
        s.send(msg.encode())
        time.sleep(0.1)
        print('Start uploading image')
        print('Waiting......')

        with open(filename, 'rb') as f:
            while True:
                a = f.read(1024)
                if not a:
                    break
                s.send(a)
            time.sleep(0.1)
            s.send('EOF'.encode())
            # self.rec()
            print('Upload completed')



        time.sleep(1)
        # s.close()

    def picture(self, filename):
        # filename = QFileDialog.getOpenFileName(self, "OpenFile", ".",
        #                                        "Image Files(*.jpg *.jpeg *.png)")[0]
        if filename:
            self.put(filename)

    def rec(self, s):
        while True:
            data = s.recv(1024)
            print('kkkkk')
            # data = data.decode()
            if data == 'Saving completed!'.encode():
                print(data.decode())
                self.imageView.clear()
                self.imageView.setText("保存成功")

                self.imageView.update()
            if data == 'Saving fail!'.encode():
                self.imageView.clear()
                self.imageView.setText("保存失败")

                self.imageView.update()
                # break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Dialog = QDialog()
    ui = MainPage(Dialog)
    ui.show()
    sys.exit(app.exec_())