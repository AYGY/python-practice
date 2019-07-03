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
import model
import predict as pre
import face
import socket
import threading
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

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
    def __init__(self, Dialog):
        super(MainPage, self).__init__()
        self.initUI(Dialog)


    def initUI(self, Dialog):
        Dialog.resize(400, 240)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', 9999))
        r = threading.Thread(target=self.rec, args=(self.s,))
        r.start()
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
        # self.form.set
        self.imageView = QLabel("请选择图片！")  # 得到一个QLabel的实例，并将它保存在成员imageView里，负责显示消息以及图片
        self.imageView.setAlignment(Qt.AlignCenter)  # 设置QLabel居中显示
        # self.imageView.setStyleSheet("border:2px solid red;")
        self.label = QLabel(" ")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border:2px solid black;")
        self.btn_open = QPushButton("open")  # 实例化一个名为"open"的按钮，并将它保存在类成员btn_open中，负责去得到图片的路径，并在QLabel中显示
        self.btn_open.clicked.connect(self.on_btn_open_clicked)  # pyqt5中信号与槽的连接
        # self.relog = QPushButton("ReLog")
        # self.relog.clicked.connect(self.relog_clicked)
        self.ok = QPushButton("Ok")
        self.ok.clicked.connect(self.ok_clicked)
        self.quit = QPushButton("Quit")
        self.quit.clicked.connect(self.quit_clicked)

        label1 = QLabel()
        label1.resize(50, 50)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.label)
        self.label.setSizePolicy(sizePolicy)
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.imageView)
        self.vlayout.addLayout(self.hbox)
        self.vlayout.addWidget(self.btn_open)
        # self.vlayout.addWidget(self.relog)
        self.vlayout.addWidget(self.ok)
        self.vlayout.addWidget(self.quit)
        Dialog.setLayout(self.vlayout)

    def on_btn_open_clicked(self):
        self.filename = QFileDialog.getOpenFileName(self, "OpenFile", ".",
                                                    "Image Files(*.jpg *.jpeg *.png)")[0]
        # print(self.filename)
        if len(self.filename):
            self.image = QImage(self.filename)
            # self.image.scaled(self.imageView.width(), self.imageView.height())
            self.imageView.setPixmap(QPixmap.fromImage(self.image).scaled(self.imageView.width(), self.imageView.height()))
            # self.resize(self.image.width(), self.image.height())

    def quit_clicked(self):
        self.s.send('quit'.encode())
        self.s.close()
        self.form.close()

    def ok_clicked(self):
        # cv2.imwrite(os.path.join(self.dir, str(self.n) + '.jpg'), self.img)
        # self.n = self.n + 1
        # self.imageView = QLabel("保存成功")
        # rec = face.Recognition()
        # print("hhh")
        # img = cv2.imread(self.filename)
        # print(img)
        # start = time.time()
        # face_1 = slim.conv2d(img, 32, [1, 1], weights_initializer=tf.ones_initializer, padding='SAME')
        # self.rec.encoder.sess.run(tf.global_variables_initializer())
        # test = self.rec.encoder.sess.run(face_1)
        # print("ddd")
        # cv2.imshow(test)
        # dur = time.time() - start
        # print(dur)
        self.picture(self.filename)

        # print("lll")
        # # label = pre.main(pre.parse_arguments(sys.argv[1:]))
        # self.form.close()

    def relog_clicked(self):
        self.form.close()


    def put(self, filename):
        s = self.s
        print(filename)
        name = filename.split('/')[-1]
        print(name)
        msg = 'rec ' + filename
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
            if data:
                print(data.decode())
                self.label.clear()
                # print("%s" % face_1[0].name)
                self.label.setText(data.decode())
                self.label.update()
                break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Dialog = QDialog()
    ui = MainPage(Dialog)
    Dialog.show()
    sys.exit(Dialog.exec_())