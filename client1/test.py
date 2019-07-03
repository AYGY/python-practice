# -*- coding: utf-8 -*-
import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import client1.test2 as test2

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
        self.form = Dialog
        self.form.setObjectName("window")
        self.form.setStyleSheet("background-image: url(background1.jpg)")
        self.form.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)

        # back = QPushButton("Back", Dialog)
        # # back.setGeometry(190, 210, 50, 25)
        # back.clicked.connect(self.backClicked)

        label_name = QLabel(Dialog)
        label_name.setText("<font size=%s><B>%s</B></font>" % ("7", "Name:"))
        label_id = QLabel(Dialog)
        label_id.setText("<font size=%s><B>%s</B></font>" % ("7", "Id:"))
        self.edit_name = QLineEdit(Dialog)
        self.edit_id = QLineEdit(Dialog)
        next = QPushButton("Next", Dialog)
        # widget.setStatusTip('  ')
        # start.resize(50, 25)
        quit = QPushButton("Quit", Dialog)
        # quit.resize(50,25)
        next.clicked.connect(self.nextClicked)
        quit.clicked.connect(self.quitClicked)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()

        label1 = QLabel()
        label1.resize(50, 50)
        vbox1.addWidget(label_name)
        vbox2.addWidget(self.edit_name)
        vbox1.addWidget(label_id)
        vbox2.addWidget(self.edit_id)
        hbox3.addWidget(next)
        hbox3.addWidget(quit)
        hbox2.addLayout(vbox1)
        hbox2.addLayout(vbox2)
        vbox = QVBoxLayout()

        # vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)

        # widget = Q()
        # vbox = QVBoxLayout()
        # vbox.addStretch()
        # vbox.addWidget(back)
        # vbox.addStretch()

        Dialog.setLayout(vbox)
        # widget.setLayout(vbox)
        # self.setCentralWidget(vbox)

    def nextClicked(self):
        name = self.edit_name.text()
        id = self.edit_id.text()
        if name and id:
            self.form.hide()
            Form = QDialog()
            ui = test2.MainPage(Form, name, id)
            Form.show()
            Form.exec_()
            self.form.close()
        else:
            QMessageBox.critical(self, "警告",
                                 self.tr("请输入Name和ID!"))
            # self.label.setText("警告")
            self.form.close()


    def quitClicked(self):
        self.form.close()

    def backClicked(self):
        self.form.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Dialog = QDialog()
    ui = MainPage(Dialog)
    ui.show()
    sys.exit(app.exec_())