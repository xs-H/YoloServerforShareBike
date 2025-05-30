# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'yoloside6.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QProgressBar, QPushButton, QSizePolicy,
    QSlider, QTextBrowser, QVBoxLayout, QWidget, QScrollArea) # <<< 导入 QScrollArea

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1369, 914) # 初始窗口大小
        # MainWindow.setMinimumSize(1200, 800) # 可以考虑设置一个最小尺寸

        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setPointSize(12)
        MainWindow.setFont(font)

        MainWindow.setStyleSheet(u"""
            /* Overall Window Style */
            QMainWindow, #centralwidget {
                background-color: #2E3440; /* Darker background (Nord Polar Night) */
                color: #D8DEE9; /* Light text (Nord Snow Storm) */
            }

            /* Main Vertical Frame - can act as a slightly lighter container */
            #verticalFrame {
                background-color: #3B4252; /* Slightly lighter dark (Nord Polar Night) */
                border: 1px solid #4C566A; /* Subtle border (Nord Polar Night) */
                border-radius: 5px; /* Rounded corners for the main content area */
            }

            /* Title Label */
            #title {
                font-size: 22pt;
                font-weight: bold;
                color: #ECEFF4; /* Brightest text (Nord Snow Storm) */
                background-color: #5E81AC; /* Accent blue (Nord Frost) */
                padding: 10px;
                border: none; /* Remove individual border */
                border-top-left-radius: 5px; /* Match parent frame */
                border-top-right-radius: 5px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                qproperty-alignment: 'AlignCenter';
            }

            /* Section Header Labels (like "模型选择", "检测对象选择") */
            QLabel[objectName^="label_"] { /* Targets labels like label_2, label_5 etc. */
                font-size: 14pt;
                font-weight: bold;
                color: #E5E9F0; /* Lighter text (Nord Snow Storm) */
                background-color: #434C5E; /* Darker accent (Nord Polar Night) */
                padding: 8px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                border: 1px solid #4C566A;
                border-bottom: 2px solid #5E81AC; /* Accent line at bottom */
                qproperty-alignment: 'AlignCenter';
            }
            /* Specific label for "检测结果" which is a sub-header */
            #label_3 {
                font-size: 14pt;
                font-weight: bold;
                color: #E5E9F0;
                background-color: #434C5E;
                padding: 8px;
                border-radius: 4px; /* Can be fully rounded */
                border: 1px solid #4C566A;
                 qproperty-alignment: 'AlignCenter';
            }


            /* Frames containing controls under section headers */
            QFrame[objectName^="horizontalFrame_"],
            QFrame[objectName^="verticalFrame_"] { /* Targets frames like horizontalFrame_2, verticalFrame_4 */
                background-color: #3B4252; /* Consistent with verticalFrame background */
                border: 1px solid #4C566A; /* Subtle border for grouping */
                border-top: none; /* Avoid double border with header label */
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
                padding: 5px;
            }
            /* Remove border from the main mid and right panel frames if desired, or style them */
            #verticalFrame_mid, #verticalFrame_10 {
                 border: 1px solid #4C566A;
                 background-color: #3B4252;
                 border-radius: 4px;
            }
             /* Remove border from specific frames that are just for layout */
            #horizontalFrame_2, #horizontalFrame_5, #horizontalFrame_6,
            #horizontalFrame_51, #horizontalFrame_7, #horizontalFrame_12,
            #verticalFrame_12, #verticalFrame_16 {
                border: none;
                background-color: transparent; /* Make them blend */
                padding: 0px;
            }


            /* Push Buttons */
            QPushButton {
                background-color: #5E81AC; /* Accent blue (Nord Frost) */
                color: #ECEFF4; /* Brightest text */
                border: 1px solid #4C566A;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #81A1C1; /* Lighter blue on hover (Nord Frost) */
            }
            QPushButton:pressed {
                background-color: #88C0D0; /* Even lighter blue when pressed (Nord Frost) */
            }
            QPushButton:disabled {
                background-color: #4C566A;
                color: #6c757d;
            }
            /* Special button: Start Detection */
            #yolo_start {
                background-color: #A3BE8C; /* Green accent (Nord Aurora) */
                font-size: 14pt;
            }
            #yolo_start:hover {
                background-color: #B4D0A0;
            }
            #yolo_start:pressed {
                background-color: #8FBCBB; /* Teal for pressed (Nord Frost) */
            }

            /* Line Edit */
            QLineEdit {
                background-color: #4C566A; /* Dark input field (Nord Polar Night) */
                color: #D8DEE9;
                border: 1px solid #5E81AC; /* Accent border */
                border-radius: 4px;
                padding: 6px;
                font-size: 12pt;
            }
            QLineEdit:focus {
                border: 1px solid #88C0D0; /* Lighter accent on focus */
            }

            /* Double Spin Box */
            QDoubleSpinBox {
                background-color: #4C566A;
                color: #D8DEE9;
                border: 1px solid #5E81AC;
                border-radius: 4px;
                padding: 4px;
                font-size: 12pt;
                qproperty-alignment: 'AlignCenter';
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                background-color: #5E81AC;
                border-radius: 2px;
                width: 18px;
                height: 10px;
            }
            QDoubleSpinBox::up-button { subcontrol-position: top right; margin: 2px; }
            QDoubleSpinBox::down-button { subcontrol-position: bottom right; margin: 2px; }
            QDoubleSpinBox::up-arrow { image: url(icons/arrow_up.png); width: 10px; height:10px; }
            QDoubleSpinBox::down-arrow { image: url(icons/arrow_down.png); width: 10px; height:10px; }


            /* Slider */
            QSlider::groove:horizontal {
                border: 1px solid #4C566A;
                height: 8px;
                background: #434C5E;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #88C0D0; /* Teal handle (Nord Frost) */
                border: 1px solid #5E81AC;
                width: 16px;
                height: 16px;
                margin: -4px 0; /* handle is centered on the groove */
                border-radius: 8px;
            }

            /* CheckBox */
            QCheckBox {
                font-size: 12pt;
                color: #D8DEE9;
                spacing: 5px; /* Space between checkbox and text */
                border: none; /* Remove border from checkbox text label part */
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #5E81AC;
                border-radius: 3px;
                background-color: #4C566A;
            }
            QCheckBox::indicator:checked {
                background-color: #A3BE8C; /* Green check (Nord Aurora) */
                image: url(icons/check.png); /* Create a small white checkmark icon */
            }
            QCheckBox::indicator:unchecked:hover {
                border: 1px solid #81A1C1;
            }
            #save_data { /* Specific styling for the "是否存储数据" checkbox label */
                qproperty-alignment: 'AlignVCenter | AlignRight';
                padding-right: 10px;
            }
            #label_9 { /* The text "是否存储数据" */
                 background-color: transparent; border: none; padding: 0;
                 font-weight: normal;
                 qproperty-alignment: 'AlignVCenter | AlignLeft';
            }


            /* Progress Bar */
            QProgressBar {
                border: 1px solid #4C566A;
                border-radius: 5px;
                text-align: center;
                color: #ECEFF4;
                background-color: #434C5E;
                height: 10px; /* Make it a bit thicker */
                font-size: 9pt; /* Text on progress bar can be smaller */
            }
            QProgressBar::chunk {
                background-color: #A3BE8C; /* Green progress (Nord Aurora) */
                border-radius: 4px;
                margin: 0.5px; /* Creates a small gap around the chunk */
            }

            /* Text Browser (Detection Result) */
            QTextBrowser {
                background-color: #292E39; /* Slightly different dark for text areas */
                color: #D8DEE9;
                border: 1px solid #4C566A;
                border-radius: 4px;
                font-size: 11pt; /* Smaller font for logs/results */
            }

            /* Image Display Labels - These are now inside QScrollArea */
            /* Style the QScrollArea instead if needed, or the QLabel directly if specific styling is required beyond placeholder */
             #finall_result_label, #upload_image_label { /* Renamed for clarity if you use objectName on the QLabel */
                background-color: #252932; /* Dark background for image areas */
                /* border: 1px dashed #4C566A; */ /* Border can be on ScrollArea or removed */
                color: #8F96A3; /* Placeholder text color */
                qproperty-alignment: 'AlignCenter';
            }

            /* QScrollArea for image displays */
            QScrollArea {
                background-color: #252932; /* Match the QLabel background */
                border: 1px dashed #4C566A; /* Dashed border to indicate placeholder */
                border-radius: 4px;
            }
            /* Remove border from specific scroll areas if their parent frame already has one */
             #scrollArea_final_result, #scrollArea_upload_image {
                  border: 1px dashed #4C566A; /* Keep placeholder border */
                  background-color: #252932; /* Ensure background color */
             }


            #logo {
                border: none; /* No border for the logo itself */
                background-color: transparent;
            }

            /* Labels for detection quantity and time */
            #label_8, #label_11 { /* "检测数量", "检测耗时" */
                background-color: #434C5E;
                color: #E5E9F0;
                padding: 6px;
                border: 1px solid #4C566A;
                border-right: none; /* No right border if next to value */
                border-top-left-radius: 4px;
                border-bottom-left-radius: 4px;
                border-top-right-radius: 0px;
                border-bottom-right-radius: 0px;
                font-weight: bold;
                font-size: 11pt;
                 qproperty-alignment: 'AlignCenter';
            }
            #detection_quantity, #detection_time { /* The actual values */
                background-color: #4C566A;
                color: #ECEFF4;
                padding: 6px;
                border: 1px solid #4C566A;
                border-left: none;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
                font-weight: bold;
                font-size: 11pt;
                qproperty-alignment: 'AlignCenter';
            }

        """)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalFrame = QFrame(self.centralwidget)
        self.verticalFrame.setObjectName(u"verticalFrame")
        self.verticalLayout = QVBoxLayout(self.verticalFrame)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0,0,0,9)
        self.verticalLayout.setSpacing(10)

        self.page_up = QHBoxLayout()
        self.page_up.setObjectName(u"page_up")
        self.title = QLabel(self.verticalFrame)
        self.title.setObjectName(u"title")
        self.page_up.addWidget(self.title)
        self.verticalLayout.addLayout(self.page_up)

        self.page_down = QHBoxLayout()
        self.page_down.setSpacing(10)
        self.page_down.setObjectName(u"page_down")
        self.page_down.setContentsMargins(9,0,9,0)

        self.left = QHBoxLayout()
        self.left.setSpacing(2)
        self.left.setObjectName(u"left")
        self.left_page = QVBoxLayout()
        self.left_page.setSpacing(10)
        self.left_page.setObjectName(u"left_page")
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalFrame_4 = QFrame(self.verticalFrame)
        self.verticalFrame_4.setObjectName(u"verticalFrame_4")
        self.verticalLayout_5 = QVBoxLayout(self.verticalFrame_4)
        self.verticalLayout_5.setSpacing(8)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 8)
        self.label_2 = QLabel(self.verticalFrame_4)
        self.label_2.setObjectName(u"label_2")
        self.verticalLayout_5.addWidget(self.label_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(8, 8, 8, 8)
        self.horizontalFrame_2 = QFrame(self.verticalFrame_4)
        self.horizontalFrame_2.setObjectName(u"horizontalFrame_2")
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalFrame_2)
        self.horizontalLayout_4.setSpacing(4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0,0,0,0)
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(10)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(-1, 0, 0, 0)
        self.model_name = QLineEdit(self.horizontalFrame_2)
        self.model_name.setObjectName(u"model_name")
        self.model_name.setMinimumSize(QSize(0, 38))
        self.horizontalLayout_7.addWidget(self.model_name)

        self.model_select = QPushButton(self.horizontalFrame_2)
        self.model_select.setObjectName(u"model_select")
        icon = QIcon()
        icon.addFile(u"icons/\u9009\u62e9\u6587\u4ef6.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.model_select.setIcon(icon)
        self.model_select.setIconSize(QSize(24, 24))
        self.horizontalLayout_7.addWidget(self.model_select)
        self.horizontalLayout_7.setStretch(0,3)
        self.horizontalLayout_7.setStretch(1,1)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_3.addWidget(self.horizontalFrame_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.verticalLayout_11.addWidget(self.verticalFrame_4)
        self.left_page.addLayout(self.verticalLayout_11)

        self.select = QVBoxLayout()
        self.select.setObjectName(u"select")
        self.verticalFrame_model = QFrame(self.verticalFrame)
        self.verticalFrame_model.setObjectName(u"verticalFrame_model")
        self.verticalLayout_8 = QVBoxLayout(self.verticalFrame_model)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setSpacing(8)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 8)
        self.label_5 = QLabel(self.verticalFrame_model)
        self.label_5.setObjectName(u"label_5")
        self.verticalLayout_8.addWidget(self.label_5)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(8,8,8,8)
        self.horizontalFrame_5 = QFrame(self.verticalFrame_model)
        self.horizontalFrame_5.setObjectName(u"horizontalFrame_5")
        self.horizontalLayout_9 = QHBoxLayout(self.horizontalFrame_5)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0,0,0,0)
        self.horizontalLayout_9.setSpacing(8)
        font1 = QFont()
        font1.setPointSize(12)
        font1.setWeight(QFont.Bold)

        self.image = QPushButton(self.horizontalFrame_5)
        self.image.setObjectName(u"image")
        self.image.setFont(font1)
        icon1 = QIcon()
        icon1.addFile(u"icons/\u7167\u7247_pic.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.image.setIcon(icon1)
        self.image.setIconSize(QSize(24, 24))
        self.horizontalLayout_9.addWidget(self.image)

        self.video = QPushButton(self.horizontalFrame_5)
        self.video.setObjectName(u"video")
        self.video.setFont(font1)
        icon2 = QIcon()
        icon2.addFile(u"icons/\u89c6\u9891_video-two.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video.setIcon(icon2)
        self.video.setIconSize(QSize(24, 24))
        self.horizontalLayout_9.addWidget(self.video)

        self.camera = QPushButton(self.horizontalFrame_5)
        self.camera.setObjectName(u"camera")
        self.camera.setFont(font1)
        icon3 = QIcon()
        icon3.addFile(u"icons/\u6444\u50cf\u5934_camera-one.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.camera.setIcon(icon3)
        self.camera.setIconSize(QSize(24, 24))
        self.horizontalLayout_9.addWidget(self.camera)

        self.dirs = QPushButton(self.horizontalFrame_5)
        self.dirs.setObjectName(u"dirs")
        self.dirs.setFont(font1)
        icon4 = QIcon()
        icon4.addFile(u"icons/\u56fe\u5c42_layers.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.dirs.setIcon(icon4)
        self.dirs.setIconSize(QSize(24, 24))
        self.horizontalLayout_9.addWidget(self.dirs)
        self.horizontalLayout_8.addWidget(self.horizontalFrame_5)
        self.verticalLayout_8.addLayout(self.horizontalLayout_8)
        self.select.addWidget(self.verticalFrame_model)
        self.left_page.addLayout(self.select)

        self.conf = QVBoxLayout()
        self.conf.setObjectName(u"conf")
        self.horizontalFrame_6 = QFrame(self.verticalFrame)
        self.horizontalFrame_6.setObjectName(u"horizontalFrame_6")
        self.horizontalLayout_11 = QHBoxLayout(self.horizontalFrame_6)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(0,0,0,0)
        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setSpacing(8)
        self.verticalLayout_10.setContentsMargins(0,0,0,8)
        self.label_6 = QLabel(self.horizontalFrame_6)
        self.label_6.setObjectName(u"label_6")
        self.verticalLayout_10.addWidget(self.label_6)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(8,8,8,8)
        self.horizontalLayout_15.setSpacing(8)
        self.conf_num = QDoubleSpinBox(self.horizontalFrame_6)
        self.conf_num.setObjectName(u"conf_num")
        self.conf_num.setMinimumSize(QSize(70,0))
        self.horizontalLayout_15.addWidget(self.conf_num)
        self.conf_slider = QSlider(self.horizontalFrame_6)
        self.conf_slider.setObjectName(u"conf_slider")
        self.conf_slider.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalLayout_15.addWidget(self.conf_slider)
        self.horizontalLayout_15.setStretch(0,1)
        self.horizontalLayout_15.setStretch(1,3)
        self.verticalLayout_10.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_11.addLayout(self.verticalLayout_10)
        self.conf.addWidget(self.horizontalFrame_6)
        self.left_page.addLayout(self.conf)

        self.IOU = QVBoxLayout()
        self.IOU.setObjectName(u"IOU")
        self.horizontalFrame_51 = QFrame(self.verticalFrame)
        self.horizontalFrame_51.setObjectName(u"horizontalFrame_51")
        self.horizontalLayout_10 = QHBoxLayout(self.horizontalFrame_51)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0,0,0,0)
        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setSpacing(8)
        self.verticalLayout_9.setContentsMargins(0,0,0,8)
        self.label_7 = QLabel(self.horizontalFrame_51)
        self.label_7.setObjectName(u"label_7")
        self.verticalLayout_9.addWidget(self.label_7)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(8,8,8,8)
        self.horizontalLayout_12.setSpacing(8)
        self.iou_number = QDoubleSpinBox(self.horizontalFrame_51)
        self.iou_number.setObjectName(u"iou_number")
        self.iou_number.setMinimumSize(QSize(70,0))
        self.horizontalLayout_12.addWidget(self.iou_number)
        self.iou_slider = QSlider(self.horizontalFrame_51)
        self.iou_slider.setObjectName(u"iou_slider")
        self.iou_slider.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalLayout_12.addWidget(self.iou_slider)
        self.horizontalLayout_12.setStretch(0,1)
        self.horizontalLayout_12.setStretch(1,3)
        self.verticalLayout_9.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_10.addLayout(self.verticalLayout_9)
        self.IOU.addWidget(self.horizontalFrame_51)
        self.left_page.addLayout(self.IOU)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalFrame_7 = QFrame(self.verticalFrame)
        self.horizontalFrame_7.setObjectName(u"horizontalFrame_7")
        self.horizontalLayout_16 = QHBoxLayout(self.horizontalFrame_7)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(8,8,8,8)
        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setContentsMargins(0,0,0,0)
        self.horizontalLayout_17.setSpacing(10)
        self.label_9 = QLabel(self.horizontalFrame_7)
        self.label_9.setObjectName(u"label_9")
        self.horizontalLayout_17.addWidget(self.label_9)
        self.save_data = QCheckBox(self.horizontalFrame_7)
        self.save_data.setObjectName(u"save_data")
        self.horizontalLayout_17.addWidget(self.save_data)
        self.horizontalLayout_17.setStretch(0, 1)
        self.horizontalLayout_17.setStretch(1, 0)
        self.horizontalLayout_16.addLayout(self.horizontalLayout_17)
        self.verticalLayout_6.addWidget(self.horizontalFrame_7)
        self.left_page.addLayout(self.verticalLayout_6)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.logo = QLabel(self.verticalFrame)
        self.logo.setObjectName(u"logo")
        self.logo.setMinimumSize(QSize(0,100))
        self.logo.setPixmap(QPixmap(u"icons/3321.png"))
        self.logo.setScaledContents(True)
        self.logo.setAlignment(Qt.AlignCenter)
        self.verticalLayout_7.addWidget(self.logo)
        self.left_page.addLayout(self.verticalLayout_7)

        self.left_page.setStretch(0, 0)
        self.left_page.setStretch(1, 0)
        self.left_page.setStretch(2, 0)
        self.left_page.setStretch(3, 0)
        self.left_page.setStretch(4, 0)
        self.left_page.setStretch(5, 1)
        self.left.addLayout(self.left_page)
        self.page_down.addLayout(self.left)

        self.mid = QHBoxLayout()
        self.mid.setObjectName(u"mid")
        self.verticalFrame_mid = QFrame(self.verticalFrame)
        self.verticalFrame_mid.setObjectName(u"verticalFrame_mid")
        self.verticalLayout_19 = QVBoxLayout(self.verticalFrame_mid)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.verticalLayout_19.setSpacing(10)
        self.verticalLayout_19.setContentsMargins(9,9,9,9)

        # --- QScrollArea for finall_result ---
        self.scrollArea_final_result = QScrollArea(self.verticalFrame_mid)
        self.scrollArea_final_result.setObjectName(u"scrollArea_final_result")
        self.scrollArea_final_result.setWidgetResizable(True)
        # self.scrollArea_final_result.setStyleSheet("QScrollArea { border: none; }") # Style in main QSS

        self.finall_result = QLabel() # Create QLabel without parent
        self.finall_result.setObjectName(u"finall_result_label") # Changed objectName for clarity if needed
        self.finall_result.setScaledContents(False) # IMPORTANT
        self.finall_result.setAlignment(Qt.AlignCenter) # Center image in label
        # Placeholder text can be set here or when no image is loaded
        self.finall_result.setText(QCoreApplication.translate("MainWindow", u"检测结果展示区域", None))


        self.scrollArea_final_result.setWidget(self.finall_result)
        # --- End QScrollArea for finall_result ---

        self.verticalLayout_23 = QVBoxLayout()
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.verticalLayout_23.addWidget(self.scrollArea_final_result) # Add ScrollArea to layout
        self.verticalLayout_19.addLayout(self.verticalLayout_23)


        self.verticalLayout_22 = QVBoxLayout()
        self.verticalLayout_22.setSpacing(0)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.verticalFrame_16 = QFrame(self.verticalFrame_mid)
        self.verticalFrame_16.setObjectName(u"verticalFrame_16")
        self.verticalLayout_24 = QVBoxLayout(self.verticalFrame_16)
        self.verticalLayout_24.setSpacing(2)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(0,0,0,0)
        self.yolo_start = QPushButton(self.verticalFrame_16)
        self.yolo_start.setObjectName(u"yolo_start")
        self.yolo_start.setFont(font1)
        self.verticalLayout_24.addWidget(self.yolo_start)
        self.verticalLayout_22.addWidget(self.verticalFrame_16)
        self.verticalLayout_19.addLayout(self.verticalLayout_22)

        self.verticalLayout_21 = QVBoxLayout()
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.horizontalFrame_12 = QFrame(self.verticalFrame_mid)
        self.horizontalFrame_12.setObjectName(u"horizontalFrame_12")
        self.horizontalLayout_21 = QHBoxLayout(self.horizontalFrame_12)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalLayout_21.setSpacing(8)
        self.horizontalLayout_21.setContentsMargins(0,0,0,0)

        self.video_start = QPushButton(self.horizontalFrame_12)
        self.video_start.setObjectName(u"video_start")
        icon5 = QIcon()
        icon5.addFile(u"icons/\u64ad\u653e.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video_start.setIcon(icon5)
        self.video_start.setIconSize(QSize(24, 24))
        self.horizontalLayout_21.addWidget(self.video_start)

        self.video_stop = QPushButton(self.horizontalFrame_12)
        self.video_stop.setObjectName(u"video_stop")
        icon6 = QIcon()
        icon6.addFile(u"icons/\u6682\u505c.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video_stop.setIcon(icon6)
        self.video_stop.setIconSize(QSize(24, 24))
        self.horizontalLayout_21.addWidget(self.video_stop)

        self.video_progressBar = QProgressBar(self.horizontalFrame_12)
        self.video_progressBar.setObjectName(u"video_progressBar")
        self.video_progressBar.setValue(0)
        self.horizontalLayout_21.addWidget(self.video_progressBar)

        self.video_termination = QPushButton(self.horizontalFrame_12)
        self.video_termination.setObjectName(u"video_termination")
        icon7 = QIcon()
        icon7.addFile(u"icons/\u7ed3\u675f.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video_termination.setIcon(icon7)
        self.video_termination.setIconSize(QSize(24, 24))
        self.horizontalLayout_21.addWidget(self.video_termination)
        self.horizontalLayout_21.setStretch(0,0)
        self.horizontalLayout_21.setStretch(1,0)
        self.horizontalLayout_21.setStretch(2,1)
        self.horizontalLayout_21.setStretch(3,0)
        self.verticalLayout_21.addWidget(self.horizontalFrame_12)
        self.verticalLayout_19.addLayout(self.verticalLayout_21)

        self.verticalLayout_19.setStretch(0, 10) # Image display (ScrollArea)
        self.verticalLayout_19.setStretch(1, 0)
        self.verticalLayout_19.setStretch(2, 0)
        self.mid.addWidget(self.verticalFrame_mid)
        self.page_down.addLayout(self.mid)

        self.right = QHBoxLayout()
        self.right.setSpacing(2)
        self.right.setObjectName(u"right")
        self.verticalFrame_10 = QFrame(self.verticalFrame)
        self.verticalFrame_10.setObjectName(u"verticalFrame_10")
        self.verticalLayout_14 = QVBoxLayout(self.verticalFrame_10)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setSpacing(10)
        self.verticalLayout_14.setContentsMargins(9,9,9,9)
        self.verticalLayout_15 = QVBoxLayout()
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setSpacing(10)

        # --- QScrollArea for upload_image ---
        self.scrollArea_upload_image = QScrollArea(self.verticalFrame_10)
        self.scrollArea_upload_image.setObjectName(u"scrollArea_upload_image")
        self.scrollArea_upload_image.setWidgetResizable(True)
        # self.scrollArea_upload_image.setStyleSheet("QScrollArea { border: none; }") # Style in main QSS

        self.upload_image = QLabel() # Create QLabel without parent
        self.upload_image.setObjectName(u"upload_image_label") # Changed objectName for clarity
        self.upload_image.setScaledContents(False) # IMPORTANT
        self.upload_image.setAlignment(Qt.AlignCenter)
        self.upload_image.setText(QCoreApplication.translate("MainWindow", u"上传的图像或视频帧", None))


        self.scrollArea_upload_image.setWidget(self.upload_image)
        # --- End QScrollArea for upload_image ---

        self.verticalLayout_20 = QVBoxLayout()
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.addWidget(self.scrollArea_upload_image) # Add ScrollArea to layout
        self.verticalLayout_15.addLayout(self.verticalLayout_20)


        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.verticalFrame_12 = QFrame(self.verticalFrame_10)
        self.verticalFrame_12.setObjectName(u"verticalFrame_12")
        self.verticalLayout_18 = QVBoxLayout(self.verticalFrame_12)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.verticalLayout_18.setSpacing(8)
        self.verticalLayout_18.setContentsMargins(0,0,0,0)
        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_3 = QLabel(self.verticalFrame_12)
        self.label_3.setObjectName(u"label_3")
        self.horizontalLayout_19.addWidget(self.label_3)
        self.verticalLayout_18.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setSpacing(0)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.label_8 = QLabel(self.verticalFrame_12)
        self.label_8.setObjectName(u"label_8")
        self.horizontalLayout_24.addWidget(self.label_8)
        self.detection_quantity = QLabel(self.verticalFrame_12)
        self.detection_quantity.setObjectName(u"detection_quantity")
        self.horizontalLayout_24.addWidget(self.detection_quantity)
        self.horizontalLayout_24.setStretch(0, 1)
        self.horizontalLayout_24.setStretch(1, 1)
        self.verticalLayout_18.addLayout(self.horizontalLayout_24)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setSpacing(0)
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.label_11 = QLabel(self.verticalFrame_12)
        self.label_11.setObjectName(u"label_11")
        self.horizontalLayout_23.addWidget(self.label_11)
        self.detection_time = QLabel(self.verticalFrame_12)
        self.detection_time.setObjectName(u"detection_time")
        self.horizontalLayout_23.addWidget(self.detection_time)
        self.horizontalLayout_23.setStretch(0, 1)
        self.horizontalLayout_23.setStretch(1, 1)
        self.verticalLayout_18.addLayout(self.horizontalLayout_23)
        self.horizontalLayout_20.addWidget(self.verticalFrame_12)
        self.verticalLayout_15.addLayout(self.horizontalLayout_20)

        self.verticalLayout_16 = QVBoxLayout()
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.detection_result = QTextBrowser(self.verticalFrame_10)
        self.detection_result.setObjectName(u"detection_result")
        sizePolicy_textBrowser = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.detection_result.setSizePolicy(sizePolicy_textBrowser) # Ensure size policy
        self.verticalLayout_16.addWidget(self.detection_result)
        self.verticalLayout_15.addLayout(self.verticalLayout_16)

        self.verticalLayout_15.setStretch(0, 3) # Upload image area (ScrollArea)
        self.verticalLayout_15.setStretch(1, 0)
        self.verticalLayout_15.setStretch(2, 2) # Detection result (QTextBrowser)
        self.verticalLayout_14.addLayout(self.verticalLayout_15)
        self.right.addWidget(self.verticalFrame_10)
        self.page_down.addLayout(self.right)

        self.page_down.setStretch(0, 2)
        self.page_down.setStretch(1, 5)
        self.page_down.setStretch(2, 3)
        self.verticalLayout.addLayout(self.page_down)

        self.verticalLayout.setStretch(0, 0)
        self.verticalLayout.setStretch(1, 1)
        self.gridLayout.addWidget(self.verticalFrame, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_2.setContentsMargins(9,9,9,9)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"共享单车智能检测系统", None))
        self.title.setText(QCoreApplication.translate("MainWindow", u"基于YOLO & PySide6的共享单车检测系统", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"模型选择", None))
        self.model_select.setText(QCoreApplication.translate("MainWindow", u"选择模型", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"检测对象选择", None))
        self.image.setText(QCoreApplication.translate("MainWindow", u"图片", None))
        self.video.setText(QCoreApplication.translate("MainWindow", u"视频", None))
        self.camera.setText(QCoreApplication.translate("MainWindow", u"摄像头", None))
        self.dirs.setText(QCoreApplication.translate("MainWindow", u"批量图片", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"设置置信度 (Confidence)", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"设置IOU阈值 (IOU Threshold)", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"是否存储检测结果", None))
        self.save_data.setText("")
        self.logo.setText("")
        # self.finall_result is now a child of QScrollArea, its text is set during creation
        # self.upload_image is now a child of QScrollArea, its text is set during creation
        self.yolo_start.setText(QCoreApplication.translate("MainWindow", u"开始检测", None))
        self.video_start.setText(QCoreApplication.translate("MainWindow", u"播放", None))
        self.video_stop.setText(QCoreApplication.translate("MainWindow", u"暂停", None))
        self.video_termination.setText(QCoreApplication.translate("MainWindow", u"停止", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"检测统计", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"检测数量:", None))
        self.detection_quantity.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"检测耗时:", None))
        self.detection_time.setText(QCoreApplication.translate("MainWindow", u"0.00 s", None))
        self.detection_result.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:11pt; font-weight:normal; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">检测日志和详细结果将显示在此处...</p></body></html>", None))
    # retranslateUi

