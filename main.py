from cv2 import (CascadeClassifier as CascadeCls, VideoCapture as VidCap,
                 COLOR_BGR2GRAY as BGR2GRY, COLOR_BGR2RGB as BGR2RGB, 
                 cvtColor, rectangle, data)
from PyQt6.QtWidgets import (QLabel, QComboBox, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QApplication)
from pygrabber.dshow_graph import FilterGraph as graph
from qdarktheme import setup_theme as setTheme
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
import sys

COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 255, 255) # White
]

CLS = f"{data.haarcascades}haarcascade_frontalface_default.xml"
SCALE = [640, 480, Qt.AspectRatioMode.KeepAspectRatio]
RGB888 = QImage.Format.Format_RGB888

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cascade  = CascadeCls(CLS)
        self.detectMS = self.cascade.detectMultiScale

        # Cam Select Dropdown
        self.cam = QComboBox(self)
        self.cam.addItems([c for c in graph().get_input_devices()])
        self.cam.currentIndexChanged.connect(self.setCam)

        # Rectangle Select Dropdown
        self.selColor = QComboBox(self)
        self.selColor.addItems(['Red', 'Green', 'Blue', 'White'])
        self.selColor.currentIndexChanged.connect(self.setColor)

        # Video Widget
        self.label = QLabel(self)

        # Video Update Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//30)

        self.rColor = COLORS[0] # Rectangle color
        self.cap    = VidCap(0) # Init Camera
      
        self.__initUI(self)

    def setCam(self, i):
        if self.cap.isOpened():
            self.cap.release()
        self.cap = VidCap(i)

    def setColor(self, i):
        self.rColor = COLORS[i] 

    def update(self):
        ret, frame = self.cap.read()
        if not ret: return
 
        for (x, y, w, h) in self.detectMS(cvtColor(frame, BGR2GRY), scaleFactor=1.1, minNeighbors=5):
            rectangle(frame, (x, y), (x+w, y+h), self.rColor, 3)

        # Display the resulting frame
        img = cvtColor(frame, BGR2RGB)
        h, w, ch = img.shape
        self.label.setPixmap(QPixmap.fromImage(QImage(img.data, w, h, ch*w, RGB888).scaled(*SCALE)))

    def __initUI(self, MainWin):
        MainWin.resize(640, 480)
        self.mainWidget = QWidget(MainWin)
        
        self.vertLayout = QVBoxLayout(self.mainWidget)
        self.vertLayout.setSpacing(15)

        self.ctrlLayout = QHBoxLayout()
        self.ctrlLayout.addWidget(self.selColor)
        self.ctrlLayout.addWidget(self.cam)

        self.vertLayout.addLayout(self.ctrlLayout)
        self.vertLayout.addWidget(self.label)

        MainWin.setCentralWidget(self.mainWidget)


if __name__ == "__main__":
    APP = QApplication(sys.argv)
    setTheme()
    WINDOW = MainWindow()
    WINDOW.setWindowTitle('Face Detection')
    WINDOW.show()
    sys.exit(APP.exec())
