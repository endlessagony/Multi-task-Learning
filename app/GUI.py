from PyQt5 import QtCore, QtWidgets
from utils.utils import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

BEAUTIFY_DIR = 'app\gui\src'

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cv_img = None

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, self.cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(self.cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.change_pixmap_signal.emit(self.cv_img)
        img_name = os.path.join(PHOTOS_PATH, 'snap.jpg')
        cv2.imwrite(img_name, self.cv_img)
        return img_name

class MainWindow(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_width = 1280
        self.window_height = 720

        self.setup_ui()

        self.setStyleSheet("""
            QLabel{
                position: absolute;
                color: #fff;
                text-decoration: none;
                border-radius: 20px;
                font-weight: 500;
                margin-left: 40px;
            }

            QPushButton {
                    width: 250px;
                    height: 50px;
                    background: transparent;
                    border: 2px solid #fff;
                    outline: none;
                    border-radius: 6px;
                    color: #fff;
                    font-weight: 500;
                    margin-left: 40px;
            }

            QPushButton::hover {
                    background: #fff;
                    color: #162938;
            }

            QGroupBox {
                    background: transparent;
                    border: 2px solid rgba(255, 255, 255, .5);
                    border-radius: 20px;
            }
        """)

        self.model = MTLModel()
        self.camera = cv2.VideoCapture(0)
        self.action = ''

        self.image_label = QLabel(self)
        self.image_label.resize(720, 720)

        self.results_label = QLabel()
        self.va_label = QLabel()
        self.ar_label = QLabel()
        self.ex_label = QLabel()
        self.au_label = QLabel()


        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
    
    def setup_ui(self):
        self.resize(self.window_width, self.window_height)
        self.setWindowTitle("emorec")

        oImage = QImage(BEAUTIFY_DIR+'\\background.jpg')
        sImage = oImage.scaled(QSize(1280, 720))                   # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))                        
        self.setPalette(palette)

        id = QFontDatabase.addApplicationFont(BEAUTIFY_DIR+"\\Poppins-SemiBold.ttf")
        if id < 0: print("Error")
        self.families = QFontDatabase.applicationFontFamilies(id)

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        self.logo_label = QLabel('emorec', self)
        self.logo_label.setFont(QFont(self.families[0], 30))
        self.logo_label.move(self.window_width-1250, 10) 

        self.quit_btn = QPushButton('Quit', self)
        self.quit_btn.setFont(QFont(self.families[0], 12))
        self.quit_btn.move(self.window_width-350, self.window_height-80)
        self.quit_btn.clicked.connect(self.close)

        self.start_btn = QPushButton('Start webcam', self)
        self.start_btn.setFont(QFont(self.families[0], 12))
        self.start_btn.move(self.window_width-350, self.window_height-180)
        self.start_btn.clicked.connect(self.start_webcam)

        self.photo_btn = QPushButton('Take photo', self)
        self.photo_btn.setFont(QFont(self.families[0], 12))
        self.photo_btn.move(self.window_width-350, self.window_height-180)
        self.photo_btn.clicked.connect(self.take_photo)
        self.photo_btn.hide()

        self.try_again_btn = QPushButton('Try again', self)
        self.try_again_btn.setFont(QFont(self.families[0], 12))
        self.try_again_btn.move(self.window_width-350, self.window_height-380)
        self.try_again_btn.clicked.connect(self.try_again)
        self.try_again_btn.hide()

        self.gallery_btn = QPushButton('Choose from gallery', self)
        self.gallery_btn.setFont(QFont(self.families[0], 12))
        self.gallery_btn.move(self.window_width-350, self.window_height-280)
        self.gallery_btn.clicked.connect(self.choose_from_gallery)

    def clear_outputs(self):
        self.results_label.clear()
        self.va_label.clear()
        self.ex_label.clear()
        self.ar_label.clear()
        self.au_label.clear()

    def try_again(self):
        self.clear_outputs()

        if self.action != 'gallery':
            self.image_label.clearMask()
            self.thread._run_flag = True
            self.start_webcam()
        else:
            pass


    def choose_from_gallery(self):
        self.clear_outputs()
        self.action = 'gallery'
        image_path = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        image = cv2.imread(image_path)
        qt_image = self.convert_cv_qt(image)
        self.image_label.setPixmap(qt_image)

        self.get_results(image_path=image_path)
		

    def start_webcam(self):
        self.start_btn.hide()
        self.photo_btn.show()
        self.thread.start()
    
    def take_photo(self):
        self.clear_outputs()
        self.action = 'take photo'
        image_path = self.thread.stop()
        self.get_results(image_path=image_path)

    def get_results(self, image_path):
        va_output, ar_output, top_expression_class, top_expression_prob, \
            top_action_unit_class, top_action_unit_prob = get_outputs_via_image(image_path, ensemble=self.model)
        
        print(top_action_unit_class, top_action_unit_prob)
        self.results_label = QLabel('Results:', self)
        self.results_label.setFont(QFont(self.families[0], 20))
        self.results_label.move(self.window_width-600, 100)
        self.results_label.show()

        self.va_label = QLabel(f'Valence: {va_output:.3f}', self)
        self.va_label.setFont(QFont(self.families[0], 20))
        self.va_label.move(self.window_width-600, 140)
        self.va_label.show()

        self.ar_label = QLabel(f'Arousal: {ar_output:.3f}', self)
        self.ar_label.setFont(QFont(self.families[0], 20))
        self.ar_label.move(self.window_width-600, 180)
        self.ar_label.show()

        max_expression_idx = torch.argmax(top_expression_prob, dim=0)
        emotions = [self.model.idx_to_class[idx.item()] for idx in top_expression_class]
        self.ex_label = QLabel(f'Expression: {emotions[max_expression_idx]} ({top_expression_prob[max_expression_idx].item()*100:.3f}%)', self)
        self.ex_label.setFont(QFont(self.families[0], 20))
        self.ex_label.move(self.window_width-600, 220)
        self.ex_label.show()

        max_action_unit_idx = torch.argmax(top_action_unit_prob, dim=0)
        action_units = [self.model.idx_to_au[idx.item()] for idx in top_action_unit_class]
        self.au_label = QLabel(f'Action Unit: {action_units[max_action_unit_idx]} ({top_action_unit_prob[max_action_unit_idx].item()*100:.3f}%)', self)
        self.au_label.setFont(QFont(self.families[0], 20))
        self.au_label.move(self.window_width-600, 260)
        self.au_label.show()

        self.try_again_btn.show()



    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(720, 720, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


app = QtWidgets.QApplication(sys.argv)
login_form = MainWindow()
login_form.show()
sys.exit(app.exec_())