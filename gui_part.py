import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal, Qt

class mian_ob(QWidget):
    rela_location = pyqtSignal(tuple)
    win_location = pyqtSignal(tuple)
    is_life = pyqtSignal(bool)
    is_show = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.abs_xy = None
        self.rela_xy = None
        self.win_xy = None
        
        self.rela_location.connect(self.re_rela)
        self.win_location.connect(self.re_win)
        self.is_life.connect(self.life)
        self.is_show.connect(self.show)

    def life(self, is_life):
        if is_life == False:
            self.deleteLater()

    def show(self, is_show):
        if is_show:
            super().show()
        else:
            self.hide()

    def re_rela(self, rela_location):
        self.rela_xy = rela_location
        self.change()

    def re_win(self, win_location):
        self.win_xy = win_location
        self.change()

    def change(self):
        if self.rela_xy is None or self.win_xy is None:
            return
        x1, y1, x2, y2 = self.rela_xy
        win_x1, win_y1, win_x2, win_y2 = self.win_xy
        abs_x1 = win_x1 + x1
        abs_y1 = win_y1 + y1
        abs_x2 = win_x1 + x2
        abs_y2 = win_y1 + y2
        self.abs_xy = (abs_x1, abs_y1, abs_x2, abs_y2)
        self.setGeometry(abs_x1, abs_y1, abs_x2 - abs_x1, abs_y2 - abs_y1)


class main(QMainWindow):
    rela_location = pyqtSignal(tuple)
    win_location = pyqtSignal(tuple)
    is_life = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.lemon = mian_ob()
        self.child_win = self.lemon
        
        self.rela_location.connect(self.re_rela)
        self.win_location.connect(self.re_win)
        self.is_life.connect(self.life)

    def life(self, is_life):
        if is_life == False:
            self.lemon.deleteLater()
            self.deleteLater()

    def re_rela(self, rela_location):
        self.lemon.rela_location.emit(rela_location)

    def re_win(self, win_location):
        self.lemon.win_location.emit(win_location)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = main()
    window.show()
    sys.exit(app.exec_())
