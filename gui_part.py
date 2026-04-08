from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QColor
import win32gui
import queue

class win_frame(QWidget):
    rela_location = pyqtSignal(list)
    is_life = pyqtSignal(bool)
    
    def __init__(self, hwnd, fps=30):
        # 继承pyqt5子窗口类
        super().__init__()
        
        # 隐藏title,窗口框，背景
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background:transparent;")
        
        # 输入：hwnd，覆盖窗口对象的句柄；fps=30
        self.hwnd = hwnd
        self.fps = fps
        
        # 容器：q_queue，队列流式久出新入，最多缓存一个
        self.q_queue = queue.Queue(maxsize=1)
        # rela_xy,缓存模型输出的四位数组相对目标窗口位置
        self.rela_xy = None
        # win_xy,缓存覆盖目标窗口的信息
        self.win_xy = None
        # pre_xy，缓存当前位置信息（保留用于调试，paintEvent中未使用）
        self.pre_xy = None
        # is_show
        self.is_show = False
        
        # 创立计时器work=qtimer(33毫秒)
        self.work_timer = QTimer()
        self.work_timer.setInterval(33)  # 约30fps
        # 注册信号work变化执行self._run()槽函数
        self.work_timer.timeout.connect(self._run)
        # 注册信号rela_location变化执行self.change_rela()槽函数
        self.rela_location.connect(self.change_rela)
        # 注册信号is_life,is_life变化执行self.life
        self.is_life.connect(self.life)
        
    def life(self, life_flag):
        if life_flag == False:
            # 销毁窗口释放内存，结束窗口循环
            self.work_timer.stop()
            self.close()
            self.deleteLater()
            
    def change_rela(self, rela_data):
        # 将rela_location加入队列（非阻塞，满则丢弃）
        try:
            self.q_queue.put(rela_data, block=False)
        except queue.Full:
            pass
            
    def _run(self):
        # 队列q_queue为空
        if self.q_queue.empty():
            self.is_show = False
            # 根据句柄获得目标窗口在桌面位置信息
            try:
                if win32gui.IsWindow(self.hwnd):
                    current_rect = win32gui.GetWindowRect(self.hwnd)
                    # 若win_xy变化且有效，更新win_xy
                    if current_rect != self.win_xy:
                        self.win_xy = current_rect
                else:
                    print("窗口句柄无效")
            except Exception as e:
                print(f"获取窗口位置失败: {e}")
            
            self.change()
        else:
            # 更新rela_xy，非阻塞get
            try:
                self.rela_xy = self.q_queue.get_nowait()
                self.is_show = True
                self.change()
            except queue.Empty:
                pass
                
    def change(self):
        if self.is_show and self.rela_xy is not None and self.win_xy is not None:
            # rela_xy格式假设为 [x1, y1, x2, y2] 相对坐标(0-1范围)
            win_x1, win_y1, win_x2, win_y2 = self.win_xy
            win_width = win_x2 - win_x1
            win_height = win_y2 - win_y1
            
            # 计算绝对位置
            abs_x1 = win_x1 + int(self.rela_xy[0] * win_width)
            abs_y1 = win_y1 + int(self.rela_xy[1] * win_height)
            abs_x2 = win_x1 + int(self.rela_xy[2] * win_width)
            abs_y2 = win_y1 + int(self.rela_xy[3] * win_height)
            
            self.pre_xy = (abs_x1, abs_y1, abs_x2, abs_y2)
            
            # 窗口改变为pre_xy
            self.setGeometry(abs_x1, abs_y1, abs_x2 - abs_x1, abs_y2 - abs_y1)
            
            # 显示并重绘
            self.show()
            self.update()
        else:
            # 隐藏
            self.hide()
            
    def paintEvent(self, event):
        # 只要窗口可见就绘制边框，不依赖pre_xy
        if self.is_show and self.width() > 0 and self.height() > 0:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 2)  # 红色边框，2像素宽
            painter.setPen(pen)
            # 绘制比窗口小1像素的框
            rect = QRect(1, 1, self.width() - 2, self.height() - 2)
            painter.drawRect(rect)
    
    def start(self):
        # 启动计时器
        self.work_timer.start()
        self.show()
        
    def stop(self):
        self.is_life.emit(False)