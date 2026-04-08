import threading
import time
import queue
from ultralytics import YOLO
import cv2
from capture import Capture
from mainreco import Mainreco
from gui_part import win_frame
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    # 创建队列
    q = queue.Queue(maxsize=5)  # 截图队列
    w = queue.Queue(maxsize=5)  # 识别结果队列
    
    # 创建截图对象a
    a = Capture(r"C:\mypy\[th11] 东方地灵殿 (汉化版+日文版)\th11c.exe", 30, q)
    time_thread, run_thread = a.start()  # 启动截图线程   
    # 创建识别对象b
    b = Mainreco(r"C:\mypy\recog_ai\runs\detect\train\weights\best.pt", q, w, fps=30)
    
    # 创建Qt应用
    app = QApplication(sys.argv)
    
    # 创建GUI对象d，传入a.hwnd
    d = win_frame(a.hwnd, fps=30)
    
    # 线程c：接收队列w，发送信号给d
    def thread_c():
        while True:
            try:
                result = w.get(timeout=1)
                if result and result.xy is not None:
                    # 发出信号给d，传递相对坐标
                    d.rela_location.emit(result.xy)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"线程c出错: {e}")
            time.sleep(0.01)  # 避免空转
    
    # 启动三个线程
    time_thread, run_thread = a.start()  # 启动截图线程
    reco_time_thread, reco_run_thread = b.start()  # 启动识别线程
    c_thread = threading.Thread(target=thread_c, daemon=True)  # 线程c
    c_thread.start()
    
    # 启动GUI
    d.start()  # 启动工作线程
    d.show()  # 显示窗口
    
    # 进入Qt主循环
    sys.exit(app.exec_())