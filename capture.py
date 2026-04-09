import subprocess
import time
import win32gui
import win32process
import dxcam
import threading
import queue
import os
from PIL import Image
import win32api

class Frame:
    def __init__(self, timestamp, img):
        # 输入：timestamp,时间戳；img，帧的数组信息
        self.timestamp = timestamp
        self.img = img

class Capture:
    def __init__(self, path, fps, q):
        self.path = path
        self.fps = fps
        self.q = q
        self.img = None
        self.hwnd = None
        self.pid = None
        self.rect = None
        self.frame = None
        self.is_life = True
        self.is_capture = False
        self.camera_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.camera = None
        self.work_event = threading.Event()
        
    def start(self):
        process = subprocess.Popen(self.path)
        self.pid = process.pid
        for _ in range(20):
            time.sleep(0.5)
            windows = []
            def enum_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                    if found_pid == self.pid:
                        windows.append(hwnd)
                return True
            win32gui.EnumWindows(enum_callback, windows)
            if windows:
                self.hwnd = windows[0]
                break
        else:
            raise TimeoutError("程序启动超时")
        
        self.check()
        self.resize()
        with self.camera_lock:
            self.camera = dxcam.create(region=self.rect, output_idx=0, backend="dxgi")
        time_thread = threading.Thread(target=self._time, daemon=True)
        run_thread = threading.Thread(target=self._run, daemon=True)
        time_thread.start()
        run_thread.start()
        return time_thread, run_thread
        
    def check(self):
        if self.hwnd:
            if not win32gui.IsWindow(self.hwnd):
                self.is_life = False
                return
            if win32gui.IsIconic(self.hwnd):
                self.is_capture = False
            else:
                self.is_capture = True
            
            current_rect = win32gui.GetWindowRect(self.hwnd)
            if current_rect != self.rect or self.rect is None:
                self.resize()
                
                screen_width = win32api.GetSystemMetrics(0)
                screen_height = win32api.GetSystemMetrics(1)
                if self.rect[2] > screen_width:
                    self.rect = (self.rect[0], self.rect[1], screen_width, self.rect[3])
                if self.rect[3] > screen_height:
                    self.rect = (self.rect[0], self.rect[1], self.rect[2], screen_height)
                if self.rect[0] < 0:
                    self.rect = (0, self.rect[1], self.rect[2], self.rect[3])
                if self.rect[1] < 0:
                    self.rect = (self.rect[0], 0, self.rect[2], self.rect[3])
                with self.camera_lock:
                    if self.camera:
                        self.camera.stop()
                        del self.camera
                    self.camera = dxcam.create(region=self.rect, output_idx=0, backend="dxgi")
        else:
            raise RuntimeError("窗口句柄为空")
            
    def resize(self):
        if self.hwnd:
            self.rect = win32gui.GetWindowRect(self.hwnd)
            
    def catch(self):
        with self.camera_lock:
            if self.camera:
                self.img = self.camera.grab()
        
    def pull(self):
        timestamp = time.time()
        self.frame = Frame(timestamp, self.img)
        self.q.put(self.frame)
    def _time(self):
        while self.is_life:
            sleep_time = 1.0 / self.fps
            time.sleep(sleep_time)
            self.work_event.set()
        
    def _run(self):
        while self.is_life:
            self.work_event.wait()
            self.work_event.clear()
            self.check()
            if self.is_capture:
                self.catch()
                self.pull()
        with self.camera_lock:
            if self.camera:
                self.camera.stop()
                self.camera = None

if __name__ == "__main__":
    q = queue.Queue(maxsize=5)
    a = Capture(r"C:\mypy\[th11] 东方地灵殿 (汉化版+日文版)\th11c.exe", 10, q)
    save_dir = r"C:\mypy\recog_ai\imgs"
    os.makedirs(save_dir, exist_ok=True)
    
    def save_frames():
        while True:
            try:
                frame = q.get(timeout=1)
                if frame and frame.img is not None:
                    img = Image.fromarray(frame.img)
                    filename = os.path.join(save_dir, f"{frame.timestamp}.png")
                    img.save(filename)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"保存出错: {e}")
            time.sleep(3)
    
    time_thread, run_thread = a.start()
    save_thread = threading.Thread(target=save_frames, daemon=True)
    save_thread.start()
    
    try:
        time_thread.join()
        run_thread.join()
    except KeyboardInterrupt:
        pass