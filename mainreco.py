import threading
import time
import queue
from ultralytics import YOLO
import cv2

class Result:
    def __init__(self, timestamp, xy):
        # timestamp: 时间戳; xy: 位置list [x1, y1, x2, y2]
        self.timestamp = timestamp
        self.xy = xy

class Mainreco:
    def __init__(self, path, pre_queue, out_queue, fps=30):
        # path: 模型地址; pre_queue: 输入队列(识别图); out_queue: 输出队列(结果); fps: 最高处理帧率
        self.path = path
        self.pre_queue = pre_queue
        self.out_queue = out_queue
        self.fps = fps
        self.is_life = True
        self.pre_result = None
        self.model = None

    def start(self):
        self.model = YOLO(self.path)
        run_thread = threading.Thread(target=self._run, daemon=True)
        run_thread.start()
        return run_thread

    def reco(self):
        frame = self.pre_queue.get()      
        timestamp = frame.timestamp
        img = frame.img

            # 压缩到416x416
        img_resized = cv2.resize(img, (416, 416))

            # 模型推理
        results = self.model(img_resized, verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            result_xy = results[0].boxes.xyxy[0].tolist()
            # 坐标缩放回原图尺寸
            h, w = img.shape[:2]
            scale_x = w / 416
            scale_y = h / 416
            result_xy = [
                result_xy[0] * scale_x,
                result_xy[1] * scale_y,
                result_xy[2] * scale_x,
                result_xy[3] * scale_y
                ]
            self.pre_result = Result(timestamp, result_xy)
        else:
            self.pre_result = Result(timestamp, None)


    def pull(self):
        if self.pre_result is not None:
            self.out_queue.put(self.pre_result)

    def _run(self):
        last_time = time.time()
        min_interval = 1.0 / self.fps

        while self.is_life:
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= min_interval:
                self.reco()
                self.pull()
                last_time = current_time
            else:
                time.sleep(min_interval - elapsed)


#这玩意是标记验证测试，在原图画框
if __name__ == "__main__":
    import cv2
    import os
    from ultralytics import YOLO
    
    path = r"C:\mypy\recog_ai\imgs"
    
    # 把path内png文件，转化为class Frame存入列表b
    class Frame:
        def __init__(self, timestamp, img):
            # 输入：timestamp,文件名称,时间戳；img，文件的数组信息
            self.timestamp = timestamp
            self.img = img
    
    b = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            img_path = os.path.join(path, file)
            # 提取文件名作为时间戳（去掉.png后缀）
            timestamp = file.replace(".png", "")
            # 读取图像为数组
            img_array = cv2.imread(img_path)
            if img_array is not None:
                b.append(Frame(timestamp, img_array))
    
    # 新建文件夹在path,为result
    result_dir = os.path.join(path, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(r"C:\mypy\recog_ai\runs\detect\train\weights\best.pt")
    
    # 遍历b，进行识别并画框
    for frame in b:
        img = frame.img
        timestamp = frame.timestamp
        
        # 模型预测
        results = model(img, verbose=False)
        
        # 打印该张图片的识别结果
        print(f"\n图片: {timestamp}.png")
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # 获取所有检测结果
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                xy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                print(f"  检测框{i+1}: 坐标={xy}, 置信度={conf:.3f}, 类别={cls}")
                
                # 在图像上画框
                x1, y1, x2, y2 = [int(coord) for coord in xy]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            print("  未检测到任何目标")
        
        # 保存到result文件夹
        save_path = os.path.join(result_dir, f"{timestamp}.png")
        cv2.imwrite(save_path, img)
        print(f"已保存: {save_path}")
