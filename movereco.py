import cv2
import numpy as np
import threading
import time
from sklearn.cluster import DBSCAN

'''
用到三帧，当前帧主要用相邻来分割实体。前一帧信息用来遗传id，追踪同一个物体。后一帧信息用来做光流图，根据方向向量，切割同一个物体的重叠物体

长难句准备，也就是缓存三帧，更新pre,last,next，pre是最旧的，

输入一帧后更新三帧

所谓reco部分都是在更新last

last先邻近提取有效目标

last核nest帧差，提取有效位置，稠密光流

其次last根据光流分割有效部分，区分重叠位置

最终依据离散类聚点输出最大外界圆参数

继承还是先放一下吧。。。。。。  

result是根据last.ob帧的输出

pov:孩子们注意注释接口和数据类型问题就不会乱。。。。。。
'''


class MoveReco:
    def __init__(self, path, in_queue, out_queue, fps=30,
                 angle_threshold=15, cluster_threshold=10, 
                 min_threshold=10, gray_threshold=0.46):
        # in_queue: 输入类管理队列对象，.get得到[Frame，Result(忽略区域)]
        # out_queue: 输出队列(结果)
        self.path = path
        self.in_queue = in_queue
        self.out_queue = out_queue

        # 缓存三帧数据: pre(最旧), last(当前), next(最新)
        self.pre = {'frame': None, 'ob': None, 'timestamp': None}
        self.last = {'frame': None, 'ob': None, 'timestamp': None}
        self.next = {'frame': None, 'ob': None, 'timestamp': None}

        self.fps = fps
        self.is_life = True

        self.result = None

        self.angle_threshold = angle_threshold
        self.min_threshold = min_threshold       # svd肘点阈值
        self.cluster_threshold = cluster_threshold
        self.gray_threshold = gray_threshold

        self.mask_labels = None  # 修改：直接保存连通域分析的labels数组
        self.mask_stats = None   # 保存连通域分析的stats
        self.mask_centroids = None  # 保存连通域分析的centroids
        self.edge_mask = []  # 边缘提取目标，01掩码列表
        self.fin_edge_p = []  # 边缘提取二次分割，点集列表
        self.fin_mask = []  # 初筛目标二次分割得到最终目标，掩码列表
        self.flow = None  # 数组[h,w,2]
        self.next_target_id = 0  # 用于生成唯一标识

    def _id_(self):
        """生成新的唯一标识"""
        self.next_target_id += 1
        return self.next_target_id

    def _gray_image_(self, img, xy):
        """
        图像预处理：灰度化、归一化、阈值过滤、忽略区域过滤和有效区域裁剪
 
        输入：img(BGR三通道), xy(忽略区域坐标 [x1,y1,x2,y2])
        输出：processed_img(灰度图 float32 0-1)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        gray[gray < self.gray_threshold] = 0

        if xy and len(xy) == 4:
            x1, y1, x2, y2 = [int(round(v)) for v in xy]
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                gray[y1:y1+h, x1:x1+w] = 0

        return gray

    def _dataup(self):
        """
        数据更新：从队列获取数据，交替接收帧和忽略区域
 
        输出：更新 self.pre, self.last, self.next
        """
        if self.last['frame'] is not None:
            self.pre['frame'] = self.last['frame']
            self.pre['ob'] = self.last['ob']
            self.pre['timestamp'] = self.last['timestamp']

        if self.next['frame'] is not None:
            self.last['frame'] = self.next['frame']
            self.last['ob'] = self.next['ob']
            self.last['timestamp'] = self.next['timestamp']

        next_data = self.in_queue.get()
        # self.in_queue.get={"xy","img"}
        xy = next_data["xy"]
        img = next_data["img"]
        timestamp = next_data["timestamp"]

        self.next['timestamp'] = timestamp
        gray_img = self._gray_image_(img, xy)

        self.next['frame'] = gray_img
        self.next['ob'] = None
        self.mask_labels = None
        self.mask_stats = None
        self.mask_centroids = None
        self.edge_mask = []
        self.fin_edge_p = []
        self.fin_mask = []
        self.flow = None

    def _connected(self):
        """
        对last二值图像做连通域分析，初步实现目标识别
        输入：self.last["frame"]
        输出：self.mask_labels, self.mask_stats, self.mask_centroids
        """
        img = self.last["frame"]
        binary = (img > 0).astype(np.uint8) * 255
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 直接保存原始labels数组，不划分为单个mask
        self.mask_labels = labels
        self.mask_stats = stats
        self.mask_centroids = centroids

    def _to_edge(self):
        """
        mask边缘提取，二宽度
        输入：self.mask_labels
        输出：self.edge_mask，01掩码数组列表
        """
        self.edge_mask = []
        kernel = np.ones((5, 5), np.uint8)  # 二宽度边缘
        
        if self.mask_labels is None:
            return
            
        unique_labels = np.unique(self.mask_labels)
        for label in unique_labels:
            if label == 0:  # 跳过背景
                continue
            # 从labels数组中提取单个物体的掩码
            mask = (self.mask_labels == label).astype(np.uint8) * 255
            edge_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            self.edge_mask.append(edge_mask)  # 01掩码数组

    def _flow(self):
        """
        计算 last 和 next 之间的稠密光流
     
        输入：self.last['frame'], self.next['frame']
        输出：self.flow
        """
        last_img = (self.last['frame'] * 255).astype(np.uint8)
        next_img = (self.next['frame'] * 255).astype(np.uint8)

        # 帧差法处理last和nextimg，所得帧差图取非零元素最大外界矩形
        diff = cv2.absdiff(last_img, next_img)
        _, diff_binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        non_zero_points = np.column_stack(np.where(diff_binary > 0))
        
        if len(non_zero_points) == 0:
            # 如果没有运动区域，计算全图光流
            flow = cv2.calcOpticalFlowFarneback(
                last_img, next_img, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )
            self.flow = flow
            return
        
        # 计算非零元素的最大外界矩形
        y_min, x_min = non_zero_points.min(axis=0)
        y_max, x_max = non_zero_points.max(axis=0)
        
        # 添加边距
        margin = 10
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(last_img.shape[0], y_max + margin)
        x_max = min(last_img.shape[1], x_max + margin)
        
        # 对last_img和next_img裁剪得到re_last和re_next
        re_last = last_img[y_min:y_max, x_min:x_max]
        re_next = next_img[y_min:y_max, x_min:x_max]
        
        # 局部稠密光流计算
        flow_roi = cv2.calcOpticalFlowFarneback(
            re_last, re_next, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        
        # 将ROI光流映射回原图坐标系
        h, w = last_img.shape
        self.flow = np.zeros((h, w, 2), dtype=np.float32)
        self.flow[y_min:y_max, x_min:x_max] = flow_roi

    def _to_flaw(self):
        """
        将edge_mask依据光流图二次划分重叠边缘，无监督类聚
        输入：self.edge_mask，self.flow
        输出：self.fin_edge_p，点集列表
        """
        self.fin_edge_p = []
        
        if self.flow is None:
            return
            
        for edge_mask in self.edge_mask:
            # 掩码提取self.flow中相应点的位置参数得到矢量速度矩阵
            ys, xs = np.where(edge_mask > 0)
            if len(ys) < 3:
                continue
                
            # 合成v_flow=[n,2]
            v_flow = []
            for y, x in zip(ys, xs):
                vx = self.flow[y, x, 0]
                vy = self.flow[y, x, 1]
                v_flow.append([vx, vy])
            
            v_flow = np.array(v_flow)
            
            # DBSCAN聚类
            clustering = DBSCAN(eps=5, min_samples=3).fit(v_flow)
            labels = clustering.labels_
            （

    def _to_mask(self):
        """
        对最终类聚点，计算弧度填充圆，再按照价值目标裁剪，得到最终区分的目标
        转化为ob

        输入：self.mask_labels，self.fin_edge_p
        输出：self.fin_mask
              self.last["ob"] = {id: {"center": (x,y), "radius": r, "direction": deg, "velocity": 速度标量}}
        """
        self.last["ob"] = {}
        self.fin_mask = []
        
        if self.mask_labels is None:
            return
            
        h, w = self.last['frame'].shape if self.last['frame'] is not None else (0, 0)
        
        # 合并所有原始mask为一个总掩码数组（基于labels）
        combined_orig_mask = np.zeros((h, w), dtype=np.uint8)
        if self.mask_labels is not None:
            combined_orig_mask = (self.mask_labels > 0).astype(np.uint8)
        
        for edge_points in self.fin_edge_p:
            if len(edge_points) < 3:
                continue
            
            # 计算最小外接圆
            points_float = edge_points.astype(np.float32)
            (rx, ry), r = cv2.minEnclosingCircle(points_float)
            rx, ry = int(round(rx)), int(round(ry))
            r = int(round(r))
            
            # 创建圆形掩码
            cir = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(cir, (rx, ry), r, 1, -1)
            
            # 数组掩码匹配：直接与合并后的原始mask求交
            intersection = cv2.bitwise_and(combined_orig_mask, cir)
            
            if np.sum(intersection) < 10:
                continue
            
            # 最终结果圆心和半径
            points_final = np.column_stack(np.where(intersection > 0))
            if len(points_final) < 3:
                continue
            
            (cx, cy), cr = cv2.minEnclosingCircle(points_final.astype(np.float32))
            cx, cy, cr = int(round(cx)), int(round(cy)), int(round(cr))
            
            # 创建最终掩码并保存
            final_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(final_mask, (cx, cy), cr, 1, -1)
            self.fin_mask.append(final_mask)
            
            # 计算平均方向（使用光流数据）
            dir_angles = []
            vel_values = []
            if self.flow is not None:
                for y, x in points_final:
                    vx = self.flow[y, x, 0]
                    vy = self.flow[y, x, 1]
                    if abs(vx) > 0.01 or abs(vy) > 0.01:
                        angle = np.arctan2(vy, vx) * 180 / np.pi
                        if angle < 0:
                            angle += 360
                        dir_angles.append(angle)
                        vel_values.append(np.sqrt(vx**2 + vy**2))
            
            dir_mean = np.mean(dir_angles) if dir_angles else 0.0
            v_mean = np.mean(vel_values) if vel_values else 0.0
            
            # 计算速度标量 (像素/秒)
            if self.last["timestamp"] is not None and self.next["timestamp"] is not None:
                dt = abs(self.next["timestamp"] - self.last["timestamp"])
                velocity = v_mean / dt if dt > 0 else 0.0
            else:
                velocity = 0.0
            
            ob_id = self._id_()
            self.last["ob"][ob_id] = {
                "center": (cx, cy),
                "radius": cr,
                "direction": dir_mean,
                "velocity": velocity
            }

    def pull(self):
        """将识别结果放入输出队列"""
        self.out_queue.put(self.last["ob"])

    def reco(self):
        """执行完整的识别流程"""
        self._dataup()
        if self.last['frame'] is None:
            return
        self._connected()
        self._to_edge()
        if self.next['frame'] is not None:
            self._flow()
        self._to_flaw()
        self._to_mask()

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

    def start(self):
        run_thread = threading.Thread(target=self._run, daemon=True)
        run_thread.start()
        return run_thread