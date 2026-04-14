import cv2
import numpy as np
import threading
import time

'''
用到三帧，当前帧主要用相邻来分割实体。前一帧信息用来遗传id，追踪同一个物体。后一帧信息用来做光流图，根据方向向量，切割同一个物体的重叠物体

长难句准备，也就是缓存三帧，更新pre,last,next，pre是最旧的，

输入一帧后更新三帧

所谓reco部分都是在更新last

继承还是先放一下吧。。。。。。  

result是根据last.ob帧的输出

pov:孩子们注意注释接口和数据类型问题就不会乱。。。。。。
'''


class MoveReco:
    def __init__(self, path, in_queue, out_queue, fps=30,
                 angle_threshold=15, cluster_threshold=10, gray_threshold=0.46):
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
        self.cluster_threshold = cluster_threshold
        self.gray_threshold = gray_threshold

        self.mask = []  # 初筛价值目标,01掩码列表
        self.edge_mask = []  # 边缘提取目标，01掩码列表
        self.fin_edge_p = []  # 边缘提取二次分割，点集列表
        self.fin_mask = []  # 初筛目标二次分割得到最终目标，掩码列表
        self.last_light_direct = None  # 缓存光流图方向阵（last -> next）
        self.last_light_v = None  # 缓存速度阵图
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
        
        self.pre['frame'] = self.last['frame']
        self.pre['ob'] = self.last['ob']
        self.pre['timestamp'] = self.last['timestamp']


        self.last['frame'] = self.next['frame']
        self.last['ob'] = self.next['ob']
        self.last['timestamp'] = self.next['timestamp']

        next_data = self.in_queue.get()

        xy = next_data[1]  # 假设Result对象可直接索引或为元组，忽略区域坐标
        img = next_data[0].img
        timestamp = next_data[0].timestamp

        self.next['timestamp'] = timestamp
        gray_img = self._gray_image_(img, xy)

        self.next['frame'] = gray_img
        self.next['ob'] = None
        self.mask = []
        self.edge_mask = [] 
        self.fin_edge_p = [] 
        self.fin_mask = []
        self.last_light_direct = None
        self.last_light_v = None 

    def _connected(self):
        """
        对last二值图像做连通域分析，初步实现目标识别
        输入：self.last["frame"]
        输出：self.mask
        """
        img = self.last["frame"]
        binary = (img > 0).astype(np.uint8) * 255
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        self.mask = []
        for i in range(1, num_labels):  # 跳过背景0
            mask = (labels == i).astype(np.uint8)
            self.mask.append(mask)

    def _to_edge(self):
        """
        mask边缘提取，二宽度
        输入：self.mask
        输出：self.edge_mask，01掩码数组列表
        """
        self.edge_mask = []
        kernel = np.ones((5, 5), np.uint8)  # 二宽度边缘
        
        for mask in self.mask:
            # 确保mask是二值图像
            mask_uint8 = (mask * 255).astype(np.uint8)
            edge_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, kernel)
            self.edge_mask.append(edge_mask)  # 01掩码数组

    def _flow(self):
        """
        计算 last 和 next 之间的稀疏光流，只要转化为360制角度的等大数组
     
        输入：self.last['frame'], self.next['frame'], self.edge_mask
        输出：self.last_light_direct，nan和float32的360角度的数组
              self.last_light_v, nan和速度的数组
        """
        last_img = (self.last['frame'] * 255).astype(np.uint8)
        next_img = (self.next['frame'] * 255).astype(np.uint8)
        
        # 提取self.edge_mask中所有数组元素的非零点位置
        all_points = []
        for edge_mask in self.edge_mask:
            pts = np.column_stack(np.where(edge_mask > 0))
            all_points.extend(pts)
        
        if len(all_points) == 0:
            h, w = last_img.shape
            self.last_light_direct = np.full((h, w), np.nan, dtype=np.float32)
            self.last_light_v = np.full((h, w), np.nan, dtype=np.float32)
            return
        
        prevPts = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)
        
        # 稀疏光流计算
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            last_img, next_img, prevPts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=0, minEigThreshold=1e-4
        )
        
        h, w = last_img.shape
        self.last_light_direct = np.full((h, w), np.nan, dtype=np.float32)
        self.last_light_v = np.full((h, w), np.nan, dtype=np.float32)
        
        for i, (pt, st) in enumerate(zip(prevPts, status)):
            if st[0] == 1:
                x, y = int(round(pt[0][0])), int(round(pt[0][1]))
                if 0 <= x < w and 0 <= y < h:
                    dx = next_pts[i][0][0] - pt[0][0]
                    dy = next_pts[i][0][1] - pt[0][1]
                    magnitude = np.sqrt(dx**2 + dy**2)
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    if angle < 0:
                        angle += 360
                    
                    self.last_light_v[y, x] = magnitude
                    self.last_light_direct[y, x] = angle

    def _to_flaw(self):
        """
        将edge_mask依据光流图二次划分重叠边缘，输出破碎边缘点集
        输入：self.edge_mask，self.last_light_direct
        输出：self.fin_edge_p，点集列表
        """
        self.fin_edge_p = []
        
        for edge_mask in self.edge_mask:
            cluster = []
            ys, xs = np.where(edge_mask > 0)
            
            for y, x in zip(ys, xs):
                angle = self.last_light_direct[y, x]
                if not np.isnan(angle):
                    cluster.append((angle, x, y))
            
            if len(cluster) == 0:
                continue
            
            # 按照第一位角度大小排序所有元素
            cluster.sort(key=lambda p: p[0])
            
            angles = np.array([p[0] for p in cluster])
            points = np.array([(p[1], p[2]) for p in cluster])
            
            # 计算相邻角度差（考虑360度循环）
            n = len(angles)
            diff = np.zeros(n)
            for i in range(n):
                diff[i] = min(abs(angles[i] - angles[i-1]), 
                              360 - abs(angles[i] - angles[i-1]))
            
            # 查找大于阈值的位置
            large_indices = np.where(diff > self.angle_threshold)[0]
            
            if len(large_indices) > 0:
                # 计算切割点
                cut_points = []
                for idx in large_indices:
                    if idx > 0:
                        mid = (angles[idx] + angles[idx-1]) / 2
                        cut_points.append(mid)
                
                cut_points.sort()
                
                # 按切割点分割
                start = 0
                for cut in cut_points:
                    # 找到第一个角度大于切割点的位置
                    split_idx = np.searchsorted(angles, cut)
                    if split_idx > start:
                        self.fin_edge_p.append(points[start:split_idx])
                        start = split_idx
                if start < n:
                    self.fin_edge_p.append(points[start:])
            else:
                self.fin_edge_p.append(points)

    def _to_mask(self):
        """
        对最终划分边缘，计算弧度填充圆，再按照价值目标裁剪，得到最终区分的目标
        转化为ob

        输入：self.mask，self.fin_edge_p
        输出：self.fin_mask
              self.last["ob"] = {id: {"center": (x,y), "radius": r, "direction": deg, "velocity": 速度标量}}
        """
        self.last["ob"] = {}
        
        h, w = self.last['frame'].shape if self.last['frame'] is not None else (0, 0)
        
        
        # 合并所有原始mask为一个总掩码数组，用于快速匹配
        combined_orig_mask = np.zeros((h, w), dtype=np.uint8)
        for orig_mask in self.mask:
            if orig_mask.shape == (h, w):
                combined_orig_mask = cv2.bitwise_or(combined_orig_mask, orig_mask)
        
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
            
            if np.sum(intersection) == 0:
                continue
            
            # 最终结果圆心和半径
            points_final = np.column_stack(np.where(intersection > 0))
            if len(points_final) < 3:
                continue
            
            (cx, cy), cr = cv2.minEnclosingCircle(points_final.astype(np.float32))
            cx, cy, cr = int(round(cx)), int(round(cy)), int(round(cr))
            
            # 计算平均方向
            dir_angles = []
            vel_values = []
            for y, x in points_final:
                angle = self.last_light_direct[y, x] if self.last_light_direct is not None else np.nan
                v_val = self.last_light_v[y, x] if self.last_light_v is not None else np.nan
                if not np.isnan(angle):
                    dir_angles.append(angle)
                if not np.isnan(v_val):
                    vel_values.append(v_val)
            
            dir_mean = np.mean(dir_angles) if dir_angles else 0.0
            v_mean = np.mean(vel_values) if vel_values else 0.0
            
            # 计算速度标量 (像素/秒)
            if self.last["timestamp"] is not None and self.next["timestamp"] is not None:
                dt = abs(self.next["timestamp"] - self.last["timestamp"])
                velocity = v_mean / dt if dt > 0 else 0.0
            else:
                velocity = 0.0
            
            original_cx = cx 
            original_cy = cy 
            
            ob_id = self._id_()
            self.last["ob"][ob_id] = {
                "center": (original_cx, original_cy),
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







"""
   暂时不知道怎么写
    def _inherit(self):

	对pre帧目标进行预测，根据预测角度与欧式距离进行继承id
	实现目标的跟踪
	输入：self.pre["ob"],self.last["ob"]
	输出：self.last["ob"]

	fore_last={}
	pra_plasce=[]
	for 遍历字典self.pre["ob"]：
		fore_last=(dir,fx,fy)(self.pre["ob"]={"center": (cx,cy), "radius":cr, "direction":dir , "velocity": v},
						dir继承"direction":dir 
						位置用速度和时间差计算，时间差是self.pre["timestamp"]-self.["timestamp"],绝对值）

"""