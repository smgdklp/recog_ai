import cv2
import numpy as np
import threading
import time

'''
用到三帧，主要用相邻来分割实体。
生成两张光流图，用来分割前两张的识别所得的物体的重合物体。

长难句准备，也就是缓存三帧
缓存三帧，更新pre,last,next，pre是最旧的，丢了
最新帧connectedComponentsWithStats，分离labels出不同物体，保存到cluster[特异标识,位置掩码]
做last和next的光流，last光流图保存移动矢量图
last.cluster每个位置掩码用last光流图做二次分割，按照angle_diff = lambda a, b: min(abs(a - b), 360 - abs(a - b))，
如果存在大于阈值个数的异类则分割。分割重合的物体，因为光流图会有空心，所以按照能囊括同向点的最大外接圆且在掩码上有值的分割，
分割的物体赋予新的特异标识，删掉原来的物体，所有保存为cluster[特异标识,[[方向，x,y],....的点集]]
然后把分割的所有cluster转化为（"特异标识"：字典{最大外接圆半径，最大外接圆坐标，方向，速度}），方向用平均角度，保存到ob
根据pre.ob内物体圆心位置,速度,和速度方向和timestamp差估计last.ob位置，如果相差欧式距离为10则取距离最小的点继承特异标识
'''

class Result:
    """输出结果类：存储当前帧所有检测到的物体"""
    def __init__(self, timestamp=None, objects=None):
        self.timestamp = timestamp
        # objects 字典: {"特异标识": {"center": (x,y), "radius": r, "direction": deg, "velocity": (vx,vy)}}
        self.objects = objects if objects is not None else {}
        self.xy = None  # 兼容忽略区域传递


class Frame:
    """帧数据容器"""
    def __init__(self, timestamp, img, xy=None):
        self.timestamp = timestamp
        self.img = img          # BGR三通道
        self.xy = xy            # 忽略区域坐标 [x1,y1,x2,y2]（来自上游 Result）


class MoveFrame:
    """内部帧数据容器（预处理后）"""
    def __init__(self, img, xy=None, timestamp=None):
        self.img = img          # 灰度图 float32 0-1
        self.xy = xy            # 忽略区域坐标
        self.timestamp = timestamp


class MoveReco:
    def __init__(self, path, in_queue, out_queue, fps=30, 
                 angle_threshold=15, cluster_threshold=10, gray_threshold=0.46):
        # in_queue: 输入队列，交替接收 Frame 和 Result(忽略区域)
        # out_queue: 输出队列(结果)
        self.path = path
        self.in_queue = in_queue
        self.out_queue = out_queue

        # 缓存三帧数据: pre(最旧), last(当前), next(最新)
        self.pre = {'frame': None, 'cluster': None, 'ob': None, 'timestamp': None}
        self.last = {'frame': None, 'cluster': None, 'ob': None, 'timestamp': None}
        self.next = {'frame': None, 'cluster': None, 'ob': None, 'timestamp': None}

        self.fps = fps
        self.is_life = True
        self.pre_result = None
        self.angle_threshold = angle_threshold
        self.cluster_threshold = cluster_threshold
        self.gray_threshold = gray_threshold

        self.last_light_direct = None   # 缓存光流图方向阵（last -> next）
        self.next_target_id = 0          # 用于生成唯一标识

        # 临时变量，避免重复创建
        self._temp_mask = None
        self._temp_binary = None


    def start(self):
        run_thread = threading.Thread(target=self._run, daemon=True)
        run_thread.start()
        return run_thread


    def _get_new_id(self):
        """生成新的唯一标识"""
        self.next_target_id += 1
        return self.next_target_id - 1

    def __preprocess_image(self, img, xy):
        """
        图像预处理：灰度化、归一化、阈值过滤、忽略区域处理
        
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
        Frame 和 Result(忽略区域) 成对出现，顺序：Frame -> Result -> Frame -> Result ...
        
        输出：更新 self.pre, self.last, self.next
        """
        # 滚动更新: pre 丢弃, last 变 pre, next 变 last
        if self.last['frame'] is not None:
            self.pre['frame'] = self.last['frame']
            self.pre['cluster'] = self.last['cluster']
            self.pre['ob'] = self.last['ob']
            self.pre['timestamp'] = self.last['timestamp']

        if self.next['frame'] is not None:
            self.last['frame'] = self.next['frame']
            self.last['cluster'] = self.next['cluster']
            self.last['ob'] = self.next['ob']
            self.last['timestamp'] = self.next['timestamp']

        # 从队列获取新帧（阻塞等待）
        frame_data = self.in_queue.get()
        if not isinstance(frame_data, Frame):
            raise TypeError(f"期望 Frame 类型，实际得到 {type(frame_data)}")
        
        # 从队列获取对应的忽略区域（阻塞等待）
        ig_data = self.in_queue.get()
        if not isinstance(ig_data, Result):
            raise TypeError(f"期望 Result(忽略区域) 类型，实际得到 {type(ig_data)}")
        
        ig_xy = ig_data.xy

        img = frame_data.img
        timestamp = frame_data.timestamp

        self.next['timestamp'] = timestamp
        processed_img = self.__preprocess_image(img, ig_xy)

        self.next['frame'] = MoveFrame(processed_img, ig_xy, timestamp)
        self.next['cluster'] = None
        self.next['ob'] = None

    def _calc_optical_flow(self):
        """
        光流法：计算 last 和 next 之间的光流场（从 last 到 next）
        
        输入：self.last['frame'].img, self.next['frame'].img
        输出：self.last_light_direct (光流场)
        """
        if self.last['frame'] is None or self.next['frame'] is None:
            self.last_light_direct = None
            return

        pre_img = self.last['frame'].img
        last_img = self.next['frame'].img

        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            pre_img, last_img, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.last_light_direct = flow


    def _connected_components(self, img):
        """
        对二值图像做连通域分析，结果存入实例变量
        
        输入：img(灰度图)
        输出：self._temp_labels, self._temp_stats, self._temp_centroids, self._temp_num_labels
        """
        binary = (img > 0).astype(np.uint8) * 255
        self._temp_labels, self._temp_stats, self._temp_centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )[:3]
        self._temp_num_labels = self._temp_labels.max() if self._temp_labels.size > 0 else 0


    def _segment_connected_components(self, target_frame_key='next'):
        """
        临近物体检测：对指定帧做连通域分析，分离出不同物体
        
        输入：target_frame_key - 'last' 或 'next'
        输出：更新 self[target_frame_key]['cluster']
        """
        frame_data = self.last if target_frame_key == 'last' else self.next
        if frame_data['frame'] is None:
            return

        img = frame_data['frame'].img
        self._connected_components(img)
        
        cluster = {}
        for i in range(1, self._temp_num_labels + 1):
            mask = (self._temp_labels == i)
            y_coords, x_coords = np.where(mask)
            if len(y_coords) < self.cluster_threshold:
                continue

            points = list(zip(x_coords.tolist(), y_coords.tolist()))
            target_id = self._get_new_id()
            
            # 获取质心
            centroid_x = self._temp_centroids[i][0] if i < len(self._temp_centroids) else sum(x_coords)/len(x_coords)
            centroid_y = self._temp_centroids[i][1] if i < len(self._temp_centroids) else sum(y_coords)/len(y_coords)
            
            # 获取边界框
            stats = self._temp_stats[i] if i < len(self._temp_stats) else None
            if stats is not None:
                bbox = (stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP],
                       stats[cv2.CC_STAT_WIDTH], stats[cv2.CC_STAT_HEIGHT])
            else:
                bbox = (min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords))
            
            cluster[target_id] = {
                'mask': mask,
                'points': points,
                'center': (centroid_x, centroid_y),
                'bbox': bbox,
                'velocity': (0, 0),
                'direction': 0,
                'radius': 0
            }

        frame_data['cluster'] = cluster


    def _extract_points_with_flow(self, mask):
        """
        从掩码区域提取光流信息，结果存入实例变量
        
        输入：mask(布尔掩码)
        输出：self._temp_points (列表 [(angle, x, y, vx, vy), ...])
        """
        if self.last_light_direct is None or mask is None:
            self._temp_points = []
            return

        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            self._temp_points = []
            return

        vx_vals = self.last_light_direct[y_coords, x_coords, 0]
        vy_vals = self.last_light_direct[y_coords, x_coords, 1]

        # 计算角度
        angles = np.arctan2(vy_vals, vx_vals) * 180 / np.pi
        angles[angles < 0] += 360

        points = []
        for i in range(len(y_coords)):
            points.append((angles[i], int(x_coords[i]), int(y_coords[i]), 
                          float(vx_vals[i]), float(vy_vals[i])))
        self._temp_points = points


    def _angle_clustering(self):
        """
        方向聚类：按角度分组，结果存入 self._temp_angle_groups
        
        输入：self._temp_points
        输出：self._temp_angle_groups (分组后的列表)
        """
        points = self._temp_points
        if not points:
            self._temp_angle_groups = []
            return
        
        points.sort(key=lambda p: p[0])
        angle_diff = lambda a, b: min(abs(a - b), 360 - abs(a - b))

        groups = []
        cur_group = [points[0]]
        for i in range(1, len(points)):
            if angle_diff(points[i][0], points[i-1][0]) > self.angle_threshold:
                groups.append(cur_group)
                cur_group = [points[i]]
            else:
                cur_group.append(points[i])
        if cur_group:
            groups.append(cur_group)
        self._temp_angle_groups = groups


    def _reconstruct_mask_from_points(self, points, shape):
        """
        从点集重建掩码
        
        输入：points列表 [(angle, x, y, vx, vy), ...], shape(图像形状)
        输出：mask(布尔掩码)
        """
        mask = np.zeros(shape, dtype=bool)
        for _, x, y, _, _ in points:
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                mask[y, x] = True
        return mask


    def _split_single_cluster_by_flow(self, target_data):
        """
        根据光流方向分割同一连通域内方向不同的点
        
        输入：target_data(单个物体数据)
        输出：self._temp_split_objects (分割后的物体列表)
        """
        self._extract_points_with_flow(target_data['mask'])
        if len(self._temp_points) < self.cluster_threshold:
            self._temp_split_objects = [target_data]
            return

        self._angle_clustering()
        if len(self._temp_angle_groups) <= 1:
            self._temp_split_objects = [target_data]
            return

        # 多个方向，需要分割
        new_objects = []
        for group in self._temp_angle_groups:
            if len(group) < 5:
                continue

            # 提取该组点的坐标和光流
            xs = [p[1] for p in group]
            ys = [p[2] for p in group]
            vxs = [p[3] for p in group]
            vys = [p[4] for p in group]

            # 计算外接圆
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            radius = max(((x - center_x)**2 + (y - center_y)**2) ** 0.5 
                        for x, y in zip(xs, ys))

            # 计算平均速度和方向
            avg_vx = sum(vxs) / len(vxs)
            avg_vy = sum(vys) / len(vys)
            direction = np.arctan2(avg_vy, avg_vx) * 180 / np.pi
            if direction < 0:
                direction += 360

            # 重建掩码
            mask = self._reconstruct_mask_from_points(group, target_data['mask'].shape)

            new_obj = {
                'mask': mask,
                'points': group,
                'center': (center_x, center_y),
                'radius': radius,
                'velocity': (avg_vx, avg_vy),
                'direction': direction,
                'bbox': target_data.get('bbox', (0, 0, 0, 0))
            }
            new_objects.append(new_obj)

        self._temp_split_objects = new_objects if new_objects else [target_data]


    def _split_by_flow_direction(self):
        """
        光流法分割：用 last->next 的光流，去分割 last['cluster']（运动物体）
        因为光流表示的是 last 帧中的点运动到 next 帧的向量，
        所以应该分割 last 帧中的物体，而不是 next
        
        输入：self.last['cluster'], self.last_light_direct
        输出：更新 self.last['cluster'] (分割后的物体)
        """
        if self.last['cluster'] is None or self.last_light_direct is None:
            return

        new_cluster = {}

        for target_id, target_data in self.last['cluster'].items():
            # 对该物体的掩码区域进行光流方向分割
            self._split_single_cluster_by_flow(target_data)

            for obj in self._temp_split_objects:
                new_id = self._get_new_id()
                new_cluster[new_id] = {
                    'mask': obj['mask'],
                    'points': obj['points'],
                    'center': obj['center'],
                    'radius': obj['radius'],
                    'velocity': obj['velocity'],
                    'direction': obj['direction']
                }

        self.last['cluster'] = new_cluster


    def _cluster_to_ob(self, cluster):
        """
        将 cluster 转化为 ob 格式（输出用），直接修改传入的 ob 字典
        
        输入：cluster(物体字典，含完整信息)
        输出：ob(精简格式 {"id": {"center": (x,y), "radius": r, "direction": deg, "velocity": (vx,vy)}})
        """
        if not cluster:
            return {}
        ob = {}
        for target_id, data in cluster.items():
            ob[target_id] = {
                'center': data.get('center', (0, 0)),
                'radius': data.get('radius', 0),
                'direction': data.get('direction', 0),
                'velocity': data.get('velocity', (0, 0))
            }
        return ob


    def _predict_and_match_ids(self):
        """
        根据 pre['ob'] 预测位置，与 last['ob'] 匹配
        如果预测位置与实际位置欧氏距离小于阈值，则继承特异标识
        
        输入：self.pre['ob'], self.last['ob'], self.pre['timestamp'], self.last['timestamp']
        输出：更新 self.last['ob'] 中的 ID（继承后的标识）
        """
        if self.pre['ob'] is None or self.last['ob'] is None:
            return

        # 计算实际时间差
       
        dt = self.last['timestamp'] - self.pre['timestamp']
        # 对 pre 中每个物体，预测其在当前帧的位置
        predictions = []
        for obj_id, obj_data in self.pre['ob'].items():
            cx, cy = obj_data['center']
            vx, vy = obj_data['velocity']
            pred_cx = cx + vx * dt
            pred_cy = cy + vy * dt
            predictions.append((obj_id, (pred_cx, pred_cy)))

        # 匹配 last['ob'] 中的物体
        last_ob = self.last['ob'].copy()
        matched_pre_ids = set()
        matched_last_ids = set()
        new_last_ob = {}

        # 先处理匹配的物体
        for pre_id, pred_pos in predictions:
            best_match_id = None
            best_dist = 15.0  # 距离阈值（像素）
            
            for last_id, last_data in last_ob.items():
                if last_id in matched_last_ids:
                    continue
                last_pos = last_data['center']
                dist = ((pred_pos[0] - last_pos[0])**2 + (pred_pos[1] - last_pos[1])**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_match_id = last_id

            if best_match_id is not None:
                matched_pre_ids.add(pre_id)
                matched_last_ids.add(best_match_id)
                # 继承 pre 的 ID，保留 last 的数据
                new_last_ob[pre_id] = last_ob[best_match_id]
            else:
                # 没有匹配的，保留原 pre_id？不，pre 是上一帧的，不应该出现在当前帧
                pass

        # 添加未匹配的 last 物体（新出现的物体）
        for last_id, last_data in last_ob.items():
            if last_id not in matched_last_ids:
                new_last_ob[last_id] = last_data

        self.last['ob'] = new_last_ob


    def _update_ob_velocity(self):
        """
        更新 next['ob'] 中物体的速度
        通过比较 last 和 next 的位置差计算实际速度
        
        输入：self.last['ob'], self.next['ob'], self.last['timestamp'], self.next['timestamp']
        输出：更新 self.next['ob'] 中的 velocity
        """
        if self.last['ob'] is None or self.next['ob'] is None:
            return

        # 计算时间差
      
        dt = self.next['timestamp'] - self.last['timestamp']

        for next_id, next_data in self.next['ob'].items():
            best_match_id = None
            best_dist = 15.0
            
            for last_id, last_data in self.last['ob'].items():
                last_pos = last_data['center']
                next_pos = next_data['center']
                dist = ((next_pos[0] - last_pos[0])**2 + (next_pos[1] - last_pos[1])**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_match_id = last_id

            if best_match_id is not None:
                last_data = self.last['ob'][best_match_id]
                last_pos = last_data['center']
                next_pos = next_data['center']
                vx = (next_pos[0] - last_pos[0]) / dt
                vy = (next_pos[1] - last_pos[1]) / dt
                self.next['ob'][next_id]['velocity'] = (vx, vy)


    def _reco(self):
        """
        识别主函数：串联所有处理步骤
        
        处理流程：
        1. 对 last 帧做连通域分割（得到 last['cluster']）
        2. 计算 last->next 的光流
        3. 用光流分割 last['cluster']（区分方向不同的运动物体）
        4. last['cluster'] 转化为 last['ob']
        5. ID 继承（用 pre 和 last 匹配）
        6. 对 next 帧做连通域分割（得到 next['cluster']）
        7. next['cluster'] 转化为 next['ob']
        8. 速度更新（用 last 和 next 匹配计算速度）
        
        输出：self.next['ob']（最终输出）
        """
        # 对 last 帧做连通域分割
        if self.last['frame'] is not None:
            self._segment_connected_components('last')
            
            # 计算光流（last -> next）
            self._calc_optical_flow()
            
            # 用光流分割 last['cluster']
            if self.last['cluster'] is not None:
                self._split_by_flow_direction()
            
            # last['cluster'] 转化为 last['ob']
            self.last['ob'] = self._cluster_to_ob(self.last['cluster'])
            
            # ID 继承（用 pre 和 last 匹配）
            self._predict_and_match_ids()
        
        # 对 next 帧做连通域分割
        if self.next['frame'] is not None:
            self._segment_connected_components('next')
            
            # next['cluster'] 转化为 next['ob']
            self.next['ob'] = self._cluster_to_ob(self.next['cluster'])
            
            # 更新速度信息（用 last 和 next 匹配）
            self._update_ob_velocity()


    def _run(self):
        """
        主运行循环：控制处理帧率上限，串联完整工作流
        """
        last_time = time.time()
        min_interval = 1.0 / self.fps

        while self.is_life:
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= min_interval:
                self._dataup()
                self._reco()
                if self.next['frame'] is not None:
                    result = Result(
                        timestamp=self.next['frame'].timestamp,
                        objects=self.next['ob']
                    )
                    self.pre_result = result
                    if self.pre_result is not None:
                        try:
                            self.out_queue.put(self.pre_result, block=False)
                        except:
                            pass

                last_time = current_time
            else:
                time.sleep(min_interval - elapsed)


    def stop(self):
        self.is_life = False