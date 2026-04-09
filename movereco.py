import cv2
import numpy as np
import threading
import time

class Move_reco:
    # 光流法，虽然识别的是上一帧的！！！！但是做轨迹预测图不差这点误差（确信
    def __init__(self, path, pre_queue, out_queue, fps=30, angle=15, cluster=15):
        # path: 模型地址; pre_queue: 输入队列(识别图); out_queue: 输出队列(结果); fps: 最高处理帧率
        self.path = path
        self.pre_queue = pre_queue
        self.out_queue = out_queue
        self.fps = fps
        self.angle = angle          # 方向聚类夹角阈值（度）
        self.cluster = cluster      # 位置聚类距离阈值（像素）
        self.is_life = True
        self.pre_result = None
        self.pre_frame = None       # 缓存上一帧
        self.last_frame = None      # 缓存这一帧

    def start(self):
        run_thread = threading.Thread(target=self._run, daemon=True)
        run_thread.start()
        return run_thread

    def remove(self):
        """从队列取一帧，预处理"""
        if self.last_frame is not None:
            self.pre_frame = self.last_frame

        frame = self.pre_queue.get()
        img = frame.img
        # 转化为灰度图，每个元素除以255，float32
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # 将元素 < threshold 的值变为0（注意：threshold 参数你需要自己传入）
        threshold = 0.5  # 示例值，你可以改成 self.threshold
        gray[gray < threshold] = 0
        self.last_frame = frame
        self.last_frame.img = gray

    def _compute_optical_flow(self, pre_img, last_img):
        """计算光流场"""
        # pre_img 和 last_img 已经是 float32 灰度图
        flow = cv2.calcOpticalFlowFarneback(
            pre_img, last_img, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow

    def _extract_motion_points(self, flow, template):
        """
        从光流场中提取运动点
        返回: [(angle, (x, y), (vx, vy)), ...]
        """
        v = []
        h, w = flow.shape[:2]
        for y in range(h):
            for x in range(w):
                if template[y, x] != 0:  # 模板中不为0的位置才考虑
                    vx, vy = flow[y, x]
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed < 0.5:  # 忽略运动太小的点（你可以把阈值设为参数）
                        continue
                    angle = np.arctan2(vy, vx) * 180 / np.pi
                    if angle < 0:
                        angle += 360
                    v.append([angle, (x, y), (vx, vy)])
        return v

    def _angle_clustering(self, points):
        """方向聚类：按角度分组"""
        if not points:
            return []
        # 按角度排序
        points.sort(key=lambda p: p[0])
        angle_diff = lambda a, b: min(abs(a - b), 360 - abs(a - b))

        clusters = []
        cur_cluster = [points[0]]
        for i in range(1, len(points)):
            if angle_diff(points[i][0], points[i-1][0]) > self.angle:
                clusters.append(cur_cluster)
                cur_cluster = [points[i]]
            else:
                cur_cluster.append(points[i])
        if cur_cluster:
            clusters.append(cur_cluster)
        return clusters

    def _position_clustering(self, angle_clusters):
        """位置聚类：在每个方向组内按空间距离分组"""
        moves = []
        for group in angle_clusters:
            if len(group) < 3:
                continue
            # 按坐标排序（先x后y）
            group.sort(key=lambda p: (p[1][0], p[1][1]))
            cur_move = [group[0]]
            for i in range(1, len(group)):
                x1, y1 = group[i-1][1]
                x2, y2 = group[i][1]
                dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
                if dist > self.cluster:
                    if len(cur_move) >= 3:
                        moves.append(cur_move)
                    cur_move = [group[i]]
                else:
                    cur_move.append(group[i])
            if len(cur_move) >= 3:
                moves.append(cur_move)
        return moves

    def _extract_targets(self, moves):
        """整形：为每个目标计算外接圆和平均运动矢量"""
        ob = []
        for move in moves:
            xs = [p[1][0] for p in move]
            ys = [p[1][1] for p in move]
            vxs = [p[2][0] for p in move]
            vys = [p[2][1] for p in move]

            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            radius = max(((x - center_x)**2 + (y - center_y)**2) ** 0.5 for x, y in zip(xs, ys))
            mean_vx = sum(vxs) / len(vxs)
            mean_vy = sum(vys) / len(vys)

            ob.append({
                'center': (center_x, center_y),
                'radius': radius,
                'velocity': (mean_vx, mean_vy),
                'point_count': len(move)
            })
        return ob

    def _predict_trajectory(self, targets, predict_steps=100):
        """预测图：根据外接圆参数，按平均方向移动N步，圆边压到的格子都算"""
        predictions = []
        for target in targets:
            cx, cy = target['center']
            vx, vy = target['velocity']
            r = target['radius']

            speed = (vx**2 + vy**2) ** 0.5
            if speed < 0.01:
                continue

            dx = vx / speed
            dy = vy / speed

            for step in range(1, predict_steps + 1):
                pred_cx = cx + dx * speed * step
                pred_cy = cy + dy * speed * step
                min_x = int(pred_cx - r)
                max_x = int(pred_cx + r) + 1
                min_y = int(pred_cy - r)
                max_y = int(pred_cy + r) + 1

                predictions.append({
                    'step': step,
                    'center': (pred_cx, pred_cy),
                    'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                    'radius': r,
                    'velocity': (vx, vy)
                })
        return predictions

    def reco(self):
        """识别主函数：拆分成多个子步骤"""
        if self.pre_frame is None:
            return

        timestamp = self.last_frame.timestamp
        pre_img = self.pre_frame.img
        last_img = self.last_frame.img

        # 模板：pre_img 中不为0的位置标记为1
        template = (pre_img > 0).astype(np.uint8)

        # 1. 光流法
        flow = self._compute_optical_flow(pre_img, last_img)

        # 2. 提取运动点
        points = self._extract_motion_points(flow, template)

        # 3. 方向聚类
        angle_clusters = self._angle_clustering(points)

        # 4. 位置聚类
        moves = self._position_clustering(angle_clusters)

        # 5. 整形为外接圆
        targets = self._extract_targets(moves)

        # 6. 预测轨迹
        predictions = self._predict_trajectory(targets, predict_steps=100)

        # 保存结果
        self.pre_result = {
            'timestamp': timestamp,
            'targets': targets,
            'predictions': predictions
        }

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
                self.remove()
                self.reco()
                self.pull()
                last_time = current_time
            else:
                time.sleep(min_interval - elapsed)